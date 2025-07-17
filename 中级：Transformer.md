## Transformer
- 重要性：Transformer 是现代自然语言处理（NLP）的核心，驱动了 ChatGPT 等大模型，代表深度学习的前沿。
- 核心概念：
Transformer 使用“注意力机制”（Attention），关注输入中最重要的部分（如句子中的关键单词）。  
比 RNN 更高效，适合处理长序列。  
- 应用：聊天机器人（如 Grok）、机器翻译、文本生成。
 为什么教：Transformer 代表 AI 的最新进展。

## 一个只有编码器的Transformer
<img width="753" height="263" alt="image" src="https://github.com/user-attachments/assets/a9203ca4-71c3-4184-89db-8b2b551d0042" />  

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SimpleTransformer(nn.Module):
    """简单的Transformer模型"""
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, num_classes=2, max_len=100):
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # 嵌入层
        x = self.embedding(x) * math.sqrt(self.d_model)  # 缩放嵌入
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # 分类
        x = self.classifier(x)
        return x

class SimpleDataset(Dataset):
    """简单的数据集类"""
    def __init__(self, sequences, labels, vocab_size=1000, max_len=50):
        self.sequences = sequences
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # 简单的tokenization（这里用随机数模拟）
        if isinstance(sequence, str):
            # 如果是字符串，转换为数字序列
            tokens = [ord(c) % self.vocab_size for c in sequence[:self.max_len]]
        else:
            # 确保tokens是列表格式
            tokens = list(sequence[:self.max_len])
        
        # 填充或截断到固定长度
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def generate_synthetic_data(num_samples=1000, seq_len=30, vocab_size=1000, num_classes=2):
    """生成合成数据用于演示"""
    np.random.seed(42)
    
    sequences = []
    labels = []
    
    for i in range(num_samples):
        # 生成随机序列
        seq = np.random.randint(1, vocab_size, seq_len)
        sequences.append(seq)
        
        # 更复杂的标签规则：基于序列的多个特征
        # 1. 序列中特定数字的出现频率
        freq_1 = np.sum(seq == 1) / seq_len
        freq_2 = np.sum(seq == 2) / seq_len
        freq_3 = np.sum(seq == 3) / seq_len
        
        # 2. 序列的方差（复杂度）
        variance = np.var(seq)
        
        # 3. 序列中连续相同数字的最大长度
        max_consecutive = 1
        current_consecutive = 1
        for j in range(1, len(seq)):
            if seq[j] == seq[j-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # 综合多个特征决定标签
        score = (freq_1 * 0.3 + freq_2 * 0.2 + freq_3 * 0.1 + 
                (variance / 1000) * 0.2 + (max_consecutive / seq_len) * 0.2)
        
        label = 1 if score > 0.5 else 0
        labels.append(label)
    
    return sequences, labels

def train_model(model, train_loader, val_loader, num_epochs=15, device='cpu'):
    """训练模型"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 降低学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # 调整调度器
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
        print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')
        print()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2, marker='o', markersize=4)
    ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue', linewidth=2, marker='o', markersize=4)
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict(model, sequences, device='cpu'):
    """使用训练好的模型进行预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequence in sequences:
            # 处理单个序列
            if isinstance(sequence, str):
                tokens = [ord(c) % 1000 for c in sequence[:50]]
            else:
                # 确保tokens是列表格式
                tokens = list(sequence[:50])
            
            # 填充到固定长度
            if len(tokens) < 50:
                tokens = tokens + [0] * (50 - len(tokens))
            else:
                tokens = tokens[:50]
            
            # 转换为tensor
            x = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # 预测
            output = model(x)
            prob = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = prob.max().item()
            
            predictions.append({
                'sequence': sequence,
                'predicted_class': pred_class,
                'confidence': confidence,
                'probabilities': prob.cpu().numpy()[0]
            })
    
    return predictions

def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成合成数据
    print("生成合成数据...")
    sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)
    
    # 划分训练集和验证集
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]
    val_sequences = sequences[split_idx:]
    val_labels = labels[split_idx:]
    
    # 创建数据集和数据加载器
    train_dataset = SimpleDataset(train_sequences, train_labels)
    val_dataset = SimpleDataset(val_sequences, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = SimpleTransformer(
        vocab_size=1000,
        d_model=64,  # 减小模型大小
        nhead=4,     # 减少注意力头数
        num_layers=2,
        num_classes=2,
        max_len=50
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=15, device=device  # 增加训练轮数
    )
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 保存模型
    torch.save(model.state_dict(), 'simple_transformer_model.pth')
    print("模型已保存到: simple_transformer_model.pth")
    
    # 测试预测
    print("\n测试预测功能...")
    test_sequences = [
        "这是一个测试序列",
        "另一个测试序列",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]
    
    predictions = predict(model, test_sequences, device=device)
    
    for pred in predictions:
        print(f"序列: {pred['sequence']}")
        print(f"预测类别: {pred['predicted_class']}")
        print(f"置信度: {pred['confidence']:.4f}")
        print(f"类别概率: {pred['probabilities']}")
        print()

if __name__ == "__main__":
    main()

```
## 注释
```
# 生成2000个样本
sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)

# 80%训练，20%验证
split_idx = int(0.8 * len(sequences))
train_sequences = sequences[:split_idx]  # 1600个
val_sequences = sequences[split_idx:]    # 400个
```
```
for epoch in range(15):
    # 训练阶段
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            output = model(data)
            # 计算验证损失和准确率
```
