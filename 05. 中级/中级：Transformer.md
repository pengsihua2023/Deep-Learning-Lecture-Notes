## Transformer
<div align="center">
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/8d064b02-6166-47ec-bfc6-fb031f94192c" />  
</div>

- 重要性：Transformer 是现代自然语言处理（NLP）的核心，驱动了 ChatGPT 等大模型，代表深度学习的前沿。
- 核心概念：
Transformer 使用“注意力机制”（Attention），关注输入中最重要的部分（如句子中的关键单词）。  
比 RNN 更高效，适合处理长序列。  
- 应用：聊天机器人（如 Grok）、机器翻译、文本生成。
 为什么教：Transformer 代表 AI 的最新进展。


## Transformer的数学描述
Transformer架构是自然语言处理和深度学习领域的核心模型，最初由Vaswani等人在2017年论文《Attention is All You Need》中提出。以下是其数学描述，涵盖主要组成部分，包括输入表示、注意力机制、位置编码、前馈网络和层归一化等。    
### 1. 整体架构  
Transformer由编码器（Encoder）和解码器（Decoder）组成，每部分包含多个堆叠的层（通常是 $ N $ 层）。编码器处理输入序列，解码器生成输出序列。核心创新是自注意力机制（Self-Attention），取代了传统循环神经网络（RNN）的序列处理方式。  
输入表示  
输入序列（如单词或标记）首先被转换为向量表示：
<img width="1390" height="474" alt="image" src="https://github.com/user-attachments/assets/fb62a0e3-59d9-41e0-b322-d4d5c2c06904" />


* **词嵌入**：将每个词映射为固定维度的向量 $x_i \in \mathbb{R}^d$，通常通过嵌入矩阵
  $E \in \mathbb{R}^{|V|\times d}$ 实现，其中 $|V|$ 是词汇表大小， $d$ 是嵌入维度。

* **位置编码**：由于 Transformer 不具备序列顺序信息，需加入位置编码（Positional Encoding）以捕捉词的位置。
  位置编码 $PE$ 可通过固定公式生成：

  $$
  PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad 
  PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
  $$

你说得很对 👌
我来帮你修正位置编码公式的 **LaTeX 表达**。

在原始 Transformer 论文中，位置编码公式是这样的：

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{\frac{2i}{d}}}\right), 
\quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)
$$

---

### 完整 LaTeX 代码（可直接放进文档里）

```latex
\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right), 
\quad 
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)
\]
```

这样就和公式中 **分母指数部分要有分数 $\frac{2i}{d}$** 一致了，而不是之前写的 $10000^{2i/d}$ 那种歧义形式。

要不要我帮你把 **词嵌入 + 位置编码 + 最终输入** 三个公式整理成一个完整的 LaTeX 小节？


  其中 $pos$ 是词在序列中的位置， $i$ 是维度索引。最终输入为：

 $$
x_i = E_i + PE(pos_i)
$$

其中：

* $E_i$ 表示词 $w_i$ 的嵌入向量（即从嵌入矩阵 $E$ 中查到的结果）；
* $PE(pos_i)$ 表示位置编码向量；
* 两者逐元素相加后作为 Transformer 的输入。







- 编码器  
每个编码器层包含两个主要子模块： 

多头自注意力机制（Multi-Head Self-Attention）  
前馈神经网络（Feed-Forward Neural Network, FFN）  

每个子模块后接残差连接（Residual Connection）和层归一化（Layer Normalization）。  
解码器  
解码器与编码器类似，但多了掩码自注意力（Masked Self-Attention，用于避免未来信息的泄露）和编码器-解码器注意力（Encoder-Decoder Attention）。  

### 2. 多头自注意力机制 
自注意力是Transformer的核心，允许模型在处理每个词时关注序列中的其他词。  
缩放点积注意力（Scaled Dot-Product Attention）
<img width="1109" height="644" alt="image" src="https://github.com/user-attachments/assets/0ebc34f6-2940-498e-a1f2-541c907b8197" />
多头机制
<img width="1180" height="634" alt="image" src="https://github.com/user-attachments/assets/8f46e5e3-645e-4d3a-9fad-2fa525721a79" />
### 3. 前馈神经网络（FFN） 
每个编码器和解码器层包含一个逐位置的前馈网络，应用于每个词的向量：

$$
\mathrm{FFN}(x) = \mathrm{ReLU}(x W_1 + b_1) W_2 + b_2
$$  

其中 $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$， $d_{ff}$ 通常远大于 $d$（如 $d_{ff} = 4d$）。

  
### 4. 残差连接与层归一化  
每个子模块（自注意力或FFN）后接残差连接和层归一化：  


$$
y = \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
$$  

其中 Sublayer 是注意力或 FFN，LayerNorm 定义为：  

$$
\mathrm{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$  

$\mu$ 和 $\sigma^2$ 是输入向量的均值和方差， $\gamma, \beta$ 是可学习参数。


### 5. 编码器-解码器注意力  
解码器中的额外注意力层使用编码器的输出K, V和解码器的Q：  
  
   

$\mathrm{Attention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})$
  
这允许解码器关注输入序列的上下文。  
 
### 6. 输出层


解码器最后一层通过线性变换和 softmax 生成输出概率：  

$$
P(y_i) = \mathrm{softmax}(z W_{\text{out}} + b_{\text{out}})
$$  

其中 $z$ 是解码器最后一层的输出， $W_{\text{out}} \in \mathbb{R}^{d \times |V|}$。





### 7. 输出层
解码器最后一层通过线性变换和softmax生成输出概率：


$P(y_i) = \mathrm{softmax}(z W_{\text{out}} + b_{\text{out}})$  

其中 $z$ 是解码器最后一层的输出， $W_{\text{out}} \in \mathbb{R}^{d \times |V|}$。
  
### 8. 损失函数 
训练时通常使用交叉熵损失，优化目标是最大化正确输出序列的概率：


$\mathcal{L} = -\sum_{i=1}^{T} \log P(y_i \mid y_{<i}, X)$  

其中 $T$ 是输出序列长度， $y_{<i}$ 是已生成的词。

### 9. 总结  
Transformer的数学核心在于：   

自注意力：通过Q, K, V捕捉序列内关系。  
多头机制：并行捕捉多种语义关系。  
位置编码：弥补序列顺序信息。  
残差与归一化：稳定训练并加速收敛。  

## 一个只有编码器的Transformer

**完整的Transformer vs 编码器Transformer**

---

### 表格内容

| 组件   | 完整Transformer | 编码器Transformer |
| ---- | ------------- | -------------- |
| 编码器  | ✅             | ✅              |
| 解码器  | ✅             | ❌              |
| 适用任务 | 翻译、摘要         | 分类、情感分析        |
| 输入输出 | 序列→序列         | 序列→类别          |

---



```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

# 设置matplotlib中文字体，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class PositionalEncoding(nn.Module):
    """位置编码层，为序列添加位置信息"""
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        
        参数:
            d_model (int): 模型维度
            max_len (int): 最大序列长度（默认5000）
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵，形状为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引，形状为(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项，用于正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码，分别填充到偶数和奇数列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 添加批次维度并转置，形状变为(max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 将位置编码注册为buffer（不参与梯度更新）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播，添加位置编码
        
        参数:
            x (torch.Tensor): 输入张量，形状为(seq_len, batch_size, d_model)
        
        返回:
            torch.Tensor: 添加位置编码后的张量，形状不变
        """
        return x + self.pe[:x.size(0), :]

class SimpleTransformer(nn.Module):
    """简单的Transformer模型，用于序列分类任务"""
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, num_classes=2, max_len=100):
        """
        初始化SimpleTransformer模型
        
        参数:
            vocab_size (int): 词汇表大小
            d_model (int): 模型维度（默认64）
            nhead (int): 多头注意力机制的头数（默认4）
            num_layers (int): Transformer编码器层数（默认2）
            num_classes (int): 分类类别数（默认2）
            max_len (int): 最大序列长度（默认100）
        """
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model  # 保存模型维度，用于后续缩放
        # 词嵌入层，将词索引转换为d_model维向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层，添加序列位置信息
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 定义单个Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 模型维度
            nhead=nhead,  # 注意力头数
            dim_feedforward=d_model * 4,  # 前馈网络隐藏层维度（通常为d_model的4倍）
            dropout=0.1,  # dropout概率，防止过拟合
            batch_first=True  # 输入格式为(batch_size, seq_len, d_model)
        )
        # 堆叠多个Transformer编码器层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头，将Transformer输出映射到分类结果
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 线性层，降维
            nn.ReLU(),  # ReLU激活函数，增加非线性
            nn.Dropout(0.1),  # dropout层，防止过拟合
            nn.Linear(d_model // 2, num_classes)  # 输出分类结果
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(batch_size, seq_len)
        
        返回:
            torch.Tensor: 分类输出，形状为(batch_size, num_classes)
        """
        batch_size, seq_len = x.shape
        
        # 词嵌入并缩放（乘以sqrt(d_model)以稳定训练）
        x = self.embedding(x) * math.sqrt(self.d_model)  # 形状: (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = x.transpose(0, 1)  # 转换为(seq_len, batch_size, d_model)
        x = self.pos_encoding(x)  # 添加位置编码
        x = x.transpose(0, 1)  # 转换回(batch_size, seq_len, d_model)
        
        # 通过Transformer编码器处理序列
        x = self.transformer_encoder(x)  # 形状: (batch_size, seq_len, d_model)
        
        # 全局平均池化，得到固定长度表示
        x = x.mean(dim=1)  # 形状: (batch_size, d_model)
        
        # 通过分类头输出分类结果
        x = self.classifier(x)  # 形状: (batch_size, num_classes)
        return x

class SimpleDataset(Dataset):
    """简单的数据集类，用于处理序列数据"""
    def __init__(self, sequences, labels, vocab_size=1000, max_len=50):
        """
        初始化数据集
        
        参数:
            sequences (list): 输入序列列表（可以是字符串或数字列表）
            labels (list): 标签列表
            vocab_size (int): 词汇表大小（默认1000）
            max_len (int): 最大序列长度（默认50）
        """
        self.sequences = sequences
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_len = max_len
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx (int): 样本索引
        
        返回:
            tuple: (序列张量, 标签张量)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # 将字符串序列转换为数字序列
        if isinstance(sequence, str):
            tokens = [ord(c) % self.vocab_size for c in sequence[:self.max_len]]
        else:
            tokens = list(sequence[:self.max_len])
        
        # 填充或截断序列到固定长度
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def generate_synthetic_data(num_samples=1000, seq_len=30, vocab_size=1000, num_classes=2):
    """生成合成数据用于演示"""
    np.random.seed(42)  # 设置随机种子以确保可重复性
    
    sequences = []
    labels = []
    
    for i in range(num_samples):
        # 生成随机序列
        seq = np.random.randint(1, vocab_size, seq_len)
        sequences.append(seq)
        
        # 计算序列特征用于生成标签
        freq_1 = np.sum(seq == 1) / seq_len  # 数字1的频率
        freq_2 = np.sum(seq == 2) / seq_len  # 数字2的频率
        freq_3 = np.sum(seq == 3) / seq_len  # 数字3的频率
        variance = np.var(seq)  # 序列方差
        # 计算最大连续相同数字长度
        max_consecutive = 1
        current_consecutive = 1
        for j in range(1, len(seq)):
            if seq[j] == seq[j-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # 综合特征生成标签
        score = (freq_1 * 0.3 + freq_2 * 0.2 + freq_3 * 0.1 + 
                (variance / 1000) * 0.2 + (max_consecutive / seq_len) * 0.2)
        label = 1 if score > 0.5 else 0
        labels.append(label)
    
    return sequences, labels

def train_model(model, train_loader, val_loader, num_epochs=15, device='cpu'):
    """训练模型"""
    model.to(device)  # 将模型移动到指定设备
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam优化器，学习率0.0005
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # 学习率调度器，每5个epoch降低学习率
    
    # 初始化记录列表
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
            data, target = data.to(device), target.to(device)  # 移动数据到设备
            optimizer.zero_grad()  # 清空梯度
            output = model(data)  # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            train_loss += loss.item()
            _, predicted = output.max(1)  # 获取预测类别
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)  # 计算平均损失
        train_accuracy = 100. * train_correct / train_total  # 计算准确率
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # 禁用梯度计算
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
        
        scheduler.step()  # 更新学习率
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # 打印训练进度
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
        print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')
        print()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """绘制训练和验证的损失及准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue', linewidth=2, marker='o', markersize=4)
    ax1.plot(val_losses, label='验证损失', color='red', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 绘制准确率曲线
    ax2.plot(train_accuracies, label='训练准确率', color='blue', linewidth=2, marker='o', markersize=4)
    ax2.plot(val_accuracies, label='验证准确率', color='red', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('轮次', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')  # 保存图像
    plt.show()

def predict(model, sequences, device='cpu'):
    """使用训练好的模型进行预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequence in sequences:
            # 处理单个序列
            if isinstance(sequence, str):
                tokens = [ord(c) % 1000 for c in sequence[:50]]  # 字符串转换为数字序列
            else:
                tokens = list(sequence[:50])  # 确保为列表格式
            
            # 填充或截断序列
            if len(tokens) < 50:
                tokens = tokens + [0] * (50 - len(tokens))
            else:
                tokens = tokens[:50]
            
            # 转换为张量
            x = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # 进行预测
            output = model(x)
            prob = F.softmax(output, dim=1)  # 计算类别概率
            pred_class = output.argmax(dim=1).item()  # 获取预测类别
            confidence = prob.max().item()  # 获取最大概率
            
            predictions.append({
                'sequence': sequence,
                'predicted_class': pred_class,
                'confidence': confidence,
                'probabilities': prob.cpu().numpy()[0]
            })
    
    return predictions

def main():
    """主函数，执行数据生成、模型训练和预测"""
    # 设置设备（优先使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成合成数据
    print("生成合成数据...")
    sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)
    
    # 划分训练集和验证集（80%训练，20%验证）
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
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        max_len=50
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=15, device=device
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
## 以下是对完整代码的详细中文注释，涵盖了每个类和函数的功能、参数及实现逻辑：

```python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

# 设置matplotlib中文字体，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class PositionalEncoding(nn.Module):
    """位置编码层，为序列添加位置信息"""
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        
        参数:
            d_model (int): 模型维度
            max_len (int): 最大序列长度（默认5000）
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵，形状为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引，形状为(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项，用于正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码，分别填充到偶数和奇数列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 添加批次维度并转置，形状变为(max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 将位置编码注册为buffer（不参与梯度更新）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播，添加位置编码
        
        参数:
            x (torch.Tensor): 输入张量，形状为(seq_len, batch_size, d_model)
        
        返回:
            torch.Tensor: 添加位置编码后的张量，形状不变
        """
        return x + self.pe[:x.size(0), :]

class SimpleTransformer(nn.Module):
    """简单的Transformer模型，用于序列分类任务"""
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, num_classes=2, max_len=100):
        """
        初始化SimpleTransformer模型
        
        参数:
            vocab_size (int): 词汇表大小
            d_model (int): 模型维度（默认64）
            nhead (int): 多头注意力机制的头数（默认4）
            num_layers (int): Transformer编码器层数（默认2）
            num_classes (int): 分类类别数（默认2）
            max_len (int): 最大序列长度（默认100）
        """
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model  # 保存模型维度，用于后续缩放
        # 词嵌入层，将词索引转换为d_model维向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层，添加序列位置信息
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 定义单个Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 模型维度
            nhead=nhead,  # 注意力头数
            dim_feedforward=d_model * 4,  # 前馈网络隐藏层维度（通常为d_model的4倍）
            dropout=0.1,  # dropout概率，防止过拟合
            batch_first=True  # 输入格式为(batch_size, seq_len, d_model)
        )
        # 堆叠多个Transformer编码器层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头，将Transformer输出映射到分类结果
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 线性层，降维
            nn.ReLU(),  # ReLU激活函数，增加非线性
            nn.Dropout(0.1),  # dropout层，防止过拟合
            nn.Linear(d_model // 2, num_classes)  # 输出分类结果
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(batch_size, seq_len)
        
        返回:
            torch.Tensor: 分类输出，形状为(batch_size, num_classes)
        """
        batch_size, seq_len = x.shape
        
        # 词嵌入并缩放（乘以sqrt(d_model)以稳定训练）
        x = self.embedding(x) * math.sqrt(self.d_model)  # 形状: (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = x.transpose(0, 1)  # 转换为(seq_len, batch_size, d_model)
        x = self.pos_encoding(x)  # 添加位置编码
        x = x.transpose(0, 1)  # 转换回(batch_size, seq_len, d_model)
        
        # 通过Transformer编码器处理序列
        x = self.transformer_encoder(x)  # 形状: (batch_size, seq_len, d_model)
        
        # 全局平均池化，得到固定长度表示
        x = x.mean(dim=1)  # 形状: (batch_size, d_model)
        
        # 通过分类头输出分类结果
        x = self.classifier(x)  # 形状: (batch_size, num_classes)
        return x

class SimpleDataset(Dataset):
    """简单的数据集类，用于处理序列数据"""
    def __init__(self, sequences, labels, vocab_size=1000, max_len=50):
        """
        初始化数据集
        
        参数:
            sequences (list): 输入序列列表（可以是字符串或数字列表）
            labels (list): 标签列表
            vocab_size (int): 词汇表大小（默认1000）
            max_len (int): 最大序列长度（默认50）
        """
        self.sequences = sequences
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_len = max_len
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx (int): 样本索引
        
        返回:
            tuple: (序列张量, 标签张量)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # 将字符串序列转换为数字序列
        if isinstance(sequence, str):
            tokens = [ord(c) % self.vocab_size for c in sequence[:self.max_len]]
        else:
            tokens = list(sequence[:self.max_len])
        
        # 填充或截断序列到固定长度
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def generate_synthetic_data(num_samples=1000, seq_len=30, vocab_size=1000, num_classes=2):
    """生成合成数据用于演示"""
    np.random.seed(42)  # 设置随机种子以确保可重复性
    
    sequences = []
    labels = []
    
    for i in range(num_samples):
        # 生成随机序列
        seq = np.random.randint(1, vocab_size, seq_len)
        sequences.append(seq)
        
        # 计算序列特征用于生成标签
        freq_1 = np.sum(seq == 1) / seq_len  # 数字1的频率
        freq_2 = np.sum(seq == 2) / seq_len  # 数字2的频率
        freq_3 = np.sum(seq == 3) / seq_len  # 数字3的频率
        variance = np.var(seq)  # 序列方差
        # 计算最大连续相同数字长度
        max_consecutive = 1
        current_consecutive = 1
        for j in range(1, len(seq)):
            if seq[j] == seq[j-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # 综合特征生成标签
        score = (freq_1 * 0.3 + freq_2 * 0.2 + freq_3 * 0.1 + 
                (variance / 1000) * 0.2 + (max_consecutive / seq_len) * 0.2)
        label = 1 if score > 0.5 else 0
        labels.append(label)
    
    return sequences, labels

def train_model(model, train_loader, val_loader, num_epochs=15, device='cpu'):
    """训练模型"""
    model.to(device)  # 将模型移动到指定设备
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam优化器，学习率0.0005
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # 学习率调度器，每5个epoch降低学习率
    
    # 初始化记录列表
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
            data, target = data.to(device), target.to(device)  # 移动数据到设备
            optimizer.zero_grad()  # 清空梯度
            output = model(data)  # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            train_loss += loss.item()
            _, predicted = output.max(1)  # 获取预测类别
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)  # 计算平均损失
        train_accuracy = 100. * train_correct / train_total  # 计算准确率
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # 禁用梯度计算
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
        
        scheduler.step()  # 更新学习率
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # 打印训练进度
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
        print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')
        print()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """绘制训练和验证的损失及准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue', linewidth=2, marker='o', markersize=4)
    ax1.plot(val_losses, label='验证损失', color='red', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 绘制准确率曲线
    ax2.plot(train_accuracies, label='训练准确率', color='blue', linewidth=2, marker='o', markersize=4)
    ax2.plot(val_accuracies, label='验证准确率', color='red', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('轮次', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')  # 保存图像
    plt.show()

def predict(model, sequences, device='cpu'):
    """使用训练好的模型进行预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequence in sequences:
            # 处理单个序列
            if isinstance(sequence, str):
                tokens = [ord(c) % 1000 for c in sequence[:50]]  # 字符串转换为数字序列
            else:
                tokens = list(sequence[:50])  # 确保为列表格式
            
            # 填充或截断序列
            if len(tokens) < 50:
                tokens = tokens + [0] * (50 - len(tokens))
            else:
                tokens = tokens[:50]
            
            # 转换为张量
            x = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # 进行预测
            output = model(x)
            prob = F.softmax(output, dim=1)  # 计算类别概率
            pred_class = output.argmax(dim=1).item()  # 获取预测类别
            confidence = prob.max().item()  # 获取最大概率
            
            predictions.append({
                'sequence': sequence,
                'predicted_class': pred_class,
                'confidence': confidence,
                'probabilities': prob.cpu().numpy()[0]
            })
    
    return predictions

def main():
    """主函数，执行数据生成、模型训练和预测"""
    # 设置设备（优先使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成合成数据
    print("生成合成数据...")
    sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)
    
    # 划分训练集和验证集（80%训练，20%验证）
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
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        max_len=50
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=15, device=device
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

### 代码总体说明：
1. **代码功能**：
   - 这是一个完整的PyTorch实现，用于基于Transformer的序列分类任务。
   - 包括位置编码、Transformer模型、数据集处理、数据生成、模型训练、结果可视化和预测功能。

2. **主要组件**：
   - **PositionalEncoding**：实现经典的正弦/余弦位置编码，为序列添加位置信息。
   - **SimpleTransformer**：一个简单的Transformer模型，包含词嵌入、位置编码、Transformer编码器和分类头。
   - **SimpleDataset**：自定义数据集类，支持字符串和数字序列，处理填充和截断。
   - **generate_synthetic_data**：生成合成数据，基于序列特征（如频率、方差、连续性）生成标签。
   - **train_model**：训练模型，记录损失和准确率，使用Adam优化器和学习率调度。
   - **plot_training_curves**：绘制训练和验证的损失及准确率曲线。
   - **predict**：对新序列进行预测，返回类别、置信度和概率。
   - **main**：主函数，协调数据生成、训练、绘图和预测。

3. **使用场景**：
   - 适用于序列分类任务的快速原型开发。
   - 合成数据用于演示，实际应用可替换为真实数据集。
   - 可通过调整模型参数（如`d_model`、`nhead`等）优化性能。

## 注释
### 1. 数据准备
```
# 生成2000个样本
sequences, labels = generate_synthetic_data(num_samples=2000, seq_len=30)

# 80%训练，20%验证
split_idx = int(0.8 * len(sequences))
train_sequences = sequences[:split_idx]  # 1600个
val_sequences = sequences[split_idx:]    # 400个
```

### 2. 训练循环
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
### 3. 损失函数
```
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam优化器
```
### 总结
数据：程序生成的合成数据，基于统计特征分类 
模型：只有编码器的Transformer，用于序列分类  
解码器：没有，因为不需要生成序列输出  
流程：数据生成 → 预处理 → 嵌入 → 位置编码 → 自注意力 → 池化 → 分类  
这个模型适合学习Transformer的基本概念，特别是自注意力机制和位置编码！  
