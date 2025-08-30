## 著名网络架构：Vision Transformer (ViT, 2020)
提出者：Google  
第一作者：Alexey Dosovitskiy   
<div align="center">
<img width="220" height="236" alt="image" src="https://github.com/user-attachments/assets/e8b41203-25a3-4583-978d-46f29ea2f38c" />  
</div>

 Vision Transformer (ViT)，简称 ViT，是由 Alexey Dosovitskiy 等人在 2020 年提出的图像识别模型，发表在论文《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》中（ICLR 2021）。ViT 是 Transformer 架构首次大规模应用于计算机视觉领域的里程碑，将自然语言处理中的 Transformer 直接移植到图像任务中，摒弃了传统卷积神经网络（CNN）的卷积操作，转而使用自注意力机制（Self-Attention）处理图像块（Patches）。ViT 在大规模数据集上表现出色，推动了视觉 Transformer 的发展浪潮，并对后续混合模型（如 Swin Transformer、DETR）产生深远影响。  
    
特点：将图像分块（patch）后用Transformer处理，取代传统CNN，适合大数据集。  
应用：图像分类、目标检测、图像分割。  
掌握要点：自注意力机制、图像分块处理。  
<div align="center">
<img width="320" height="250" alt="image" src="https://github.com/user-attachments/assets/c3611f64-f4e1-4f03-bbd6-efd480e0cc6e" />
</div>

<div align="center">
(此图引自Internet。)
</div>

 
 ## 代码
```
 主要功能
1. Vision Transformer 模型实现
图像分块嵌入: 将32×32图像分割成4×4的小块
位置编码: 为每个patch添加位置信息
多头自注意力: 捕获patch之间的全局依赖关系
Transformer块: 多层编码器结构
分类头: 最终输出10个类别的概率
2. 真实数据集处理
CIFAR-10数据集: 50,000张训练图像 + 10,000张测试图像
自动下载: 首次运行自动下载数据集
数据预处理: 标准化、数据增强（随机翻转）
数据加载: 高效的DataLoader实现
3. 完整训练流程
模型初始化: 权重初始化、设备配置
训练循环: 前向传播、损失计算、反向传播
验证评估: 每个epoch后进行测试集评估
学习率调度: CosineAnnealingLR优化器
进度监控: 实时显示训练进度和性能指标
4. 性能评估系统
准确率计算: 训练和测试准确率
损失监控: 训练损失变化
详细分析: 每个类别的准确率统计
过拟合检测: 训练vs测试准确率对比
5. 可视化功能
训练曲线: 损失和准确率变化趋势
预测可视化: 真实图像与预测结果对比
颜色编码: 绿色=正确预测，红色=错误预测
高质量输出: 保存为PNG格式
```
### Python代码
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class LightVisionTransformer(nn.Module):
    """轻量级Vision Transformer for CIFAR-10"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=64, depth=3, num_heads=4, mlp_ratio=2, dropout=0.1):
        super(LightVisionTransformer, self).__init__()
        
        # 分块嵌入
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 位置编码和分类token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 分块嵌入
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加分类token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # 通过Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 分类
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x

def load_cifar10_data_light():
    """加载CIFAR-10数据集（轻量级版本）"""
    print("Loading CIFAR-10 dataset...")
    
    # 简化的数据预处理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 创建数据加载器（较小的批次大小）
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    # CIFAR-10类别名称
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    return trainloader, testloader, classes

def train_light_vision_transformer():
    """训练轻量级Vision Transformer"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    trainloader, testloader, classes = load_cifar10_data_light()
    
    # 创建模型
    model = LightVisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=3,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.1
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 训练循环
    print("Starting training...")
    num_epochs = 15  # 减少训练轮数
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(trainloader)}, '
                      f'Loss: {loss.item():.4f}')
        
        train_accuracy = 100. * correct / total
        avg_loss = total_loss / len(trainloader)
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        
        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Test Acc: {test_accuracy:.2f}%')
        print('-' * 50)
    
    return model, train_losses, train_accuracies, test_accuracies, classes

def visualize_sample_predictions(model, testloader, classes, device, num_samples=8):
    """可视化样本预测结果"""
    print(f"\nVisualizing {num_samples} sample predictions...")
    model.eval()
    
    # 获取一些测试样本
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 反归一化图像
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    images_denorm = images.cpu() * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images_denorm[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        
        # 颜色编码：绿色=正确，红色=错误
        color = 'green' if predicted[i] == labels[i] else 'red'
        axes[i].set_title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}', 
                         color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_light_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_light_training_curves(train_losses, train_accuracies, test_accuracies):
    """绘制轻量级训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_title('Light Vision Transformer Training Loss (CIFAR-10)', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accuracies, 'b-', linewidth=2, label='Train Accuracy')
    ax2.plot(test_accuracies, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_title('Light Vision Transformer Accuracy (CIFAR-10)', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cifar10_light_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== Light Vision Transformer on CIFAR-10 Dataset ===")
    
    # 训练模型
    model, train_losses, train_accuracies, test_accuracies, classes = train_light_vision_transformer()
    
    # 绘制训练曲线
    plot_light_training_curves(train_losses, train_accuracies, test_accuracies)
    
    # 可视化预测结果
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader, _ = load_cifar10_data_light()
    visualize_sample_predictions(model, testloader, classes, device)
    
    print("\nTraining completed!")
    print("Results saved:")
    print("- cifar10_light_training_curves.png: Training curves")
    print("- cifar10_light_predictions.png: Sample predictions")

if __name__ == "__main__":
    main()

```

## 训练结果

Epoch: 15/15, Batch: 1400/1563, Loss: 0.8820  
Epoch 15/15:  
  Train Loss: 0.8538, Train Acc: 69.65%  
  Test Acc: 67.46%  

Loading CIFAR-10 dataset...  
Training samples: 50000  
Test samples: 10000 

Visualizing 8 sample predictions...  

Training completed!  
Results saved:  
- cifar10_light_training_curves.png: Training curves  
- cifar10_light_predictions.png: Sample predictions  

<img width="1488" height="497" alt="image" src="https://github.com/user-attachments/assets/5a7569bc-82c4-40bc-b626-28462c50bf49" />
  

图2 训练loss和训练accuracy

<img width="1578" height="802" alt="image" src="https://github.com/user-attachments/assets/4e875295-be21-4d4f-969b-d20a5c76ac36" />
 
图3 模型预测结果
