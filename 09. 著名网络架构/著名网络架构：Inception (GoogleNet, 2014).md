## 著名网络架构：Inception (GoogleNet, 2014)
提出者：Google  
第一作者：Christian Szegedy  
<img width="200" height="250" alt="image" src="https://github.com/user-attachments/assets/c4120069-66c0-4625-b257-fc28c310bce6" />   
  
Inception（也称为 GoogLeNet）是由 Christian Szegedy 等人在 2014 年提出的卷积神经网络（Convolutional Neural Network, CNN）架构，发表在论文《Going Deeper with Convolutions》中（CVPR 2015）。Inception 在 2014 年 ImageNet 大规模视觉识别挑战赛（ILSVRC）中夺冠，以其高效的计算性能和深层架构在计算机视觉领域掀起热潮。GoogLeNet 是第一个使用 Inception 模块 的网络，通过多尺度卷积并行处理提升性能，同时保持较低的计算成本。     
   
特点：Inception模块并行处理多尺度卷积，优化计算效率，引入1x1卷积降维。  
应用：图像分类、特征提取。  
掌握要点：多尺度特征提取、计算效率优化。  
<img width="1703" height="490" alt="image" src="https://github.com/user-attachments/assets/7944d7cd-6b8f-4753-a453-38146ed9b160" />  

## 代码
该代码实现了一个**简化的Inception模型**（基于GoogLeNet的Inception架构），用于在**CIFAR-10数据集**上进行图像分类任务。主要功能如下：

1. **模型定义**：
   - 实现`InceptionModule`，包含四个并行分支：1x1卷积、3x3卷积、5x5卷积和池化+1x1卷积，输出拼接以捕获多尺度特征。
   - 定义`SimpleInception`，包括初始卷积层、两个Inception模块、最大池化层和全连接分类器，输出10类分类结果。

2. **数据预处理**：
   - 加载CIFAR-10数据集（32x32图像），应用归一化变换。
   - 使用DataLoader进行批处理（batch_size=64）。

3. **训练过程**：
   - 使用SGD优化器（学习率0.001，动量0.9）和交叉熵损失函数，训练模型50个epoch。
   - 每200个批次记录并打印平均损失。

4. **测试过程**：
   - 在测试集上评估模型，计算并输出分类准确率。

5. **可视化**：
   - 绘制训练过程中的损失曲线，保存为`inception_training_curve.png`。
   - 从测试集取8张图像，显示预测和真实标签，保存为`inception_predictions.png`，支持中文显示（使用SimHei字体）。

代码运行在CPU或GPU上，训练完成后输出测试集准确率，并生成损失曲线和预测结果的可视化图像，用于分析模型性能和分类效果。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

# 配置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# 定义简化的Inception模块
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        # 1x1卷积分支
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        # 3x3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # 池化+1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# 定义简化的Inception模型
class SimpleInception(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleInception, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            InceptionModule(64, 16, 32, 16, 16),  # 输出: 16+32+16+16=80
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            InceptionModule(80, 32, 64, 32, 32),  # 输出: 32+64+32+32=160
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(160 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 160 * 4 * 4)
        x = self.classifier(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleInception(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 存储训练指标
train_losses = []

# 训练函数
def train_model(epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                avg_loss = running_loss / 200
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}')
                train_losses.append(avg_loss)
                running_loss = 0.0
    print('训练完成！')

# 测试函数
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    return accuracy

# 可视化验证结果
def visualize_predictions():
    model.eval()
    images, labels = next(iter(testloader))  # 获取一批测试数据
    images, labels = images[:8].to(device), labels[:8].to(device)  # 取8个样本
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 反归一化图像以便显示
    images = images.cpu() * 0.5 + 0.5  # 还原到[0,1]
    classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].set_title(f'预测: {classes[predicted[i]]}\n真实: {classes[labels[i]]}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('inception_predictions.png', dpi=300, bbox_inches='tight')
    print('预测结果已保存为: inception_predictions.png')
    plt.close()

# 绘制训练损失曲线
def plot_training_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('训练批次 (每200批)')
    plt.ylabel('损失')
    plt.title('简化版Inception训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('inception_training_curve.png', dpi=300, bbox_inches='tight')
    print('训练损失曲线已保存为: inception_training_curve.png')
    plt.close()

# 执行训练、测试和可视化
if __name__ == "__main__":
    train_model(epochs=50)
    test_model()
    plot_training_curve()
    visualize_predictions()
```


## 训练结果
<img width="1253" height="612" alt="image" src="https://github.com/user-attachments/assets/1edf2e07-fda4-4094-a3b7-3786cc7dc393" />  

图2 损失曲线  

<img width="1354" height="221" alt="image" src="https://github.com/user-attachments/assets/eb3a6692-2abd-448f-a01f-1428945e6a62" />     
图3 预测结果  
