## 著名网络架构：EfficientNet (2019)
提出者：Google 
第一作者：Mingxiing Tan  
<img width="256" height="233" alt="image" src="https://github.com/user-attachments/assets/5b194e36-40bf-47ca-a03a-fd24faf436ed" />  

EfficientNet 是一种高效的卷积神经网络（Convolutional Neural Network, CNN）架构，由 Mingxing Tan 和 Quoc V. Le 在 2019 年提出，发表在论文《EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks》中（ICML 2019）。EfficientNet 通过系统化的复合缩放（Compound Scaling）方法，在计算资源有限的情况下实现高精度图像分类，标志着 CNN 设计从手动调参向自动化和高效化的转变。EfficientNet 系列模型（B0 到 B7）在 ImageNet 上取得了 SOTA（State-of-the-Art）性能，同时参数量和计算量远低于 ResNet 和 Inception 等模型，广泛应用于移动设备和边缘计算场景。   
   
特点：通过复合缩放（深度、宽度、分辨率）平衡性能和效率，适合资源受限场景。  
应用：高效图像分类、嵌入式设备。  
掌握要点：模型缩放策略、轻量化设计。  
<img width="600" height="3200" alt="image" src="https://github.com/user-attachments/assets/f88060f1-8ac4-4b2c-8cc7-6186af74255c" />  
## 代码
该代码实现了一个**简化的EfficientNet模型**，用于在**CIFAR-10数据集**上进行图像分类任务。主要功能如下：

1. **模型定义**：
   - 实现了一个基于EfficientNet的`MBConv`模块，包括扩展卷积、深度可分离卷积和压缩卷积，支持残差连接。
   - 定义`SimpleEfficientNet`，包含初始卷积层、多个MBConv块和全局平均池化，输出10类分类结果。

2. **数据预处理**：
   - 加载CIFAR-10数据集（32x32图像），应用归一化变换。
   - 使用DataLoader进行批处理（batch_size=64）。

3. **训练过程**：
   - 使用Adam优化器（学习率0.001）和交叉熵损失函数，训练模型50个epoch。
   - 每200个批次记录并打印平均损失。

4. **测试过程**：
   - 在测试集上评估模型，计算并输出分类准确率。

5. **可视化**：
   - 绘制训练过程中的损失曲线，保存为`efficientnet_training_curve.png`。
   - 从测试集取8张图像，显示预测和真实标签，保存为`efficientnet_predictions.png`，支持中文显示（使用SimHei字体）。

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

# 定义MBConv块（EfficientNet核心组件）
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        mid_channels = in_channels * expand_ratio
        self.use_residual = in_channels == out_channels and stride == 1

        layers = []
        # 扩展卷积（1x1）
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU6(inplace=True))
        
        # 深度可分离卷积
        layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False))
        layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU6(inplace=True))
        
        # 压缩卷积（1x1）
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out

# 定义简化的EfficientNet模型
class SimpleEfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleEfficientNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),  # 32x32
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),  # 16x16
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),  # 8x8
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
        )
        self.classifier = nn.Sequential(
            nn.Linear(40, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 40)
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
model = SimpleEfficientNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    plt.savefig('efficientnet_predictions.png', dpi=300, bbox_inches='tight')
    print('预测结果已保存为: efficientnet_predictions.png')
    plt.close()

# 绘制训练损失曲线
def plot_training_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('训练批次 (每200批)')
    plt.ylabel('损失')
    plt.title('简化版EfficientNet训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('efficientnet_training_curve.png', dpi=300, bbox_inches='tight')
    print('训练损失曲线已保存为: efficientnet_training_curve.png')
    plt.close()

# 执行训练、测试和可视化
if __name__ == "__main__":
    train_model(epochs=50)
    test_model()
    plot_training_curve()
    visualize_predictions()

```

## 训练结果



<img width="1251" height="616" alt="image" src="https://github.com/user-attachments/assets/119877a7-6a80-447d-9326-416810006a9d" />  

图2 损失曲线  

<img width="1359" height="221" alt="image" src="https://github.com/user-attachments/assets/ca55c897-aaa3-457b-9682-2cfb47dba08f" />


图3 预测结果  
