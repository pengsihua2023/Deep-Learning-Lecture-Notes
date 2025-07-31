## 著名网络架构：AlexNet (2012)
提出者：Alex Krizhevsky 等  
<img width="206" height="245" alt="image" src="https://github.com/user-attachments/assets/062a3e51-cb54-4711-adaa-f68671fca005" />  

特点：引入ReLU激活函数、Dropout正则化、数据增强和GPU加速，在ImageNet竞赛中大幅提升性能。  
应用：图像分类、特征提取、迁移学习基础。  
掌握要点：深层CNN设计、过拟合控制。  
<img width="1459" height="775" alt="image" src="https://github.com/user-attachments/assets/5bd0deb5-051a-43ba-95f7-931fcd671b32" />  
图1 AlexNet 架构图 （现代的架构，现在GPU内存很大了，无需分支。）
<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/e27296cb-2aee-4389-a119-7c2ac8120d4d" />  
图2 同图1
<img width="1339" height="503" alt="image" src="https://github.com/user-attachments/assets/7b145c0e-205a-4c61-ad7f-b477203e8db6" />  
 图3 最先的双GPU计算架构。 在AlexNet的原始设计中，第二层卷积（CONV2）到第三层卷积（CONV3）的“交叉”指的是跨GPU连接，即CONV3的内核会从前一层（CONV2）的所有内核地图（kernel maps，包括两个GPU上的）获取输入。这种设计是为了在多GPU并行训练时，确保模型能够捕获更全面的特征信息，同时控制计算开销。原论文中指出，这种连接模式是通过交叉验证（cross-validation）实验选择的，以平衡GPU间的通信量和整体性能——如果所有层都交叉，通信开销会过高，成为训练瓶颈；如果完全无交叉，则模型准确率会下降（实验显示，相比单GPU一半内核的版本，这种设计降低了top-1错误率1.7%和top-5错误率1.2%）。 

### 代码
该代码实现了一个**简化的AlexNet卷积神经网络**，用于在**CIFAR-10数据集**上进行图像分类任务。主要功能如下：

1. **模型定义**：实现了一个适配CIFAR-10的AlexNet模型，包含5层卷积（`features`）和3层全连接层（`classifier`），使用ReLU激活、最大池化和Dropout正则化，输出10类分类结果。

2. **数据预处理**：加载CIFAR-10数据集（训练集和测试集），应用变换（调整大小到32x32、归一化），并使用DataLoader进行批处理（batch_size=64）。

3. **训练过程**：使用SGD优化器（学习率0.001，动量0.9）和交叉熵损失函数，训练模型30个epoch。每200个批次记录平均损失和准确率，并打印损失。

4. **测试过程**：在测试集上评估模型，计算并输出分类准确率。

5. **可视化**：绘制训练过程中的损失曲线，保存为`alexnet_training_curve.png`，支持中文显示（使用SimHei字体）。

代码运行在CPU或GPU上，训练完成后输出测试集准确率并生成损失曲线图，用于分析模型训练效果。
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

# 配置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# 定义 AlexNet（适配 CIFAR-10）
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),  # 适配 3x3x256=2304
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 存储训练指标
train_losses = []
train_accuracies = []

# 训练函数
def train_model(epochs=30):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 200 == 199:
                avg_loss = running_loss / 200
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}')
                train_losses.append(avg_loss)
                train_accuracies.append(100 * correct / total)
                running_loss = 0.0
                correct = 0
                total = 0
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

# 绘制训练损失曲线
def plot_training_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('训练批次 (每200批)')
    plt.ylabel('损失')
    plt.title('简化版AlexNet训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('alexnet_training_curve.png', dpi=300, bbox_inches='tight')
    print('训练损失曲线已保存为: alexnet_training_curve.png')
    plt.close()

# 执行训练、测试和绘图
if __name__ == "__main__":
    train_model(epochs=30)
    test_model()
    plot_training_curve()
```
### 训练结果
[Epoch 29, Batch 600] Loss: 0.421  
[Epoch 30, Batch 200] Loss: 0.378  
[Epoch 30, Batch 400] Loss: 0.396  
[Epoch 30, Batch 600] Loss: 0.409  
训练完成！  
测试集准确率: 77.61%  
训练损失曲线已保存为: alexnet_training_curve.png   
<img width="1239" height="609" alt="image" src="https://github.com/user-attachments/assets/f102aae9-d87d-4f43-bbb1-b3fac9d373b7" />
