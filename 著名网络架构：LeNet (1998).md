## 著名网络架构：LeNet (1998)
提出者：Yann LeCun  
特点：最早的卷积神经网络（CNN），包含卷积层、池化层和全连接层，用于手写数字识别。结构简单，奠定CNN基础。  
应用：简单图像分类（如MNIST数据集）。  
掌握要点：理解卷积操作、池化机制。  

<img width="1416" height="535" alt="image" src="https://github.com/user-attachments/assets/f2ccce70-ad11-40d2-bf71-3651aa4fd10b" />  

## 代码
```
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

# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # 输入1通道，输出6通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),  # 展平后的尺寸
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),  # 输出10类（MNIST）
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 存储训练指标
train_losses = []

# 训练函数
def train_model(epochs=5):
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
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'预测: {predicted[i].item()}\n真实: {labels[i].item()}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('lenet_predictions.png', dpi=300, bbox_inches='tight')
    print('预测结果已保存为: lenet_predictions.png')
    plt.close()

# 绘制训练损失曲线
def plot_training_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('训练批次 (每200批)')
    plt.ylabel('损失')
    plt.title('LeNet训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lenet_training_curve.png', dpi=300, bbox_inches='tight')
    print('训练损失曲线已保存为: lenet_training_curve.png')
    plt.close()

# 执行训练、测试和可视化
if __name__ == "__main__":
    train_model(epochs=5)
    test_model()
    plot_training_curve()
    visualize_predictions()
```
### 训练结果
[Epoch 5, Batch 200] Loss: 0.026  
[Epoch 5, Batch 400] Loss: 0.036  
[Epoch 5, Batch 600] Loss: 0.030  
[Epoch 5, Batch 800] Loss: 0.033  
训练完成！ 
测试集准确率: 98.84%  
训练损失曲线已保存为: lenet_training_curve.png  
预测结果已保存为: lenet_predictions.png  

<img width="1247" height="613" alt="image" src="https://github.com/user-attachments/assets/8c73bc50-ef8d-4d43-9e4c-e8bc89443317" />   
图2 损失曲线  

<img width="2528" height="379" alt="image" src="https://github.com/user-attachments/assets/48c22003-4e35-4df9-a438-e82280ea2be7" />  

图3 预测结果  

