## CNN
卷积神经网络（Convolutional Neural Network, CNN）  
- 重要性：CNN 是计算机视觉领域的基石，广泛用于图像识别、自动驾驶等，适合展示深度学习的实际威力。
- 核心概念：
CNN 使用“卷积”操作，像一个“放大镜”扫描图片，提取特征（如边缘、形状）。  
池化层（Pooling）缩小数据，保留重要信息，减少计算量。  
最后用全连接层做分类或预测。  
- 应用：图像分类（猫狗识别）、人脸识别、医疗影像分析。
<img width="708" height="353" alt="image" src="https://github.com/user-attachments/assets/c404062e-9dc5-4c41-bf8d-93cf080c6181" />

## 代码（Pytorch）
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 设置随机种子以确保结果可重复
torch.manual_seed(42)

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR10数据集
trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 2. 定义卷积神经网络模型
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # 卷积层部分
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输入3通道，输出32通道
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        # 全连接层部分
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # CIFAR10图像经过两次池化后为8x8
        self.fc2 = nn.Linear(512, 10)  # 10个类别
        self.dropout = nn.Dropout(0.5)  # Dropout防止过拟合
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
def train_model(num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# 5. 测试模型
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# 6. 执行训练和测试
if __name__ == "__main__":
    print("Training started...")
    train_model(num_epochs=10)
    print("\nTesting started...")
    test_model()
```
### 训练结果
Epoch [10/10], Step [600], Loss: 0.4115  
Epoch [10/10], Step [700], Loss: 0.4196  

Testing started...  
Test Accuracy: 74.14%  
