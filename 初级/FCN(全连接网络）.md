## FCN (全连接网络)
## FCN(MLP)
全连接神经网络（Fully Connected Neural Network / MLP）  
- 重要性：这是深度学习的基础模型，理解它有助于掌握神经网络的核心概念（如神经元、权重、激活函数、梯度下降）。  
- 核心概念：  
神经网络由输入层、隐藏层和输出层组成，像大脑的神经元一样工作。  
每个神经元接收输入，计算加权和，经过激活函数（如 sigmoid 或 ReLU）输出。  
训练是通过“试错”调整权重，使预测更接近真实值（用梯度下降）。  
- 应用：房价预测、分类任务（如判断图片是猫还是狗）。
- 为什么教：MLP 是所有深度学习模型的基础，帮助学生理解神经网络的核心机制。  
<img width="850" height="253" alt="image" src="https://github.com/user-attachments/assets/4f07aa2a-dd72-4e95-8543-7f71810d8023" />

## 代码（pytorch）
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
    transforms.Normalize((0.5,), (0.5,))
])

# 加载Fashion MNIST数据集
trainset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 2. 定义神经网络模型
class FashionMNISTClassifier(nn.Module):
    def __init__(self):
        super(FashionMNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)      # 隐藏层到输出层（10个类别）
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 3. 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
def train_model(num_epochs=5):
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
    train_model(num_epochs=5)
    print("\nTesting started...")
    test_model()

```
## 训练结果
Epoch [5/5], Step [800], Loss: 0.3124   
Epoch [5/5], Step [900], Loss: 0.2941   

Testing started...   
Test Accuracy: 87.24%   
