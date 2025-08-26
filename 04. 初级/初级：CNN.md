## CNN
卷积神经网络（Convolutional Neural Network, CNN）  
- 重要性：CNN 是计算机视觉领域的基石，广泛用于图像识别、自动驾驶等，适合展示深度学习的实际威力。
- 核心概念：
CNN 使用“卷积”操作，像一个“放大镜”扫描图片，提取特征（如边缘、形状）。  
池化层（Pooling）缩小数据，保留重要信息，减少计算量。  
最后用全连接层做分类或预测。  
- 应用：图像分类（猫狗识别）、人脸识别、医疗影像分析。
  
<div align="center">
<img width="708" height="353" alt="image" src="https://github.com/user-attachments/assets/c404062e-9dc5-4c41-bf8d-93cf080c6181" />
</div>
    
---

## 卷积神经网络（CNN）的数学描述

CNN 的核心由以下几个基本运算组成：**卷积层（Convolutional Layer）**、**非线性激活函数（Activation Function）**、**池化层（Pooling Layer）**，以及最后的 **全连接层（Fully Connected Layer）**。我们逐一描述。

---

### 1. 卷积层（Convolutional Layer）

设输入特征图为

$$
\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}
$$

其中 $H$ 是高度， $W$ 是宽度， $C_{in}$ 是输入通道数。

卷积核（滤波器）为

$$
\mathbf{K} \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}
$$

其中 $k_h, k_w$ 为卷积核大小， $C_{out}$ 为输出通道数。

卷积运算定义为：

$$
Y_{i,j,c_{out}} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{c_{in}=0}^{C_{in}-1} 
X_{i+m, j+n, c_{in}} \cdot K_{m,n,c_{in},c_{out}} + b_{c_{out}}
$$

其中 $b_{c_{out}}$ 是偏置项。输出特征图为

$$
\mathbf{Y} \in \mathbb{R}^{H' \times W' \times C_{out}}
$$

具体尺寸取决于步幅（stride）和填充（padding）。

---

### 2. 激活函数（Activation Function）

常用激活函数为 ReLU（线性整流单元）：

$$
f(z) = \max(0, z)
$$

应用到卷积输出：

$$
Z_{i,j,c} = f(Y_{i,j,c})
$$

---

### 3. 池化层（Pooling Layer）

池化操作用于降低特征图尺寸。
以最大池化（Max Pooling）为例：

$$
P_{i,j,c} = \max_{0 \leq m < p_h,  0 \leq n < p_w} Z_{i \cdot s + m,  j \cdot s + n,  c}
$$

其中 $p_h, p_w$ 为池化窗口大小， $s$ 为步幅。

---

### 4. 全连接层（Fully Connected Layer）

经过若干层卷积和池化后，得到展平的特征向量：

$$
\mathbf{x} \in \mathbb{R}^d
$$

全连接层输出为：

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

其中 $W \in \mathbb{R}^{k \times d}$， $\mathbf{b} \in \mathbb{R}^k$。

---

### 5. 分类层（Softmax）

在分类任务中，最后通过 Softmax 输出概率分布：

![Softmax 公式](https://latex.codecogs.com/png.latex?\hat{y}_i%20=%20\frac{\exp(y_i)}{\sum_{j=1}^{k}%20\exp(y_j)})

---
## 代码（Pytorch）
```python
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
