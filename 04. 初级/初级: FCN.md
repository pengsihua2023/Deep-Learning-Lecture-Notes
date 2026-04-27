# FCN
全连接神经网络（Fully Connected Neural Network, FCNN），也称为前馈神经网络（Feedforward Neural Network），是一类最基本的人工神经网络结构。其定义如下：

### 基本定义

在全连接神经网络中，相邻层之间的**每一个神经元都与下一层的所有神经元相连**。也就是说，一层的输出会作为加权输入传递给下一层的所有节点。
<div align="center">
<img width="700" height="210" alt="image" src="https://github.com/pengsihua2023/Deep-Learning-Lecture-Notes-English/blob/main/images/FCN.png" />  
</div>

### 结构特点

1. **输入层（Input Layer）**：接收原始数据（如特征向量）。
2. **隐藏层（Hidden Layers）**：由多个神经元组成，通过加权求和和激活函数进行非线性变换，从而提取和组合特征。
3. **输出层（Output Layer）**：给出预测结果或分类结果。
4. **权重与偏置（Weights & Biases）**：每一条连接线都有对应的权重参数，每个神经元通常还带有偏置项。
5. **激活函数（Activation Function）**：在隐藏层或输出层中引入非线性，使网络具备拟合复杂函数的能力。

### 特点与应用

* **优点**：结构简单，通用性强，能逼近任意连续函数（通用近似定理）。
* **缺点**：参数量大（尤其在输入维度很高时），容易过拟合，训练效率较低。
* **应用**：早期多用于简单分类与回归问题，如手写数字识别（MNIST）、表格数据预测等。

 
## 📖 数学描述

## 1. 网络结构

一个典型的全连接神经网络由若干 **层 (layers)** 构成：

* 输入层（input layer）
* 一个或多个隐藏层（hidden layers）
* 输出层（output layer）

在全连接结构中，**每一层的每个神经元都与上一层的所有神经元相连**。



## 2. 数学符号

设：

* 输入向量为


$$
\mathbf{x} \in \mathbb{R}^{d}
$$


* 第 $l$ 层有 $n_l$ 个神经元，输出记为

$$
\mathbf{h}^{(l)} \in \mathbb{R}^{n_l}
$$

* 权重矩阵与偏置为

$$
\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}, \quad \mathbf{b}^{(l)} \in \mathbb{R}^{n_l}
$$

* 激活函数为

$$
\sigma(\cdot)
$$



## 3. 前向传播 (Forward Propagation)

输入层记为

$$
\mathbf{h}^{(0)} = \mathbf{x}
$$

对于第 $l$ 层 ($l=1,2,\dots,L$)，有：

1. **线性变换：**

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}
$$

2. **非线性激活：**

$$
\mathbf{h}^{(l)} = \sigma\left(\mathbf{z}^{(l)}\right)
$$

最终，输出层结果为：

$$
\hat{\mathbf{y}} = \mathbf{h}^{(L)}
$$



## 4. 损失函数 (Loss Function)

训练时，给定目标输出 $\mathbf{y}$，常用损失函数包括：

* **回归问题：** 均方误差（MSE）

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{N}\sum_{i=1}^N \|\hat{\mathbf{y}}^{(i)} - \mathbf{y}^{(i)}\|^2
$$

* **分类问题：** 交叉熵损失（Cross-Entropy）

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{k=1}^K y_k \log \hat{y}_k
$$



## 5. 参数更新 (Backpropagation + Gradient Descent)

通过反向传播 (Backpropagation) 计算损失函数对参数的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

再用梯度下降或其变种（如 Adam, SGD, RMSProp）更新：

$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}
$$

$$
\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

其中 $\eta$ 为学习率。



### 总结来说，全连接神经网络可以抽象为：

$$
\hat{\mathbf{y}} = f(\mathbf{x}; \Theta) = \sigma^{(L)}\Big(\mathbf{W}^{(L)} \sigma^{(L-1)}(\cdots \sigma^{(1)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) \cdots ) + \mathbf{b}^{(L)}\Big)
$$

其中 $\Theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)} \mid l=1,\dots,L\}$ 为模型参数。



## 📖 代码（pytorch）
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
## 📖 训练结果
Epoch [5/5], Step [800], Loss: 0.3124   
Epoch [5/5], Step [900], Loss: 0.2941   

Testing started...   
Test Accuracy: 87.24%   
