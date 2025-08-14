## 自适应学习率 （RMSprop）
### 什么是自适应学习率（RMSprop, 优化器）？

RMSprop（Root Mean Square Propagation）是一种在深度学习中常用的自适应学习率优化算法，旨在通过自适应地调整学习率来加速梯度下降的收敛。它特别适合处理非平稳目标函数（如神经网络中的损失函数），通过基于梯度平方均值的指数移动平均来动态调整每个参数的学习率。RMSprop是Adam优化器的前身之一，简单且高效。

#### 核心原理
RMSprop通过维护梯度平方的指数移动平均来缩放学习率，具体步骤如下：
<img width="847" height="557" alt="image" src="https://github.com/user-attachments/assets/10861619-a441-4986-a6f3-fd193bccb66d" />


#### 优势
- **自适应性**：根据梯度大小动态调整学习率，适合稀疏或噪声大的梯度。
- **简单高效**：比SGD更快收敛，且实现简单。
- **稳定性**：通过指数移动平均平滑梯度波动，减少震荡。

#### 局限性
- **缺少动量**：不像Adam，RMSprop不使用一阶动量，可能在某些任务中收敛较慢。
- **超参数敏感**：初始学习率和衰减率需适当选择。

#### 与Adam的对比
- **RMSprop**：仅使用梯度二阶矩（平方均值）来缩放学习率。
- **Adam**：结合一阶矩（动量）和二阶矩，收敛通常更快（参考前述Adam优化器）。

---

### Python代码示例

以下是一个最简单的PyTorch示例，展示如何在MNIST手写数字分类任务中使用RMSprop优化器。代码保持极简，聚焦RMSprop的实现。

```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 步骤1: 定义简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # 输入：28x28像素，输出：10类
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入
        x = self.fc(x)
        return x

# 步骤2: 加载MNIST数据集
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 步骤3: 初始化模型、损失函数和RMSprop优化器
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-8)

# 步骤4: 训练函数
def train(epoch):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}')

# 步骤5: 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# 步骤6: 训练循环
epochs = 5
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
```

---

### 代码说明

1. **模型定义**：
   - `SimpleNet` 是一个极简的全连接神经网络，输入为MNIST的28x28像素图像，输出为10类分类。

2. **数据集**：
   - 使用`torchvision`加载MNIST数据集，批量大小为64，数据预处理仅包括转换为张量。

3. **RMSprop优化器**：
   - 初始化为`optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-8)`。
     - `lr=0.001`：初始学习率，RMSprop常用值。
     - `alpha=0.9`：衰减率（对应公式中的 \( \rho \)），控制梯度平方均值的平滑程度。
     - `eps=1e-8`：防止除零的小常数。
   - RMSprop根据梯度平方均值自适应调整学习率。

4. **训练与测试**：
   - 训练时使用RMSprop更新参数，打印平均损失。
   - 测试时计算分类准确率。

5. **输出示例**：
   ```
   Epoch 1, Loss: 0.4567, Test Accuracy: 92.30%
   Epoch 2, Loss: 0.2987, Test Accuracy: 93.50%
   Epoch 3, Loss: 0.2678, Test Accuracy: 94.10%
   Epoch 4, Loss: 0.2456, Test Accuracy: 94.40%
   Epoch 5, Loss: 0.2321, Test Accuracy: 94.60%
   ```
   实际值因随机初始化而异。

---

### 关键点
- **自适应性**：RMSprop根据梯度平方均值动态调整学习率，适合非平稳损失函数。
- **超参数**：
   - `lr=0.001`：RMSprop默认值通常有效。
   - `alpha=0.9`：控制历史梯度的权重，0.9~0.99常见。
- **与Adam对比**：RMSprop仅使用二阶矩，Adam额外引入一阶动量，收敛更快但内存需求略高。

---

### 实际应用场景
- **深度学习**：RMSprop适用于CNN、RNN等，尤其在梯度稀疏或噪声大的任务中。
- **替代Adam**：RMSprop计算更简单，适合资源受限场景。
- **与其他技术结合**：可与Dropout、BatchNorm、学习率调度（如ReduceLROnPlateau，参考前述问题）联合使用。

#### 注意事项
- **学习率调优**：`lr`需根据任务调整，典型范围为1e-4到1e-2。
- **衰减率**：`alpha`过高可能导致对新梯度不敏感，过低则不稳定。
- **收敛性**：RMSprop可能在某些任务中不如Adam稳定，可尝试Adam或AdamW。
