## 自适应学习率 （Adam Optimizer）
### 📖 什么是自适应学习率（Adam优化器）？

Adam优化器（Adaptive Moment Estimation，适应性矩估计）是一种在深度学习中广泛使用的优化算法，结合了动量法和RMSProp的优点，通过自适应地调整学习率来加速梯度下降的收敛。它特别适合处理稀疏梯度或噪声较大的优化问题。

### 📖 核心原理
Adam通过跟踪梯度的一阶矩（均值）和二阶矩（方差）的指数移动平均值来动态调整每个参数的学习率。主要步骤：

1. **计算梯度**：对损失函数求参数的梯度 $g_t$。

2. **更新一阶矩（动量）**：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

类似动量法。

3. **更新二阶矩（方差）**：

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

类似 RMSProp。

4. **偏差校正**：对 $m_t$ 和 $v_t$ 进行偏差校正，确保初期估计无偏。

5. **参数更新**：使用自适应学习率更新参数：

$$
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$



* **其中**:

  * $\eta$：初始学习率（通常 0.001）。
  * $\beta_1, \beta_2$：动量参数（通常 0.9 和 0.999）。
  * $\epsilon$：小值防止除零（通常 $1e^{-8}$）。


### 📖 优势
- **自适应性**：根据梯度历史自动调整学习率，无需手动调参。
- **高效性**：适合大规模数据集和复杂模型（如深度神经网络）。
- **稳定性**：对稀疏梯度或噪声优化问题表现良好。

### 📖 局限性

* 在某些任务上可能不如 SGD + 动量 收敛到最优解。
* 超参数（如 $\beta_1, \beta_2$）仍需适当选择。


---

### 📖 Python代码示例

以下是一个简单的PyTorch示例，使用Adam优化器训练一个全连接神经网络，解决MNIST手写数字分类问题。代码聚焦于Adam的实现，保持简洁。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 步骤1: 定义简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入：28x28像素
        self.fc2 = nn.Linear(128, 10)       # 输出：10类
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 步骤2: 加载MNIST数据集
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 步骤3: 初始化模型、损失函数和Adam优化器
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# 步骤4: 训练函数
def train(epoch):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()  # 清空梯度
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # 反向传播
        optimizer.step()  # 使用Adam更新参数
        total_loss += loss.item()
    print(f'Epoch {epoch}, Average Loss: {total_loss / len(train_loader):.4f}')

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

# 步骤6: 运行训练和测试
epochs = 5
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
```

---

### 📖 代码说明

1. **模型定义**：
   - `SimpleNet` 是一个简单的全连接神经网络，输入为MNIST的28x28像素图像，输出为10类分类。

2. **数据集**：
   - 使用`torchvision`加载MNIST数据集，批量大小为64，数据预处理仅包括转换为张量。

3. **Adam优化器**：
   - 初始化为`optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)`。
   - `lr=0.001`：初始学习率，Adam的默认值通常效果良好。
   - `betas=(0.9, 0.999)`：一阶和二阶矩的衰减率，标准值。
   - `eps=1e-8`：防止除零的小值。

4. **训练与测试**：
   - 训练时，Adam优化器根据梯度自适应调整学习率，更新模型参数。
   - 每个epoch打印平均损失，测试时计算分类准确率。

5. **输出示例**：
   ```
   Epoch 1, Average Loss: 0.3256
   Test Accuracy: 94.50%
   Epoch 2, Average Loss: 0.1423
   Test Accuracy: 96.20%
   ...
   Epoch 5, Average Loss: 0.0854
   Test Accuracy: 97.80%
   ```
   实际值因随机初始化而异。

---

### 📖 Adam与SGD的对比
为了展示Adam的自适应性，可以对比SGD的实现（不含动量）：

```python
# 使用SGD优化器（无动量）
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 替换Adam
```

- **Adam**：自适应调整学习率，通常收敛更快，对初始学习率不敏感。
- **SGD**：固定学习率，可能需要手动调整或结合学习率调度（如StepLR）。

---

### 📖 实际应用场景
- **深度学习**：Adam是CNN、RNN、Transformer等模型的默认优化器，因其收敛快、稳定性好。
- **稀疏数据**：Adam对稀疏梯度（如NLP任务）表现优异。
- **超参数调优**：虽然Adam对学习率不敏感，但仍可通过贝叶斯优化（如前述问题）调整`lr`、`betas`等。

### 📖 注意事项
- **学习率**：Adam默认`lr=0.001`通常有效，但对于特定任务可能需要微调（如1e-4到1e-2）。
- **收敛性**：在某些任务上，Adam可能收敛到次优解，可尝试SGD+动量或AdamW（改进版Adam，加入L2正则化）。
- **内存开销**：Adam需存储一阶和二阶矩，内存需求略高于SGD。

---
