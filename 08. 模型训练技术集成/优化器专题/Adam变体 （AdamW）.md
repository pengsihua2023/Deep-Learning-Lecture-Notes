## Adam变体 （AdamW）
### 什么是Adam变体（AdamW）？

AdamW（Adaptive Moment Estimation with Weight Decay）是Adam优化器的一种变体，改进了Adam在正则化（特别是L2正则化或权重衰减）方面的处理方式。AdamW通过将权重衰减（Weight Decay）与自适应学习率解耦，解决了原始Adam在权重衰减上的次优表现问题，使其在许多任务中收敛更快且泛化能力更强。

#### 核心原理
Adam优化器结合了一阶动量（梯度均值）和二阶动量（梯度平方均值）来调整学习率（见前述Adam问题）。原始Adam将权重衰减直接融入梯度更新，相当于在损失函数中添加L2正则化项：

<img width="909" height="833" alt="image" src="https://github.com/user-attachments/assets/c7a8fc23-4407-4990-b537-b4b6756d2d5f" />


$$
Loss_{Adam} = Loss_{original} + \frac{\lambda}{2} \sum w_i^2
$$

然而，这种方式与 Adam 的自适应学习率机制（基于梯度平方均值）相互干扰，导致正则化效果不佳。
AdamW 通过解耦权重衰减，直接在参数更新步骤中减去权重衰减项，而不是将其融入梯度计算：


1. **计算梯度**：对原始损失函数求梯度 $g_t$。

2. **更新一阶动量**：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

3. **更新二阶动量**：

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

4. **偏差校正**：对 $m_t$ 和 $v_t$ 进行校正以消除初始化偏差。

5. **参数更新（AdamW 的区别在此）**：

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_t \right)
$$

---

* 其中：

  * $\eta$：初始学习率（通常 0.001）。
  * $\beta_1, \beta_2$：动量参数（通常 0.9 和 0.999）。
  * $\epsilon$：防止除零（通常 $1e^{-8}$）。
  * $\lambda$：权重衰减系数（通过 *weight\_decay* 设置）。

---

AdamW 直接对参数施加 $\lambda \theta_t$ 的衰减，而不是将其作为梯度的一部分，从而更好地平衡优化和正则化。


#### 优势
- **更好的正则化**：解耦权重衰减提高泛化能力，优于原始Adam的L2正则化。
- **收敛更快**：在许多任务（如Transformer、CNN）中，AdamW比Adam更稳定。
- **广泛应用**：AdamW是现代深度学习（如BERT、GPT）的默认优化器。

#### 局限性
- **超参数敏感**：权重衰减系数 lambda 需调优。
- **内存需求**：与Adam相同，需存储一阶和二阶动量。

---

### Python代码示例

以下是一个最简单的PyTorch示例，展示如何在MNIST手写数字分类任务中使用AdamW优化器。代码保持极简，聚焦AdamW的实现。

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

# 步骤3: 初始化模型、损失函数和AdamW优化器
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

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

3. **AdamW优化器**：
   - 初始化为`optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`。
     - `lr=0.001`：初始学习率，AdamW常用值。
     - `betas=(0.9, 0.999)`：一阶和二阶动量参数，与Adam相同。
     - `eps=1e-8`：防止除零。
     - `weight_decay=0.01`：权重衰减系数，控制正则化强度（比Adam的L2正则化更有效）。

4. **训练与测试**：
   - 训练时使用AdamW更新参数，打印平均损失。
   - 测试时计算分类准确率。

5. **输出示例**：
   ```
   Epoch 1, Loss: 0.4321, Test Accuracy: 92.50%
   Epoch 2, Loss: 0.2876, Test Accuracy: 93.80%
   Epoch 3, Loss: 0.2564, Test Accuracy: 94.20%
   Epoch 4, Loss: 0.2345, Test Accuracy: 94.50%
   Epoch 5, Loss: 0.2213, Test Accuracy: 94.70%
   ```
   实际值因随机初始化而异。

---

### 关键点
- **解耦权重衰减**：AdamW直接对参数施加衰减（`λθ`），而不是将其融入梯度，效果优于Adam的L2正则化。
- **超参数**：
   - `lr=0.001`：默认值通常有效。
   - `weight_decay=0.01`：常见值，需根据任务调优（范围1e-4到1e-1）。
- **与Adam对比**：AdamW在大多数任务（尤其是Transformer）中泛化性能更好。

---

### 实际应用场景
- **Transformer模型**：AdamW是BERT、GPT等模型的标准优化器，因其正则化效果更好。
- **深度学习**：适用于CNN、RNN等任务，特别是在需要强正则化的场景。
- **与其他技术结合**：可与Dropout、BatchNorm、ReduceLROnPlateau（如前述问题）联合使用。

#### 注意事项
- **权重衰减调优**：`weight_decay`需通过交叉验证或贝叶斯优化调整。
- **学习率**：AdamW对学习率敏感，可结合学习率调度（如ReduceLROnPlateau）。
- **内存需求**：与Adam相同，需存储动量信息。
