## SGD优化器（随机梯度下降，Stochastic Gradient Descent）
### 原理和用法

#### **原理**
SGD（Stochastic Gradient Descent，随机梯度下降）是一种经典的优化算法，广泛用于深度学习模型的训练。其核心思想是通过计算损失函数对模型参数的梯度，沿着梯度反方向更新参数以最小化损失。SGD 的“随机”体现在每次更新只使用一个样本或一个小批量（mini-batch）数据，而非整个数据集，从而加速计算。

1. **核心公式**：

* **参数更新规则**:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t, x_i, y_i)
$$

其中：

* $\theta_t$：当前参数。
* $\eta$：学习率（learning rate），控制步长。
* $\nabla_\theta L$：损失函数 $L$ 对参数的梯度，基于样本 $(x_i, y_i)$ 或小批量数据。



* **如果使用动量法（Momentum），更新规则变为**:

$$
v_t = \gamma v_{t-1} + \eta \cdot \nabla_\theta L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - v_t
$$

其中 $\gamma$ 是动量系数（通常 0.9），$v_t$ 是速度（累积历史梯度）。


2. **特点**：
   - **优点**：
     - 计算效率高，适合大规模数据集。
     - 随机性有助于逃离局部最小值。
   - **缺点**：

* 梯度噪声大，可能导致收敛不稳定。
* 需手动调整学习率 $\eta$ 和动量 $\gamma$。

   - **适用场景**：图像分类、文本分类、回归等深度学习任务，适合简单优化或作为基准算法。

3. **与 Adadelta 的对比**：
   - SGD 需要手动设置学习率，Adadelta 自适应调整学习率。
   - SGD 简单直接，Adadelta 引入梯度平方和更新平方的指数移动平均，计算更复杂但更鲁棒。



#### **PyTorch 用法**
PyTorch 提供了内置的 `torch.optim.SGD` 优化器，支持基础 SGD 和带动量的变体。以下是最简洁的代码示例，展示如何在图像分类任务中使用 SGD 优化器训练一个简单的 CNN。

##### **代码示例**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 1. 定义模型（使用预训练 ResNet18）
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 假设 10 个类别

# 2. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD 优化器

# 3. 准备数据（示例数据）
inputs = torch.rand(16, 3, 224, 224)  # 批量图像 (batch, channels, height, width)
labels = torch.randint(0, 10, (16,))   # 随机标签

# 4. 训练步骤
model.train()
optimizer.zero_grad()  # 清かせ

System: 清空梯度
loss = criterion(outputs, labels)  # 计算损失
loss.backward()  # 反向传播
optimizer.step()  # 更新参数

print(f"Loss: {loss.item()}")
```

##### **代码说明**
- **模型**：使用预训练 ResNet18，替换最后一层以适应 10 类分类任务。
- **优化器**：`optim.SGD` 初始化，参数包括：
  - `model.parameters()`：模型的可学习参数。
  - `lr`：学习率（learning rate），这里设为 0.01。
  - `momentum`：动量系数，设为 0.9，增强收敛稳定性。
- **训练**：标准的前向传播、损失计算、反向传播和参数更新流程。
- **数据**：使用随机生成的图像和标签作为示例，实际应用中需替换为真实数据集（如 CIFAR-10）。

---

#### **注意事项**
1. **超参数**：
   - `lr`：学习率需根据任务调整，常用范围 0.001~0.1。
   - `momentum`：通常设为 0.9，若不使用动量可设为 0。
2. **数据预处理**：
   - 图像需归一化（如均值 [0.485, 0.456, 0.406]，标准差 [0.229, 0.224, 0.225]）。
   - 使用 `torchvision.transforms` 进行预处理。
3. **计算设备**：
   - 若使用 GPU，需将模型和数据移动到 GPU：`model.cuda()`，`inputs.cuda()`，`labels.cuda()`。
4. **实际应用**：
   - 替换示例数据为真实数据集（如 `torchvision.datasets.CIFAR10`）。
   - 添加数据加载器（`DataLoader`）和多轮训练循环。

---

#### **总结**
SGD 是一种简单高效的优化器，通过随机梯度更新参数，适合多种深度学习任务。PyTorch 的 `optim.SGD` 实现易用，只需指定学习率和动量等参数即可。相比 Adadelta，SGD 需要手动调参，但计算更轻量，适合快速实验。
