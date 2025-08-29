## AMSGrad优化器
### Adagrad 优化器（Adaptive Gradient Algorithm）原理和用法

#### **原理**
Adagrad（Adaptive Gradient Algorithm）是一种自适应学习率的优化算法，专门设计用于处理稀疏数据和凸优化问题。它通过根据历史梯度的累积调整每个参数的学习率，使得频繁更新的参数具有较小的学习率，而稀疏更新的参数具有较大的学习率。以下是 Adagrad 的核心原理：

1. **核心思想**：
   - Adagrad 为每个参数维护一个历史梯度平方的累积和，用于自适应地缩放学习率。
   - 梯度较大的参数（频繁更新）学习率逐渐减小，梯度较小的参数（稀疏更新）保持较大的学习率，从而加速稀疏特征的收敛。

2. **更新公式**：
<img width="665" height="516" alt="image" src="https://github.com/user-attachments/assets/c80b9e42-44b8-42ee-a975-d066bb7f1541" />  


---

* **梯度平方的累积**:

$$
G_t = G_{t-1} + g_t^2
$$

其中，$g_t$ 是当前梯度，$G_t$ 是历史梯度平方的累积和。

---

* **参数更新**:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t
$$

---

* **其中**:

  * $\theta_t$：当前参数。
  * $\eta$：初始学习率（通常 0.01）。
  * $\epsilon$：小常数（防止除零，通常 $1e^{-8}$）。
  * $\sqrt{G_t} + \epsilon$：自适应缩放因子，确保学习率随梯度累积而减小。


3. **特点**：
   - **优点**：
     - 自适应学习率，无需过多手动调参。
     - 特别适合稀疏数据（如自然语言处理中的词嵌入）。
   - **缺点**：
     - 梯度平方累积单调递增，导致学习率可能过快衰减到接近零，停止学习。
     - 对非凸问题（如深度神经网络）收敛可能较慢。
   - **与 Adadelta 的对比**：
     - Adagrad 累积所有历史梯度平方，可能导致学习率过早衰减。
     - Adadelta 使用指数移动平均（EMA）限制历史梯度影响，改进 Adagrad 的衰减问题。
   - **与 SGD 的对比**：
     - SGD 使用固定学习率，需手动调整。
     - Adagrad 自适应调整学习率，适合稀疏数据，但可能因学习率衰减过快而停止优化。

4. **适用场景**：
   - 适合稀疏数据任务，如 NLP（词嵌入训练）、推荐系统。
   - 对于深度学习中的非凸问题，可能需结合 Adadelta 或 RMSProp 等改进算法。

---

#### **PyTorch 用法**
PyTorch 提供了内置的 `torch.optim.Adagrad` 优化器，使用非常简单。以下是最简洁的代码示例，展示如何在图像分类任务中使用 Adagrad 优化器训练一个简单的 CNN。

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
optimizer = optim.Adagrad(model.parameters(), lr=0.01, eps=1e-8)  # Adagrad 优化器

# 3. 准备数据（示例数据）
inputs = torch.rand(16, 3, 224, 224)  # 批量图像 (batch, channels, height, width)
labels = torch.randint(0, 10, (16,))   # 随机标签

# 4. 训练步骤
model.train()
optimizer.zero_grad()  # 清空梯度
outputs = model(inputs)  # 前向传播
loss = criterion(outputs, labels)  # 计算损失
loss.backward()  # 反向传播
optimizer.step()  # 更新参数

print(f"Loss: {loss.item()}")
```

##### **代码说明**
- **模型**：使用预训练 ResNet18，替换最后一层以适应 10 类分类任务。
- **优化器**：`optim.Adagrad` 初始化，参数包括：
  - `model.parameters()`：模型的可学习参数。
  - `lr`：初始学习率，设为 0.01（常用范围 0.001~0.1）。
  - `eps`：防止除零的小常数，设为 1e-8（默认值）。
- **训练**：标准的前向传播、损失计算、反向传播和参数更新流程。
- **数据**：使用随机生成的图像和标签作为示例，实际应用中需替换为真实数据集（如 CIFAR-10）。

---

#### **注意事项**
1. **超参数**：
   - `lr`：初始学习率，通常设为 0.01，需根据任务调整。
   - `eps`：防止除零，常用 1e-8，影响较小。
   - 可选参数 `weight_decay`（默认 0）用于 L2 正则化。
2. **数据预处理**：
   - 图像需归一化（如均值 [0.485, 0.456, 0.406]，标准差 [0.229, 0.224, 0.225]）。
   - 使用 `torchvision.transforms` 进行预处理。
3. **计算设备**：
   - 若使用 GPU，需将模型和数据移动到 GPU：`model.cuda()`，`inputs.cuda()`，`labels.cuda()`。
4. **实际应用**：
   - 替换示例数据为真实数据集（如 `torchvision.datasets.CIFAR10`）。
   - 添加数据加载器（`DataLoader`）和多轮训练循环。
5. **局限性**：
   - 若发现收敛过慢或停止（因学习率衰减过快），可尝试 Adadelta 或 RMSProp。

---

#### **总结**
Adagrad 是一种自适应学习率的优化器，通过累积梯度平方动态调整学习率，特别适合稀疏数据任务。PyTorch 的 `optim.Adagrad` 实现简单，只需指定学习率等少量参数即可。相比 SGD，Adagrad 减少了学习率调参；相比 Adadelta，Adagrad 更简单但可能因学习率衰减过快而停止优化。
