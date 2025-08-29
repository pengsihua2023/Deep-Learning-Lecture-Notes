## Adadelta优化器
### Adadelta 优化器的原理和用法

#### **原理**
Adadelta（Adaptive Delta）是一种自适应学习率的优化算法，旨在解决 Adagrad 学习率随时间单调递减到接近零的问题。它是 Adagrad 的改进版本，结合了动量法和自适应学习率的思想，适用于深度学习模型的优化。以下是 Adadelta 的核心原理：

1. **核心思想**：
   - Adadelta 不直接累积所有历史梯度的平方（如 Adagrad），而是使用**滑动窗口**来计算梯平方的**指数移动平均**（Exponential Moving Average, EMA），限制历史信息的影响。
   - 它引入了两个累积量：
<img width="714" height="103" alt="image" src="https://github.com/user-attachments/assets/2dba7b30-7464-4751-8dfb-090fad2e73af" />


---

* 梯度平方的 EMA $(E[g^2])$：用于自适应地缩放学习率。

* 参数更新平方的 EMA $(E[\Delta x^2])$：用于归一化更新步长，模拟动量效应。

   - 通过这两者，Adadelta 实现了无需手动设置全局学习率的自适应更新。

2. **更新公式**：
   - 梯度平方的 EMA：
<img width="636" height="112" alt="image" src="https://github.com/user-attachments/assets/6c86702f-2710-40f3-ad13-7f419583cf37" />

   - 参数更新步长：
<img width="686" height="360" alt="image" src="https://github.com/user-attachments/assets/14694fee-d0a3-492d-8a60-02ed16330bd9" />


3. **优点**：
   - **无需设置学习率**：通过 EMA 自适应调整步长。
   - **对稀疏梯度鲁棒**：适合深度学习中非平稳目标函数。
   - **收敛稳定**：避免了 Adagrad 学习率过快衰减的问题。

4. **缺点**：
<img width="521" height="97" alt="image" src="https://github.com/user-attachments/assets/129dbbcf-b299-4e91-88cd-945ead570b74" />


#### **适用场景**
- 适用于深度学习模型（如 CNN、RNN）的优化，特别在数据稀疏或梯度变化较大的任务中。
- 常用于图像分类、文本分类等任务，尤其当希望减少学习率调参时。

---

#### **PyTorch 用法**
PyTorch 提供了内置的 `torch.optim.Adadelta` 优化器，使用非常简单。以下是最简洁的代码示例，展示如何在图像分类任务中使用 Adadelta 优化器训练一个简单的 CNN。

##### **代码示例**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 1. 定义模型（简单使用预训练 ResNet18）
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 假设 10 个类别

# 2. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), rho=0.9, eps=1e-6)  # Adadelta 优化器

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
- **优化器**：`optim.Adadelta` 初始化，参数包括：
  - `model.parameters()`：模型的可学习参数。
  - `rho`：EMA 的衰减率，默认 0.9。
  - `eps`：防止除零的小常数，默认 1e-6。
  - `lr`：初始学习率（默认 1.0，通常无需调整）。
- **训练**：标准的前向传播、损失计算、反向传播和参数更新流程。
- **数据**：使用随机生成的图像和标签作为示例，实际应用中需替换为真实数据集（如 CIFAR-10）。

---

#### **注意事项**
1. **超参数**：
   - `rho`：控制历史梯度的影响，通常设为 0.9~0.95。
   - `eps`：防止除零，通常设为 1e-6~1e-8。
   - Adadelta 对初始学习率 `lr` 不敏感，默认 1.0 即可。
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
Adadelta 是一种高效的自适应优化器，通过 EMA 动态调整学习率，适合深度学习任务。PyTorch 的 `optim.Adadelta` 实现简单易用，只需指定模型参数和少量超参数即可。
