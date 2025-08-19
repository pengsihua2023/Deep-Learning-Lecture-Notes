## 自动混合精度（AMP）训练
“自动混合精度（Automatic Mixed Precision，AMP）”是深度学习训练中一种 **自动使用不同数值精度（FP16 和 FP32）进行计算** 的技术，主要目的是在 **保持模型精度的同时加速训练并减少显存占用**。

---

### 1. 背景

在深度学习中常用的浮点数精度有：

* **FP32 (单精度浮点数)**：常规训练的标准格式，数值范围大、稳定性好，但运算速度慢、显存占用高。
* **FP16 (半精度浮点数)**：精度较低，但运算速度更快，占用显存更少。

如果把所有运算都切换到 FP16，训练可能会因为溢出、下溢、舍入误差而失败。
于是 AMP 就出现了——它能 **智能地决定哪些操作用 FP16，哪些必须保留 FP32**。

---

### 2. AMP 的核心思想

AMP 会：

* 在 **适合 FP16 的地方**（比如矩阵乘法、卷积）使用 FP16 → 加速运算、节省显存。
* 在 **需要高精度的地方**（比如 loss 计算、梯度累积、softmax、batch norm）保留 FP32 → 保证数值稳定性。
* 使用 **动态 loss scaling（损失缩放）** 避免 FP16 下的下溢问题。

这样就能既快又稳。

---

### 3. 在主流框架中的实现

* **PyTorch**：
  提供 `torch.cuda.amp`，通过 `autocast` 和 `GradScaler` 实现。

  ```python
  scaler = torch.cuda.amp.GradScaler()

  for data, target in loader:
      optimizer.zero_grad()
      with torch.cuda.amp.autocast():
          output = model(data)
          loss = loss_fn(output, target)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
  ```

* **TensorFlow**：
  可以通过 `mixed_float16` policy 自动混合精度。

* **NVIDIA Apex**（早期）：
  提供 `amp.initialize` 来简化 FP16 训练，但现在 PyTorch 自带 `torch.cuda.amp` 已成为主流。

---

### 4. AMP 的优点

- 加快训练速度（特别是 GPU Tensor Cores 上）
- 显存占用更低，可以训练更大 batch size / 模型
- 保持几乎相同的收敛精度

---
### 代码例子
**PyTorch + AMP** 的最小示例，用一个简单的 **全连接网络在 MNIST 上训练** 来说明自动混合精度（AMP）的用法。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 数据加载
transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

# 2. 简单的模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 3. AMP 相关对象
scaler = torch.cuda.amp.GradScaler()  # 自动缩放避免溢出

# 4. 训练循环
for epoch in range(1, 3):  # 只跑 2 个 epoch 示例
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # 在 autocast 下自动混合精度
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)

        # 反向传播时用 scaler 来缩放梯度
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 200 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}")
```

---

### 🔑 关键点：

1. `torch.cuda.amp.autocast()`

   * 前向传播时自动选择 FP16 / FP32。
   * 比如卷积、矩阵乘法会用 FP16，加速并减少显存；而损失计算仍用 FP32 保持稳定。

2. `torch.cuda.amp.GradScaler()`

   * 反向传播时自动缩放 loss，避免 FP16 下梯度下溢。

3. 其余地方和普通训练流程几乎一样，几乎不用改动代码，就能启用 AMP。

---

