# Rprop（Resilient Backpropagation）优化器


## 1. 定义

**Rprop (Resilient Backpropagation)** 是一种基于梯度符号的优化算法，最初用于训练前馈神经网络。
它的核心思想是：

* **只利用梯度的符号（sign），而不是大小**，来决定参数更新的方向。
* 每个参数有独立的更新步长（学习率），根据梯度方向是否一致来**自适应调整步长大小**。
* 这样可以避免梯度消失（太小）或梯度爆炸（太大）的问题。

和 SGD、Adam 不同，Rprop 不直接用梯度值来缩放更新，而是维护一个**独立的步长 Δ**，根据梯度符号动态调整。



## 2. 数学公式

设：

* 参数： $\theta_{t}$
* 梯度： $g_t = \frac{\partial L}{\partial \theta_t}$
* 更新步长： $\Delta_t$
* 初始步长： $\Delta_0$（如 0.1）
* 放大因子： $\eta^+ > 1$（如 1.2）
* 缩小因子： $\eta^- < 1$（如 0.5）
* 步长范围： $[\Delta_{\min}, \Delta_{\max}]$

规则如下：

1. **比较当前梯度和上一次梯度的符号**：

   * 如果 $\text{sign}(g_t) = \text{sign}(g_{t-1})$ ：说明方向一致，增加步长：

$$
\Delta_t = \min(\Delta_{t-1} \cdot \eta^+, \Delta_{\max})
$$

   * 如果 $\text{sign}(g_t) \neq \text{sign}(g_{t-1})$ ：说明越过极小值，减小步长，并撤销上次更新：

$$
\Delta_t = \max(\Delta_{t-1} \cdot \eta^-, \Delta_{\min})
$$

     并令 $g_t = 0$（避免振荡）。
   * 否则（梯度为 0），保持步长不变。

2. **更新参数**：

$$
\theta_{t+1} = \theta_t - \text{sign}(g_t) \cdot \Delta_t
$$



## 3. 最简代码例子

用 **NumPy** 实现一个极简版 Rprop 优化器：

```python
import numpy as np

# 目标函数 f(x) = x^2
def grad(theta):
    return 2 * theta

theta = np.array([5.0])   # 初始参数
delta = np.array([0.1])   # 初始步长
prev_grad = np.array([0.0])

eta_plus, eta_minus = 1.2, 0.5
delta_min, delta_max = 1e-6, 50

for t in range(50):
    g = grad(theta)

    # 符号一致 -> 增大步长
    if g * prev_grad > 0:
        delta = np.minimum(delta * eta_plus, delta_max)
    # 符号相反 -> 减小步长，并撤销更新
    elif g * prev_grad < 0:
        delta = np.maximum(delta * eta_minus, delta_min)
        g = 0  # 避免振荡

    # 参数更新 (只用符号)
    theta -= np.sign(g) * delta
    prev_grad = g

    print(f"迭代 {t+1}: theta = {theta[0]:.6f}")

print("优化后的参数:", theta)
```

运行后，参数 $\theta$ 会逐渐收敛到 0（目标函数最小值）。

## PyTorch Rprop 使用示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 一个简单的线性模型 y = wx + b
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 构造数据 y = 2x + 3
x = torch.randn(100, 1)
y = 2 * x + 3 + 0.1 * torch.randn(100, 1)  # 加点噪声

# 定义模型、损失函数和 Rprop 优化器
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Rprop(model.parameters(), lr=0.01)

# 训练
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")

# 打印学到的参数
w, b = model.linear.weight.item(), model.linear.bias.item()
print(f"学到的参数: w = {w:.2f}, b = {b:.2f}")
```



###  解释

1. **模型**：`SimpleModel` 定义了一个单层线性模型。
2. **数据**：我们用 `y = 2x + 3` 加上一点噪声作为训练目标。
3. **优化器**：使用 `torch.optim.Rprop`，注意它只依赖梯度符号，因此学习率含义和 SGD 不太一样。
4. **训练循环**：标准的前向传播 → 计算损失 → 反向传播 → 参数更新。
5. **结果**：最后学到的参数 `w` 接近 2，`b` 接近 3。


