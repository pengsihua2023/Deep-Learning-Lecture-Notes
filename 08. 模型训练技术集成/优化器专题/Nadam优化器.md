# Nadam 优化器


## 1. 定义

**Nadam (Nesterov-accelerated Adaptive Moment Estimation)** 是一种深度学习常用的优化算法，它结合了 **Adam** 和 **Nesterov 动量（Nesterov momentum）** 的思想。

* Adam：通过一阶矩估计（动量）和二阶矩估计（梯度平方的指数加权平均）来动态调整学习率。
* Nesterov 动量：改进的动量法，通过提前看一步（look-ahead）来加快收敛并减少震荡。

Nadam 就是在 Adam 的更新公式中引入了 Nesterov 动量，从而在自适应学习率的同时利用前瞻性的梯度修正。


## 2. 数学公式

设：

* 参数： $\theta_t$
* 损失函数梯度： $g_t = \nabla_\theta f_t(\theta_{t-1})$
* 一阶矩（动量）： $m_t$
* 二阶矩： $v_t$
* 学习率： $\alpha$
* 衰减系数： $\beta_1, \beta_2 \in [0,1)$
* 数值稳定项： $\epsilon$

公式步骤：

1. 梯度计算：

$$
g_t = \nabla_\theta f_t(\theta_{t-1})
$$

2. 一阶与二阶矩更新：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

3. 偏差修正（bias correction）：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

4. **Nesterov 修正**：

$$
\tilde{m}_t = \beta_1 \hat{m}_t + \frac{(1-\beta_1)}{1-\beta_1^t} g_t
$$

5. 参数更新：

$$
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\tilde{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

---

## 3. 最简代码例子

用 **NumPy** 实现 Nadam 的一个极简版（只展示核心更新逻辑）：

```python
import numpy as np

# 初始化参数
theta = np.array([1.0])       # 初始参数
m, v = 0, 0                   # 一阶矩 & 二阶矩
beta1, beta2 = 0.9, 0.999
alpha, eps = 0.001, 1e-8

# 一个简单的目标函数 f(x) = x^2
def grad(theta):
    return 2 * theta

# 迭代更新
for t in range(1, 101):
    g = grad(theta)

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    m_nesterov = beta1 * m_hat + (1 - beta1) / (1 - beta1 ** t) * g

    theta -= alpha * m_nesterov / (np.sqrt(v_hat) + eps)

print("优化后的参数:", theta)
```

这段代码会将参数 $\theta$ 逐渐逼近 0（即目标函数 $f(x)=x^2$ 的最小值）。

## PyTorch Nadam 使用示例

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

# 定义模型、损失函数和 Nadam 优化器
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.NAdam(model.parameters(), lr=0.01)

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


### 代码解释

1. **模型**：定义了一个最简单的线性模型 `y = wx + b`。
2. **数据**：随机生成输入 `x`，输出 `y` 按照真实函数 `2x + 3` 加上少量噪声。
3. **优化器**：用 `torch.optim.NAdam`，设置 `lr=0.01`。
4. **训练循环**：前向传播 → 计算损失 → 反向传播 → 参数更新。
5. **结果**：训练后，参数 `w, b` 会非常接近 `2` 和 `3`。


