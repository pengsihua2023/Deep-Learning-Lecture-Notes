# ASGD（Averaged Stochastic Gradient Descent）优化器


## 1. 定义

**ASGD (Averaged Stochastic Gradient Descent)** 是 **SGD** 的一种改进版，核心思想是：

* 普通 SGD 在训练过程中参数更新波动较大，收敛过程中可能不稳定。
* ASGD 在 SGD 的基础上引入了 **参数平均（Polyak-Ruppert averaging）**：

  * 除了正常的参数更新外，还维护一个 **平均参数向量**。
  * 最终使用平均后的参数作为模型权重，从而减少噪声，提高泛化能力。

它的典型应用是在凸优化任务（如线性模型、逻辑回归）中能更快收敛并达到更优解。


## 2. 数学公式

设：

* 参数： $\theta_t$
* 学习率： $\eta_t$
* 梯度： $g_t = \nabla_\theta f_t(\theta_t)$
* 平均参数： $\bar{\theta}_t$
* 衰减因子： $\lambda$

### 更新步骤：

1. **SGD 更新**：

$$
\theta_{t+1} = \theta_t - \eta_t \cdot g_t
$$

2. **平均参数更新**（从某个迭代 $t_0$ 开始）：

$$
\bar{\theta}_{t+1} = \frac{1}{t - t_0 + 1} \sum_{k=t_0}^{t} \theta_k
$$

最后得到的模型参数为：

$$
\theta^* = \bar{\theta}_T
$$

这样可以平滑掉 SGD 的震荡，提高收敛稳定性。



## 3. 最简代码例子

用 **PyTorch** 实现一个极简版 ASGD 训练线性模型：

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
y = 2 * x + 3 + 0.1 * torch.randn(100, 1)

# 定义模型、损失函数和 ASGD 优化器
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.ASGD(model.parameters(), lr=0.05)

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



### 解释

1. `torch.optim.ASGD` 内部实现了参数平均。
2. `lr=0.05` 是学习率，可以根据任务调整。
3. 最终得到的参数会比普通 SGD 更稳定，更接近真实值（这里接近 `w=2, b=3`）。


