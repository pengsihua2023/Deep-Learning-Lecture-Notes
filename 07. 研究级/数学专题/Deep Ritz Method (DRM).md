## Deep Ritz Method (DRM)
Deep Ritz Method (DRM) 是一种基于深度学习的数值方法，用于求解变分问题，特别是那些源于偏微分方程 (PDE) 的变分形式。它由 Weinan E 和 Bing Yu 于 2017 年提出，灵感来源于经典的 Ritz 方法，该方法通过最小化能量泛函来逼近 PDE 的解。DRM 使用深度神经网络作为试函数（trial function），通过蒙特卡罗采样近似积分来最小化变分能量，从而实现网格无关（mesh-free）的求解。相比于 Physics-Informed Neural Networks (PINN)，DRM 更注重变分弱形式，适用于椭圆型 PDE（如 Poisson 方程）、高维问题和非线性 PDE，具有自然的自适应性和非线性能力，但可能在边界条件处理上需要额外罚项。

DRM 的优势在于能处理高维问题（如金融中的期权定价或量子力学），并已被扩展到分数阶 PDE、线性弹性等。它的局限性包括对能量泛函的依赖（不是所有 PDE 都有变分形式）和训练时的数值积分误差。

<img width="932" height="982" alt="image" src="https://github.com/user-attachments/assets/33a8a6e4-6900-4473-9614-93f8673dfb8b" />

### 数学描述

考虑一个典型的椭圆型 PDE，例如 Poisson 方程：在域 $\Omega \subset \mathbb{R}^d\$ 上，

$$
-\Delta u(x) = f(x), \quad x \in \Omega,
$$

伴随 Dirichlet 边界条件：

$$
u(x) = g(x), \quad x \in \partial \Omega.
$$

其变分形式是通过最小化能量泛函 \$I\[u]\$ 在 Sobolev 空间 \$H^1\_g(\Omega)\$ （满足边界条件的函数空间）上实现的：

$$
I[u] = \int_\Omega \left( \frac{1}{2} |\nabla u(x)|^2 - f(x)u(x) \right) dx,
$$

其中最小化器 $u\$ 即为 PDE 的弱解（根据 Ritz 定理）。

在 DRM 中，使用参数化神经网络 $u\_\theta(x)\$ （ $\theta\$ 为网络参数）逼近 $u(x)\$ 。由于直接计算积分困难，采用蒙特卡罗采样近似损失函数：

$$
J(\theta) = \frac{1}{N_\Omega} \sum_{i=1}^{N_\Omega} \left( \frac{1}{2} |\nabla u_\theta(x_i)|^2 - f(x_i)u_\theta(x_i) \right) 
 +  \frac{\lambda}{N_{\partial \Omega}} \sum_{j=1}^{N_{\partial \Omega}} |u_\theta(y_j) - g(y_j)|^2,
$$

其中：

* $x\_i \sim \mathcal{U}(\Omega)\$ 是从内部域均匀随机采样的点（ $N\_\Omega\$ 个样本）。
* $y\_j \sim \mathcal{U}(\partial \Omega)\$ 是从边界均匀随机采样的点（ $N\_{\partial \Omega}\$ 个样本）。
* $\lambda > 0\$ 是罚项权重，用于软强制边界条件（也可用硬约束，如修改网络架构）。
* 梯度 $\nabla u\_\theta\$ 通过自动微分计算。

训练通过随机梯度下降（SGD）最小化 $J(\theta)\$ 。对于更一般的变分问题，泛函可为

$$
I[u] = \int_\Omega F(x, u, \nabla u) dx,
$$

损失类似地构造。DRM 的收敛性在某些条件下已分析，如对于线性 PDE 的两层网络。

---

### 代码实现

下面是用 PyTorch 实现的一个简单 1D DRM 例子，用于求解 Poisson 方程：

$$
-u''(x) = \pi^2 \sin(\pi x), \quad x \in [0,1],
$$

边界条件：

$$
u(0) = u(1) = 0.
$$

真实解为：

$$
u(x) = \sin(\pi x).
$$

代码包括神经网络定义、变分损失计算和训练循环。



```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 神经网络逼近 u(x)
class DRMNet(nn.Module):
    def __init__(self, hidden_size=50, num_layers=3):
        super(DRMNet, self).__init__()
        self.fc_in = nn.Linear(1, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc_in(x))
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.fc_out(x)

# 变分能量损失：(1/2) ∫ (u')^2 dx - ∫ f u dx
def energy_loss(model, x_int, f_int):
    x_int.requires_grad_(True)
    u = model(x_int)
    u_x = torch.autograd.grad(u, x_int, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    term1 = 0.5 * (u_x ** 2)  # (1/2) |∇u|^2
    term2 = f_int * u  # f u
    return torch.mean(term1 - term2)

# 边界罚项：u(0)^2 + u(1)^2
def boundary_penalty(model):
    x0 = torch.tensor([[0.0]])
    x1 = torch.tensor([[1.0]])
    return model(x0)**2 + model(x1)**2

# 训练
def train(model, optimizer, num_epochs=5000, num_int_samples=100, num_bd_samples=10, lambda_bd=10.0):
    losses = []
    for epoch in range(num_epochs):
        # 内部采样
        x_int = torch.rand((num_int_samples, 1))
        f_int = (np.pi ** 2) * torch.sin(np.pi * x_int)
        loss_energy = energy_loss(model, x_int, f_int)
        # 边界采样（简单1D边界）
        loss_bd = boundary_penalty(model)
        loss = loss_energy + lambda_bd * loss_bd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return losses

# 主程序
model = DRMNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses = train(model)

# 测试
x_test = torch.linspace(0, 1, 100).unsqueeze(1)
u_pred = model(x_test).detach().numpy()
u_true = np.sin(np.pi * x_test.numpy())
plt.plot(x_test.numpy(), u_pred, label='Prediction')
plt.plot(x_test.numpy(), u_true, label='True')
plt.legend()
plt.show()
```

### 代码解释
1. **DRMNet**：一个简单的全连接网络，使用 tanh 激活，输入 1 维（x），输出 1 维（u(x)）。
2. **energy_loss**：计算变分能量的蒙特卡罗近似，使用自动微分求导。
3. **boundary_penalty**：软罚项强制 Dirichlet 边界条件（对于复杂边界，可随机采样边界点）。
4. **train**：每轮随机采样内部点，计算总损失（能量 + 边界罚项），优化网络。运行后，损失下降，预测曲线接近真实解。

这个实现是最简单的演示；实际中可扩展到高维 PDE（如使用更高维度输入）或添加自适应采样。如果运行代码，需要 PyTorch 环境。
