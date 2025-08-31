## Discontinuous Galerkin (DG) Based Neural Networks
Discontinuous Galerkin (DG) Based Neural Networks 是一种新兴的混合方法，将不连续Galerkin (DG) 方法（一种用于求解偏微分方程 PDE 的数值技术，允许解在网格元素边界不连续）与神经网络结合，用于高效求解 PDE，特别是高维、非线性或具有不连续解的问题。它通过神经网络参数化分段试验函数（trial functions），或丰富 DG 基函数，以捕捉复杂动态，同时保留 DG 的局部性和灵活性。这种方法在处理复杂几何、扰动或稳态问题时表现出色，已应用于 Poisson 方程、热方程和双曲平衡律等。

典型变体包括：
- **DGNet**：受内部罚项 DG (IPDG) 启发，使用分段神经网络作为试验空间，分段多项式作为测试空间，提高准确性和训练效率。
- **Local Randomized Neural Networks with DG (LRNN-DG)**：结合随机化 NN 在子域上逼近，并用 DG 耦合，提高效率。
- **DG-PINNs**：使用 Physics-Informed Neural Networks (PINN) 丰富 DG 基函数，实现近似 well-balanced 属性，适用于平衡律。

### 数学描述

考虑一个一般 PDE，例如 Poisson 方程：

$$
-\Delta u = f \quad \text{在域 } \Omega \subset \mathbb{R}^d 上,
$$

伴随 Dirichlet 边界条件

$$
u = g \quad \text{在 } \partial \Omega 上.
$$

在经典 DG 方法中，域 $\Omega\_h\$ 离散化为网格 $\Omega\_h = \bigcup\_{i=1}^N E\_i\$ （元素 $E\_i\$ ），试验和测试空间均为分段多项式 $V\_h^k = { v: v|\_{E\_i} \in \mathcal{P}\_k(E\_i) }\$ ，允许边界不连续。通过弱形式求解：找 $u\_h \in V\_h^k\$ ，使得 $\forall v\_h \in V\_h^k\$ :

<img width="724" height="70" alt="image" src="https://github.com/user-attachments/assets/634cae4e-29b9-48a8-9b6c-bb9d8c1f6441" />


其中 ${ \cdot }\$ 是平均和跳跃算子， $\alpha\$ 是罚项参数（IPDG）。

在 DG Based Neural Networks 中（如 DGNet），试验空间替换为分段神经网络空间 $\mathcal{N}_ {\Omega_h} = \{ u_\theta: u_\theta|_ {E_i} \in \mathcal{N}_{l,\text{nn}}(\theta_i) \},$ 其中 $\mathcal{N}\_{l,\text{nn}}(\theta\_i)\$ 是浅层 NN（层数 $L \leq 2\$ ，隐藏单元 r）在元素 $E\_i\$ 上空间，每个元素有独立参数 $\theta\_i\$ :

$$
u_\theta(x) = \sum_{i=1}^N u^i_{\text{NN}}(x; \theta_i) \chi_{E_i}(x),
$$

其中 $u^i\_{\text{NN}}\$ 是 NN， $\chi\_{E\_i}\$ 是指示函数。

测试空间保持为分段多项式 $V\_h^k\$ 。弱形式类似，但通过最小化残差损失训练：

<img width="7543" height="74" alt="image" src="https://github.com/user-attachments/assets/96c73b51-a207-4110-abfe-3296c8ff0526" />


其中积分通过蒙特卡罗采样或正交点测近似，梯度用自动微分计算。训练最小化 $J(\theta)\$ 以优化 \$\theta\$ 。

对于时间相关 PDE（如抛物方程 $u\_t - \Delta u = f\$ ），可扩展到时空 DG，或使用时间步法。

在 DG-PINNs 中，DG 基函数通过 PINN 形式替换：标准 DG 基 $\phi\_j\$ 加上 NN 逼近的稳定项 $\psi\_\theta\$ ，以实现 well-balanced（稳态解的精确捕捉）。

---

### 代码实现

下面是用 PyTorch 实现的一个简单 1D DGNet 例子，用于求解 Poisson 方程：

$$
-u''(x) = \pi^2 \sin(\pi x), \quad x \in [0,1],
$$

边界条件

$$
u(0) = u(1) = 0.
$$

真实解为

$$
u(x) = \sin(\pi x).
$$

我们将域划分成 4 个元素，每个元素一个独立浅层 NN（单隐藏层），使用 IPDG 弱形式作为损失。测试函数为分段线性多项式（k=1），积分用蒙特卡罗采样近似。



```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 局部 NN：每个元素一个单隐藏层 NN
class LocalNN(nn.Module):
    def __init__(self, hidden_size=20):
        super(LocalNN, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))

# DGNet：分段 NN
class DGNet(nn.Module):
    def __init__(self, num_elements=4, hidden_size=20):
        super(DGNet, self).__init__()
        self.num_elements = num_elements
        self.element_size = 1.0 / num_elements
        self.locals = nn.ModuleList([LocalNN(hidden_size) for _ in range(num_elements)])
    
    def forward(self, x):
        u = torch.zeros_like(x)
        for i in range(self.num_elements):
            mask = (x >= i * self.element_size) & (x < (i+1) * self.element_size)
            local_x = x[mask] - i * self.element_size  # 映射到 [0, h]
            u[mask] = self.locals[i](local_x)
        return u

# 计算一阶导数
def grad_u(model, x):
    x.requires_grad_(True)
    u = model(x)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return du

# DG 损失：IPDG 弱形式近似（蒙特卡罗采样）
def dg_loss(model, num_int_samples=50, num_bd_samples=10, alpha=10.0, lambda_bd=10.0):
    # 内部积分点
    x_int = torch.rand((num_int_samples, 1))
    f = (np.pi ** 2) * torch.sin(np.pi * x_int)
    
    # 简单测试函数：分段线性 (v=1 或 v=x 在每个元素，但为简化，用随机点采样残差)
    du = grad_u(model, x_int)
    loss_int = torch.mean(du ** 2 / 2 - f * model(x_int))  # 简化变分形式（对称 IPDG 近似）
    
    # 边界罚项（元素边界跳跃 + 域边界）
    loss_jump = 0.0
    for i in range(1, model.num_elements):
        x_e = torch.tensor([[i * model.element_size]])
        u_left = model(x_e - 1e-5)
        u_right = model(x_e + 1e-5)
        du_left = grad_u(model, x_e - 1e-5)
        du_right = grad_u(model, x_e + 1e-5)
        jump_u = u_left - u_right
        avg_du = (du_left + du_right) / 2
        loss_jump += alpha * jump_u ** 2 - avg_du * jump_u  # IPDG 罚项
    
    # 域边界条件
    x0 = torch.tensor([[0.0]])
    x1 = torch.tensor([[1.0]])
    loss_bd = model(x0)**2 + model(x1)**2
    
    return loss_int + loss_jump / model.num_elements + lambda_bd * loss_bd

# 训练
model = DGNet()
optimizer = optim.Adam(model.parameters(), lr=0.005)
losses = []
for epoch in range(5000):
    loss = dg_loss(model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

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
1. **LocalNN**：每个元素独立的浅层 NN（tanh 激活）。
2. **DGNet**：组合局部 NN，分段评估。
3. **dg_loss**：近似 IPDG 弱形式损失，包括内部残差、边界跳跃罚项（[u] 和 {\nabla u}）和边界条件。使用自动微分计算导数，蒙特卡罗采样积分。
4. **train**：最小化损失优化所有局部 NN 参数。运行后，预测接近真实解。

这个是简化演示（1D，采样近似测试空间）；实际实现（如 DGNet）可使用正交规则积分多项式测试函数，或扩展到高维/时间相关 PDE。对于完整实现，可参考 DG-PINNs GitHub 中的 notebook（使用 PyTorch 训练 PINN 先验并丰富 DG 基）。
