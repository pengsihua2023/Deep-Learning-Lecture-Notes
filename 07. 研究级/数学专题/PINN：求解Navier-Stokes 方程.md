## PINN：求解Navier-Stokes 方程
是的，物理信息神经网络（PINNs）可以用来求解 **Navier-Stokes 方程**，这是一组描述流体运动的非线性偏微分方程，广泛应用于流体力学、天气预报、航空航天等领域。PINNs 通过将 Navier-Stokes 方程的残差、初始条件和边界条件嵌入神经网络的损失函数，利用自动微分计算导数，逼近方程的解。以下是详细介绍，包括原理、实现方法以及一个简单的 PyTorch 代码示例，用于求解二维不可压缩 Navier-Stokes 方程，并加入可视化。

---

### 1. **Navier-Stokes 方程简介**

Navier–Stokes 方程描述流体的速度场 \$\mathbf{u} = (u, v)\$ 和压力场 \$p\$。对于二维不可压缩流体，Navier–Stokes 方程的形式为：

### 动量方程：

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} 
= -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} 
= -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

### 连续性方程（不可压缩条件）：

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$


其中：

* \$u, v\$: 速度分量（沿 \$x, y\$ 方向）。
* \$p\$: 压力。
* \$\rho\$: 流体密度（常设为 1）。
* \$\nu\$: 动黏性系数。
* 定义域： \$(x,y) \in \Omega, , t \in \[0, T]\$。


### 初始和边界条件：

* **初始条件**:

$$
u(x,y,0) = u_0(x,y), \quad v(x,y,0) = v_0(x,y)
$$

* **边界条件**: 例如 Dirichlet 条件

$$
u = g_u, \quad v = g_v
$$

或 Neumann 条件。


### 2. **PINNs 求解 Navier-Stokes 方程的原理**


PINNs 使用神经网络逼近速度场 \$u(x,y,t), v(x,y,t)\$ 和压力场 \$p(x,y,t)\$， 通过以下步骤求解：

1. **神经网络**: 定义一个网络，输入为 \$(x,y,t)\$，输出为 \$(u,v,p)\$。

2. **损失函数**:

   * **PDE 残差**: 计算动量方程和连续性方程的残差。
   * **初始条件**: 确保 \$t=0\$ 时满足初始速度场。
   * **边界条件**: 确保边界上满足指定条件（如无滑移边界）。
   * **总损失**: $L = L_{\text{PDE}} + L_{\text{init}} + L_{\text{bc}}$ 

3. **自动微分**: 使用 PyTorch 的 `torch.autograd` 计算偏导数（如 $\frac{\partial u}{\partial t}, \quad \frac{\partial^2 u}{\partial x^2}$ ）。

4. **优化**: 通过优化器（如 Adam）最小化损失函数，调整网络参数。




### 3. **简单代码示例：二维 Navier-Stokes 方程**

以下是一个简化的 PyTorch 代码示例，求解二维不可压缩 Navier–Stokes 方程在一个矩形区域 $\[0,1] \times \[0,1]\$，时间范围 $\[0,1]\$。我们假设：

* **初始条件**: $u(x,y,0) = \sin(\pi x) \cos(\pi y), \quad v(x,y,0) = -\cos(\pi x) \sin(\pi y).$ 

* **边界条件**: 无滑移边界， $u=v=0\$ 在边界。

* **参数**: $\rho = 1, \quad \nu = 0.01.$ 

* **问题**: 稳态或瞬态流体运动，连续性方程确保不可压缩性。



#### **代码**
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 定义 PINN 网络
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),  # 输入 (x, y, t)
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)  # 输出 (u, v, p)
        )

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)

# 2. 定义损失函数
def compute_loss(model, x_f, y_f, t_f, x_i, y_i, t_i, u_i, v_i, x_b, y_b, t_b, u_b, v_b, rho=1.0, nu=0.01):
    x_f, y_f, t_f = x_f.requires_grad_(True), y_f.requires_grad_(True), t_f.requires_grad_(True)
    uvp = model(x_f, y_f, t_f)
    u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]

    # PDE 残差
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    v_t = torch.autograd.grad(v, t_f, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x_f, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y_f, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_f, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_f, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x_f, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y_f, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    # Navier-Stokes 动量方程
    pde_u = u_t + u * u_x + v * u_y + (1/rho) * p_x - nu * (u_xx + u_yy)
    pde_v = v_t + u * v_x + v * v_y + (1/rho) * p_y - nu * (v_xx + v_yy)
    # 连续性方程
    pde_cont = u_x + v_y
    
    loss_pde = torch.mean(pde_u**2 + pde_v**2 + pde_cont**2)
    
    # 初始条件
    uvp_i = model(x_i, y_i, t_i)
    u_init, v_init = uvp_i[:, 0:1], uvp_i[:, 1:2]
    loss_init = torch.mean((u_init - u_i)**2 + (v_init - v_i)**2)
    
    # 边界条件
    uvp_b = model(x_b, y_b, t_b)
    u_bc, v_bc = uvp_b[:, 0:1], uvp_b[:, 1:2]
    loss_bc = torch.mean((u_bc - u_b)**2 + (v_bc - v_b)**2)
    
    # 总损失
    return loss_pde + 10 * loss_init + 10 * loss_bc

# 3. 准备数据
N_f = 10000  # PDE 内部点
N_i = 200    # 初始点
N_b = 400    # 边界点

# PDE 内部点
x_f = torch.rand(N_f, 1)
y_f = torch.rand(N_f, 1)
t_f = torch.rand(N_f, 1)

# 初始条件
x_i = torch.rand(N_i, 1)
y_i = torch.rand(N_i, 1)
t_i = torch.zeros(N_i, 1)
u_i = torch.sin(np.pi * x_i) * torch.cos(np.pi * y_i)
v_i = -torch.cos(np.pi * x_i) * torch.sin(np.pi * y_i)

# 边界条件（无滑移）
x_b_left = torch.zeros(N_b//4, 1)  # x=0
y_b_left = torch.rand(N_b//4, 1)
t_b_left = torch.rand(N_b//4, 1)
x_b_right = torch.ones(N_b//4, 1)  # x=1
y_b_right = torch.rand(N_b//4, 1)
t_b_right = torch.rand(N_b//4, 1)
x_b_bottom = torch.rand(N_b//4, 1)  # y=0
y_b_bottom = torch.zeros(N_b//4, 1)
t_b_bottom = torch.rand(N_b//4, 1)
x_b_top = torch.rand(N_b//4, 1)  # y=1
y_b_top = torch.ones(N_b//4, 1)
t_b_top = torch.rand(N_b//4, 1)

x_b = torch.cat([x_b_left, x_b_right, x_b_bottom, x_b_top], dim=0)
y_b = torch.cat([y_b_left, y_b_right, y_b_bottom, y_b_top], dim=0)
t_b = torch.cat([t_b_left, t_b_right, t_b_bottom, t_b_top], dim=0)
u_b = torch.zeros(N_b, 1)
v_b = torch.zeros(N_b, 1)

# 4. 训练模型
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []

for epoch in range(10000):
    optimizer.zero_grad()
    loss = compute_loss(model, x_f, y_f, t_f, x_i, y_i, t_i, u_i, v_i, x_b, y_b, t_b, u_b, v_b)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 5. 可视化
# 损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# 速度场动画 (t=0 到 t=1)
nx, ny = 50, 50
x_grid = torch.linspace(0, 1, nx).reshape(-1, 1)
y_grid = torch.linspace(0, 1, ny).reshape(-1, 1)
X, Y = torch.meshgrid(x_grid.squeeze(), y_grid.squeeze(), indexing='ij')
X_flat, Y_flat = X.reshape(-1, 1), Y.reshape(-1, 1)

fig, ax = plt.subplots()
def animate(t_frame):
    ax.clear()
    t_test = torch.ones_like(X_flat) * (t_frame / 50.0)
    uvp_pred = model(X_flat, Y_flat, t_test)
    u_pred = uvp_pred[:, 0].detach().numpy().reshape(nx, ny)
    ax.imshow(u_pred, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'u(x, y, t={t_frame/50:.2f})')
ani = FuncAnimation(fig, animate, frames=50, interval=100)
plt.close(fig)
ani.save('navier_stokes_animation.gif', writer='imagemagick')  # 需要 imagemagick
plt.show()
```

#### **代码说明**
<img width="804" height="364" alt="image" src="https://github.com/user-attachments/assets/856b5c44-c6dd-4820-883a-8e632df4b674" />

- **数据**：
  - 随机采样内部点（10000）、初始点（200）、边界点（400）。
  - 初始条件模拟旋涡流场，边界为无滑移。
- **训练**：
  - 使用 Adam 优化器，10000 次迭代，记录损失。
- **可视化**：
  - 损失曲线：显示训练收敛性。
  - 动画：绘制 \( u(x, y, t) \) 的热图随时间变化，保存为 GIF。

#### **注意事项**
1. **计算复杂度**：
   - Navier-Stokes 方程非线性强，训练可能较慢，建议使用 GPU（`model.cuda(), x_f.cuda(), ...`）。
   - 可尝试 L-BFGS 优化器提高收敛性。
2. **采样点**：
   - 增加采样点（`N_f, N_i, N_b`）可提高精度，但增加计算成本。
   - 确保边界点均匀分布，避免偏向。
3. **初始条件**：
   - 这里使用解析初始条件，实际应用可从实验数据或数值模拟获取。
<img width="772" height="404" alt="image" src="https://github.com/user-attachments/assets/6fc65d92-229d-4769-8bdb-114fc8e456db" />

* **网络：**

  * 输入：\$(x, y, t)\$，输出：\$(u, v, p)\$。
  * 使用 3 层全连接网络，50 个神经元，Tanh 激活函数。

* **损失函数：**

  * PDE 残差：动量方程（\$x\$ 和 \$y\$ 方向）+ 连续性方程。
  * 初始条件：\$u(x, y, 0) = \sin(\pi x) \cos(\pi y), ; v(x, y, 0) = - \cos(\pi x) \sin(\pi y)\$。
  * 边界条件：无滑移，边界上 \$u = v = 0\$。
  * 损失加权：初始和边界损失权重设为 \$10\$，加强约束。


```python
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 3)
        )
        self.nu = nn.Parameter(torch.tensor(0.1))  # 可学习 nu

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)
```
在 `compute_loss` 中使用 `model.nu`，并添加观测数据损失（类似前述反问题示例）。

#### **总结**
PINNs 可以有效求解 Navier-Stokes 方程，代码通过嵌入动量和连续性方程的残差，结合初始和边界条件，逼近速度和压力场。上述示例展示了二维问题的基础实现，包含可视化动画。
