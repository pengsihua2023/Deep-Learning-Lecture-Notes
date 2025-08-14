## PINN：求解Navier-Stokes 方程
是的，物理信息神经网络（PINNs）可以用来求解 **Navier-Stokes 方程**，这是一组描述流体运动的非线性偏微分方程，广泛应用于流体力学、天气预报、航空航天等领域。PINNs 通过将 Navier-Stokes 方程的残差、初始条件和边界条件嵌入神经网络的损失函数，利用自动微分计算导数，逼近方程的解。以下是详细介绍，包括原理、实现方法以及一个简单的 PyTorch 代码示例，用于求解二维不可压缩 Navier-Stokes 方程，并加入可视化。

---

### 1. **Navier-Stokes 方程简介**
<img width="996" height="841" alt="image" src="https://github.com/user-attachments/assets/4ea5b9ed-db47-4f80-b31e-03c081bcace6" />


---

### 2. **PINNs 求解 Navier-Stokes 方程的原理**
<img width="886" height="424" alt="image" src="https://github.com/user-attachments/assets/4c9894da-9101-4002-afdc-fc1ae24a933a" />

---

### 3. **简单代码示例：二维 Navier-Stokes 方程**
<img width="995" height="271" alt="image" src="https://github.com/user-attachments/assets/3bd349fd-08c2-4bb5-a1c6-a25ab09cc218" />


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
- **网络**：
  - 输入：\( (x, y, t) \)，输出：\( (u, v, p) \)。
  - 使用 3 层全连接网络，50 个神经元，Tanh 激活函数。
- **损失函数**：
  - PDE 残差：动量方程（x 和 y 方向）+ 连续性方程。
  - 初始条件：\( u(x, y, 0) = \sin(\pi x) \cos(\pi y), v(x, y, 0) = -\cos(\pi x) \sin(\pi y) \)。
  - 边界条件：无滑移，边界上 \( u = v = 0 \)。
  - 损失加权：初始和边界损失权重设为 10，加强约束。
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
4. **可视化**：
   - 需要安装 `matplotlib` 和 `imagemagick`（用于保存 GIF）。
   - 可添加 \( v \) 和 \( p \) 的热图，或绘制速度矢量场（`plt.quiver`）。
5. **扩展**：
   - **反问题**：将 \( \nu \) 或 \( \rho \) 设为可学习参数（`nn.Parameter`），加入观测数据。
   - **复杂几何**：定义复杂边界条件（如圆形障碍物）。
   - **三维问题**：扩展输入为 \( (x, y, z, t) \)，输出为 \( (u, v, w, p) \)。

#### **扩展：反问题**
若要估计 \( \nu \)，可修改 `PINN` 类：
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
