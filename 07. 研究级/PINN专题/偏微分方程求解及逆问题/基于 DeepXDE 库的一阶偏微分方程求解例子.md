下面给你一个用 **DeepXDE**（基于 PyTorch）求解一阶偏微分方程的完整示例：线性对流方程

$$
u_t + cu_x = 0,\quad (x,t)\in(0,1)\times(0,1],\quad
u(x,0)=\sin(2\pi x),\ c=1.
$$

此问题有解析解 $u(x,t)=\sin(2\pi(x-ct))$。对 $c>0$ 的超定式超曲型方程只需要在入流边界 $x=0$ 施加狄利克雷边界：

$$
u(0,t)=\sin(-2\pi t).
$$

---

### 代码（可直接运行）

```python
# pip install deepxde torch matplotlib
import numpy as np
import deepxde as dde

# 使用 PyTorch 后端
dde.backend.set_default_backend("pytorch")
from deepxde.backend import torch

# 常数
c = 1.0
pi2 = 2 * np.pi

# PDE: u_t + c u_x = 0
def pde(x, u):
    du_x = dde.grad.jacobian(u, x, i=0)     # ∂u/∂x, x在第0维
    du_t = dde.grad.jacobian(u, x, i=1)     # ∂u/∂t, t在第1维
    return du_t + c * du_x

# 解析解（用于IC、边界与误差评估）
def true_solution(x):
    # x: (N,2) -> [:,0]=space, [:,1]=time
    return np.sin(pi2 * (x[:, 0:1] - c * x[:, 1:2]))

# 初始条件: t=0
def ic_func(x):
    return np.sin(pi2 * x[:, 0:1])  # u(x,0)

# 入流边界 x=0
def inflow_bc(x):
    # u(0,t) = sin(2π(0 - c t)) = sin(-2π t)
    return np.sin(-pi2 * x[:, 1:2])

# 选取边界：x=0
def on_x0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

# 几何与时间区域
geom = dde.geometry.Interval(0.0, 1.0)
timedomain = dde.geometry.TimeDomain(0.0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 条件定义
ic = dde.icbc.IC(geomtime, ic_func, lambda x, on_b: on_b and np.isclose(x[1], 0.0))
bc_in = dde.icbc.DirichletBC(geomtime, inflow_bc, on_x0)

# 组建数据（采样点数量可按需调整）
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc_in],
    num_domain=8000,
    num_boundary=2000,
    num_initial=2000,
    train_distribution="pseudo",
)

# 网络
net = dde.maps.FNN([2] + [64] * 4 + [1], "tanh", "Glorot uniform")

# 模型
model = dde.Model(data, net)

# 训练（先 Adam 再 L-BFGS）
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=15000, display_every=1000)
model.compile("L-BFGS")
losshistory, train_state = model.train()

# 误差评估
X_test = geomtime.random_points(2000)
u_pred = model.predict(X_test)
u_true = true_solution(X_test)
rel_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
print("Relative L2 error:", rel_l2)

# 可视化：取定多个时刻画 u(x,t)
import matplotlib.pyplot as plt
xs = np.linspace(0, 1, 200)
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    XT = np.stack([xs, np.full_like(xs, t)], axis=1)
    u_p = model.predict(XT).squeeze()
    u_t = true_solution(XT).squeeze()
    plt.plot(xs, u_p, label=f"PINN t={t:.2f}")
    plt.plot(xs, u_t, "--", label=f"True t={t:.2f}")
plt.xlabel("x"); plt.ylabel("u(x,t)")
plt.title("Advection: u_t + u_x = 0")
plt.legend(ncol=2, fontsize=8)
plt.show()
```

---

### 说明与要点

* **问题类型**：一阶双曲型 PDE（线性对流）。PINN 非常适合这类初值—入流边值问题。
* **边界设置**：对 $c>0$ 只在 **入流** 边界 $x=0$ 施加狄利克雷条件即可；出流端不必强加条件。
* **损失构成**：DeepXDE 会综合 PDE 残差、初始条件、边界条件的 MSE。
* **网络与采样**：隐藏层宽度/层数与采样点数（`num_domain/num_boundary/num_initial`）会显著影响收敛。上面的值在普通笔记本上即可跑通；若误差偏大，可增加采样或训练轮数。
* **验证**：提供了解析解用于计算相对 $L^2$ 误差与对比曲线。

---



