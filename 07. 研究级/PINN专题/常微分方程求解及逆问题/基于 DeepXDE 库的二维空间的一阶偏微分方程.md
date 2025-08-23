## 基于 DeepXDE 库的二维空间的一阶偏微分方程

二维线性传输方程：

$$
u_t + a u_x + b u_y = 0, \quad (x,y,t)\in(0,1)\times(0,1)\times(0,1],
$$

初始条件：

$$
u(x,y,0) = \sin(2\pi x)\sin(2\pi y),
$$

解析解：

$$
u(x,y,t) = \sin\big(2\pi(x-a t)\big)\sin\big(2\pi(y-b t)\big).
$$

其中 $(a,b)$ 是速度向量。

---

### 代码示例（DeepXDE + PyTorch）

```python
# pip install deepxde torch matplotlib
import numpy as np
import deepxde as dde

# 使用 PyTorch 后端
dde.backend.set_default_backend("pytorch")

# 常数
a, b = 1.0, 0.5
pi2 = 2 * np.pi

# PDE: u_t + a u_x + b u_y = 0
def pde(x, u):
    du_x = dde.grad.jacobian(u, x, i=0)   # ∂u/∂x
    du_y = dde.grad.jacobian(u, x, i=1)   # ∂u/∂y
    du_t = dde.grad.jacobian(u, x, i=2)   # ∂u/∂t
    return du_t + a * du_x + b * du_y

# 真解
def true_solution(x):
    return np.sin(pi2*(x[:,0:1] - a*x[:,2:3])) * np.sin(pi2*(x[:,1:2] - b*x[:,2:3]))

# 初始条件 t=0
def ic_func(x):
    return np.sin(pi2*x[:,0:1]) * np.sin(pi2*x[:,1:2])

def on_initial(x, on_boundary):
    return on_boundary and np.isclose(x[2], 0.0)

# 入流边界条件：需要在速度方向上的入流面上施加 Dirichlet 条件
def inflow_bc(x):
    return true_solution(x)

def on_x0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

def on_y0(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)

# 几何与时间区域
geom = dde.geometry.Rectangle([0,0],[1,1])    # 空间 (x,y)
timedomain = dde.geometry.TimeDomain(0,1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 条件定义
ic = dde.icbc.IC(geomtime, ic_func, on_initial)
bc_x0 = dde.icbc.DirichletBC(geomtime, inflow_bc, on_x0)
bc_y0 = dde.icbc.DirichletBC(geomtime, inflow_bc, on_y0)

# 数据
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc_x0, bc_y0],
    num_domain=10000,
    num_boundary=2000,
    num_initial=2000,
    train_distribution="pseudo",
)

# 网络
net = dde.maps.FNN([3] + [64]*4 + [1], "tanh", "Glorot uniform")

# 模型
model = dde.Model(data, net)

# 训练
model.compile("adam", lr=1e-3)
model.train(epochs=15000, display_every=1000)
model.compile("L-BFGS")
model.train()

# 误差
X_test = geomtime.random_points(2000)
u_pred = model.predict(X_test)
u_true = true_solution(X_test)
rel_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
print("Relative L2 error:", rel_l2)
```

---

### 说明

* **输入维度**：这里的输入是 $(x,y,t)$，所以网络输入是 3 维。
* **边界条件**：

  * 如果 $a>0$，则在 $x=0$ 需要入流条件；如果 $b>0$，则在 $y=0$ 需要入流条件。
  * 若 $a<0$ 或 $b<0$，则相应的入流边界应换成 $x=1$ 或 $y=1$。
* **网络结构**：四层 64 宽的全连接网络，一般足够。可根据精度需求调整。
* **采样数量**：`num_domain` / `num_boundary` / `num_initial` 会影响训练稳定性和精度。

---

