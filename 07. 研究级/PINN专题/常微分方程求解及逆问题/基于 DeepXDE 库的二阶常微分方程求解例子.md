
## 一个二阶常微分方程** 的 DeepXDE 示例：简谐振子

$$
u''(x) + u(x) = 0,\quad x\in(0, 2\pi),\qquad
u(0)=1,\; u'(0)=0.
$$

解析解： $u^\*(x)=\cos x$ 。




```python
# pip install deepxde tensorflow matplotlib  (若尚未安装)

import numpy as np
import deepxde as dde
from deepxde.backend import tf  # 使用 TF 后端
import matplotlib.pyplot as plt

# 1) 区间 [0, 2π]
L = 2 * np.pi
geom = dde.geometry.Interval(0.0, L)

# 2) PDE 残差: u''(x) + u(x) = 0
def pde(x, y):
    dy_x   = dde.grad.jacobian(y, x, i=0, j=0)   # u'(x)
    d2y_xx = dde.grad.hessian(y, x, i=0, j=0)    # u''(x)
    return d2y_xx + y                             # u'' + u = 0

# 3) 仅在左端点 x=0 施加边界条件
def on_left(x, on_boundary):
    # DeepXDE 传入 x 为 [N, d]，这里 d=1
    return on_boundary and tf.less_equal(x[:, 0:1], 1e-8)

# Dirichlet: u(0) = 1
bc_u0 = dde.DirichletBC(geom, lambda x: 1.0, on_left)

# Neumann: u'(0) = 0 （对区间来说等价于法向导数）
bc_du0 = dde.NeumannBC(geom, lambda x: 0.0, on_left)

# 4) （可选）解析解用于评估
def exact(x):
    return np.cos(x)

# 5) 构建数据（域内/边界采样）
data = dde.data.PDE(
    geom,
    pde,
    [bc_u0, bc_du0],
    num_domain=200,        # 域内配点
    num_boundary=20,       # 边界采样点（两端都会采到，但只在 on_left 处起作用）
    solution=exact,
    num_test=1000,
)

# 6) 网络
net = dde.maps.FNN(
    layer_sizes=[1, 64, 64, 64, 1],
    activation="tanh",
    initializer="Glorot normal",
)

model = dde.Model(data, net)

# 7) 训练：Adam -> L-BFGS
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=6000, display_every=1000)

model.compile("L-BFGS")
losshistory, train_state = model.train()

# 8) 预测与评估
X = np.linspace(0, L, 400)[:, None]
u_pred  = model.predict(X)
u_exact = exact(X)
l2_rel = np.linalg.norm(u_pred - u_exact, 2) / np.linalg.norm(u_exact, 2)
print("Relative L2 error:", l2_rel)

# 可视化
plt.figure(figsize=(7,4))
plt.plot(X, u_exact, label="Exact: cos(x)", linestyle="--")
plt.plot(X, u_pred,  label="PINN prediction")
plt.xlabel("x"); plt.ylabel("u(x)")
plt.title("2nd-order ODE: u'' + u = 0 with u(0)=1, u'(0)=0")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()
```

### 说明

* 这里把两个条件 $u(0)=1$ 与 $u'(0)=0$ 都放在左端点 $x=0$（相当于初值问题）。
* `on_left` 用于只在左端点施加 `DirichletBC` 与 `NeumannBC`；右端点不会被这些约束作用。
* 训练目标依旧是让 **PDE 残差** 与 **边界（初值）条件** 同时满足；无需真解标签。
* 若想改成边值问题（例如 $u(0)=1,\,u(L)=\cos L$），把第二个条件改为另一个只作用在右端点的 `DirichletBC` 即可：写一个 `on_right` 判定 $x\approx L$。
