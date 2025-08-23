下面给出一个**基于 DeepXDE 库的简单 ODE（常微分方程）求解代码例子**。我们以经典的一阶常微分方程

$$
\frac{dy}{dx} = -y, \quad y(0) = 1
$$

它的解析解为 $y(x) = e^{-x}$。

---

```python
import deepxde as dde
import numpy as np


# 定义微分方程 dy/dx + y = 0
def ode(x, y):
    dy_dx = dde.grad.jacobian(y, x, i=0, j=0)
    return dy_dx + y


# 定义初始条件 y(0) = 1
ic = dde.IC(
    geom=dde.geometry.Interval(0, 5),  # 定义区间 [0,5]
    func=lambda x: 1,                  # 初始值 y(0)=1
    on_boundary=lambda x, _: np.isclose(x[0], 0),
)

# 定义几何区域
geom = dde.geometry.Interval(0, 5)

# 构建数据集
data = dde.data.PDE(geom, ode, ic, num_domain=50, num_boundary=2)

# 构建神经网络
net = dde.nn.FNN([1] + [50] * 3 + [1], "tanh", "Glorot normal")

# PINN模型
model = dde.Model(data, net)

# 训练
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=5000)

# 测试预测
X = np.linspace(0, 5, 100)[:, None]
y_pred = model.predict(X)

# 真实解
y_true = np.exp(-X)

# 绘图
import matplotlib.pyplot as plt

plt.plot(X, y_true, "k-", label="True solution")
plt.plot(X, y_pred, "r--", label="DeepXDE prediction")
plt.legend()
plt.show()
```

---

### 代码说明：

1. **ode函数** 定义了微分方程 $\frac{dy}{dx} + y = 0$。
2. **IC** 定义了初始条件 $y(0) = 1$。
3. 使用 `dde.data.PDE` 构造训练数据。
4. 神经网络 `FNN` 用于近似解。
5. 训练完成后进行预测，并与解析解 $e^{-x}$ 对比。

这样就能用 **DeepXDE** 得到神经网络近似的 ODE 解。

---


