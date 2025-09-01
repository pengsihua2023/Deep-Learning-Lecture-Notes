# L-BFGS 优化器

## 简介

L-BFGS（Limited-memory Broyden–Fletcher–Goldfarb–Shanno）是一种拟牛顿（Quasi-Newton）优化方法。它通过近似海森矩阵（Hessian）的逆来加速梯度下降，在高维问题中仍然高效，但仅需有限的内存，因此适合大规模无约束优化问题。

---

## 数学描述

1. **优化目标**
   L-BFGS 用于求解无约束优化问题：

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

2. **迭代更新**
   在第 $k$ 步迭代时，更新规则为：

$$
x_{k+1} = x_k + \alpha_k p_k
$$

   其中， $\alpha_k$ 为步长（由线搜索确定）， $p_k$ 为搜索方向。

3. **搜索方向**
   搜索方向由近似的逆海森矩阵决定：

$$
p_k = - H_k \nabla f(x_k)
$$

4. **差分向量**
   定义变量和梯度的增量：

$$
s_k = x_{k+1} - x_k, \quad y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
$$

5. **BFGS 更新公式**
   经典 BFGS 的逆海森矩阵更新公式为：

$$
H_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) H_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}
$$

6. **L-BFGS 的有限记忆思想**
   完整 BFGS 需存储整个矩阵，代价过高。L-BFGS 仅保存最近 $m$ 步的 $(s_i, y_i)$ 向量对，并通过 **两次循环递归（two-loop recursion）** 高效计算搜索方向，而无需显式存储逆海森矩阵。

   **两次循环递归：**

   * 初始化： $q = \nabla f(x_k)$
   * 向后循环： $\alpha_i = \frac{s_i^T q}{y_i^T s_i}, \quad q \leftarrow q - \alpha_i y_i$
   * 设置初始矩阵： $H_0^k = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}} I$
   * 向前循环： $\beta = \frac{y_i^T r}{y_i^T s_i}, \quad r \leftarrow r + s_i(\alpha_i - \beta)$
   * 最终得到： $p_k = -r$

---

## 核心思想

1. **基本原理**

   * 牛顿法利用 Hessian 提供二阶信息，但在高维问题中存储与计算代价过大。
   * L-BFGS 通过有限历史信息近似 Hessian 的逆，从而实现“有限内存”的高效更新。

2. **算法核心**

   * 使用最近 $m$ 步的 $(s_i, y_i)$ 构造近似矩阵。
   * 用该矩阵计算下降方向，结合线搜索得到迭代点。

---

## 特点与优势

* **内存友好**：只存储有限的历史信息，适合高维优化问题。
* **收敛较快**：利用二阶信息，收敛速度接近牛顿法。
* **应用广泛**：逻辑回归、条件随机场（CRF）、词向量训练（如 word2vec）等场景均有应用。

---

## 应用示例

在 PyTorch 中使用 L-BFGS：

```python
import torch
from torch.optim import LBFGS

model = ...  # 定义模型
optimizer = LBFGS(model.parameters(), lr=0.1)

def closure():
    optimizer.zero_grad()
    loss = loss_fn(model(input), target)
    loss.backward()
    return loss

for i in range(20):
    optimizer.step(closure)
```

> 注意：需要定义 `closure` 函数，因为 L-BFGS 在每次迭代中会多次计算目标函数与梯度。

---

## 适用场景与局限

* **适用**：中小规模稠密数据的优化问题。
* **局限**：

  * 对非凸优化问题可能陷入局部最优。
  * 在大规模深度神经网络中使用有限，因计算和存储仍较昂贵；SGD 与 Adam 更常见。
