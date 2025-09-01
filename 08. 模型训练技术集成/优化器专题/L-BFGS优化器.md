8L-BFGS（Limited-memory Broyden–Fletcher–Goldfarb–Shanno）优化器是一种拟牛顿法（Quasi-Newton method）的变体，主要用于解决大规模无约束优化问题。它常见于机器学习、深度学习和数值优化中。下面我帮你分几个方面讲清楚：

---
L-BFGS（Limited-memory Broyden–Fletcher–Goldfarb–Shanno）优化器是一种拟牛顿法优化算法，它通过近似海森矩阵的逆来加速梯度下降，适用于高维问题，但只需有限的内存。


---

## 数学描述

1. 优化问题

L-BFGS 用于求解无约束优化问题：

$\min_{x \in \mathbb{R}^n} f(x),$

2. 迭代更新

在迭代  步时，算法更新规则为：

$x_{k+1} = x_k + \alpha_k p_k,$

 是搜索方向，

 是步长（通常由线搜索确定）。


3. 搜索方向

搜索方向由近似的逆海森矩阵  与梯度决定：

$p_k = - H_k \nabla f(x_k).$ 

4. 差分向量

定义：

$s_k = x_{k+1} - x_k, \quad y_k = \nabla f(x_{k+1}) - \nabla f(x_k).$ 

5. BFGS 更新公式

经典 BFGS 公式对逆海森矩阵的更新为：

$H_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) H_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}.$ 

6. L-BFGS 的有限记忆思想

完整 BFGS 需要存储和更新  矩阵，代价过高。L-BFGS 通过只保存最近  步的  向量对，利用“两次循环递归”（two-loop recursion）高效计算 ：


---

两次循环递归

给定最近的  对，计算搜索方向：

1. 初始化：



$q = \nabla f(x_k).$ 

2. 第一次循环（向后）：



$\alpha_i = \frac{s_i^T q}{y_i^T s_i}, \quad q \leftarrow q - \alpha_i y_i.$ 

3. 设置初始矩阵：



$H_0^k = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}} I.$ 

4. 第二次循环（向前）：



$\beta = \frac{y_i^T r}{y_i^T s_i}, \quad r \leftarrow r + s_i (\alpha_i - \beta).$ 

最终：

$p_k = -r.$ 


---

总结

目标：近似二阶信息，加快收敛。

存储：只保存有限步的 ，适合大规模问题。

核心：通过两次循环递归避免显式存储逆海森矩阵。



---

要不要我帮你画一张 L-BFGS 更新流程的数学结构图（含两次循环递归），让逻辑更直观？


---

1. 基本原理

目标：最小化某个函数 ，通常是损失函数。

思路：牛顿法需要计算海森矩阵（Hessian，二阶导数矩阵），但在高维问题中存储和计算非常昂贵。

L-BFGS做法：不显式存储整个海森矩阵，而是通过有限的历史信息（梯度和变量的更新）来近似其逆矩阵，因此称为“limited-memory”。



---

2. 算法核心

在每一步迭代中，L-BFGS会利用最近的 m 步的变量更新  和梯度更新  来构建一个近似的逆海森矩阵。

用这个近似矩阵来计算下降方向，比单纯的梯度下降更高效、更快收敛。



---

3. 优势

内存友好：相比标准BFGS只需存储有限的历史信息，适合高维问题。

收敛快：相比普通梯度下降，利用了二阶信息，往往收敛速度接近牛顿法。

广泛应用：尤其在机器学习中的逻辑回归、条件随机场（CRF）、词向量训练（如word2vec）中应用广泛。



---

4. 应用举例

在深度学习框架 PyTorch 中，可以直接使用：

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
这里需要提供一个 closure函数，因为L-BFGS每一步迭代会多次计算目标函数和梯度。


---

5. 适用场景

适合 中小规模 的问题（特别是稠密数据），梯度需要多次重复计算。

对 非凸优化 问题可能卡在局部最优，不如随机梯度方法稳定。

大规模深度神经网络训练中 较少使用，因为计算和存储仍然不够高效（SGD、Adam更常用）。





