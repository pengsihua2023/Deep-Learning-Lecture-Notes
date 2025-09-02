LSUV初始化 (Layer-Sequential Unit-Variance Initialization)



## 定义

**LSUV (Layer-Sequential Unit-Variance Initialization)** 是一种神经网络权重初始化方法，由 Mishkin 和 Matas 在 2015 年提出（论文 *All you need is a good init*）。

它的核心思想是：

1. 先用已有的初始化方法（通常是 **Orthogonal Initialization**）初始化权重；
2. 然后逐层地（Layer-Sequential）前向传播小批量数据；
3. 调整每一层输出的方差，使其接近 **单位方差（Unit Variance, 即方差=1）**；
4. 如果需要，还可以进一步调整均值使其接近 0。

这样可以确保在训练开始时，所有层的输出分布比较稳定，避免梯度消失或爆炸。

---

## 数学描述

假设某一层的输出为：

$$
y = W x + b
$$

其中 $x$ 是输入，$W$ 是权重矩阵。

LSUV 的步骤：

1. **初始权重**
   使用正交初始化：

   $$
   W_0 = \text{orthogonal}(shape)
   $$

2. **前向传播**
   用一小批数据（mini-batch）计算输出：

   $$
   y = f(Wx + b)
   $$

3. **调整方差**
   计算当前输出的方差：

   $$
   \sigma^2 = \text{Var}(y)
   $$

   更新权重：

   $$
   W \leftarrow \frac{W}{\sqrt{\sigma^2}}
   $$

4. **重复** 直到输出方差接近 1（容忍范围 $\epsilon$，如 $|\sigma^2 - 1| < 0.1$）。



## 最简单的代码例子

### PyTorch 伪实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 一个简单的两层网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

def lsuv_init(layer, data, tol=0.1, max_iter=10):
    # 正交初始化
    nn.init.orthogonal_(layer.weight)
    
    for _ in range(max_iter):
        # 前向传播
        with torch.no_grad():
            output = layer(data)
        
        var = output.var().item()
        if abs(var - 1.0) < tol:
            break
        
        # 调整权重
        layer.weight.data /= torch.sqrt(torch.tensor(var))

# 用随机数据演示
net = SimpleNet()
x = torch.randn(16, 100)  # batch_size=16

# 对第一层应用 LSUV
lsuv_init(net.fc1, x)
print("fc1 权重初始化后，输出方差:", net.fc1(x).var().item())
```



### NumPy 简化版（单层演示）

```python
import numpy as np

def lsuv_init(weights, x, tol=0.1, max_iter=10):
    # 正交初始化
    u, _, v = np.linalg.svd(weights, full_matrices=False)
    weights = u if u.shape == weights.shape else v
    
    for _ in range(max_iter):
        y = x.dot(weights.T)
        var = np.var(y)
        if abs(var - 1.0) < tol:
            break
        weights /= np.sqrt(var)
    return weights

# 输入数据 (batch=16, dim=100)
x = np.random.randn(16, 100)
weights = np.random.randn(50, 100)  # 输出50维

weights_lsuv = lsuv_init(weights, x)
print("LSUV 后输出方差:", np.var(x.dot(weights_lsuv.T)))
```



✅ 总结：

* **LSUV 初始化** = 正交初始化 + 逐层调整方差到 1；
* 数学上就是通过缩放权重来保证 $\text{Var}(y) \approx 1$；
* 适合深层网络，可以显著改善收敛速度。


