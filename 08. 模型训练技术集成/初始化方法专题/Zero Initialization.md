# Zero Initialization（零初始化）



## 📖 定义

**Zero Initialization** 就是把神经网络的 **权重和偏置全部初始化为 0**。

它是最朴素的初始化方式，但在深度学习中几乎从不使用（至少不用在权重上），因为它会导致 **神经元对称性问题**（symmetry problem）：

* 不同神经元在训练开始时输出完全相同；
* 反向传播时它们的梯度也完全相同；
* 导致它们无法学习到不同的特征，训练失效。

不过，**偏置初始化为 0** 是常见做法，因为偏置不会造成对称性问题。



## 📖 数学描述

假设某层的参数为权重矩阵 $W$ 和偏置向量 $b$，那么零初始化定义为：

$$
W_{ij} = 0, \quad b_i = 0
$$

对于任意输入 $x$，前向传播结果为：

$$
y = f(Wx + b) = f(0) = f(\mathbf{0})
$$

如果 $f$ 是 ReLU / Sigmoid / Tanh 等，所有神经元输出相同，学习无法展开。



## 📖 最简单的代码例子

### PyTorch 例子

```python
import torch
import torch.nn as nn

# 定义一个线性层 (输入=3, 输出=2)
linear = nn.Linear(3, 2)

# 使用 Zero Initialization
nn.init.zeros_(linear.weight)
nn.init.zeros_(linear.bias)

print("权重:\n", linear.weight)
print("偏置:\n", linear.bias)
```

### NumPy 例子

```python
import numpy as np

# 输入维度=3, 输出维度=2
weights = np.zeros((2, 3))
bias = np.zeros(2)

# 输入数据
x = np.random.randn(4, 3)

# 前向传播
y = x.dot(weights.T) + bias
print("输出:\n", y)
```



## 📖 小结

* **权重零初始化**：不可取（会导致所有神经元学习相同的东西）。
* **偏置零初始化**：常见且合理。
* 现代网络通常采用 Xavier、He、LSUV 等初始化方法来解决梯度消失/爆炸问题。


