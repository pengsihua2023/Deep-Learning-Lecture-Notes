

# DeepLIFT 模型解释方法

## 1. 定义

**DeepLIFT (Deep Learning Important FeaTures)** 是一种 **模型解释方法**，通过比较神经网络在 **参考输入（reference input）** 和 **实际输入** 下的激活差异，来分配每个输入特征对输出的贡献值。

与基于梯度的方法不同，DeepLIFT 使用 **差异传播规则（rescale rules & reveal-cancel rules）**，避免了梯度消失或梯度爆炸问题，能更稳定地解释模型预测。

👉 简单理解：

* 参考点（baseline） = 模型在某个“中性输入”上的输出（例如全零向量、均值输入）。
* DeepLIFT 解释结果 = 实际输入相对于参考输入，对预测结果造成的 **变化贡献分解**。


## 2. 数学描述

### 2.1 基本思想

设：

* 输入特征：$x$
* 参考输入：$x'$
* 模型输出：$f(x)$，参考输出：$f(x')$
* 差异：

  $$
  \Delta x = x - x', \quad \Delta y = f(x) - f(x')
  $$

DeepLIFT 通过计算 **贡献分数 $C_{\Delta x_i \to \Delta y}$** 来分配每个输入特征的影响：

$$
\sum_i C_{\Delta x_i \to \Delta y} = \Delta y
$$

### 2.2 传播规则

在神经网络的逐层传播中，DeepLIFT 定义了几种规则：

* **Rescale Rule**：当输入和输出是单调关系时，将贡献按比例分配：

  $$
  C_{\Delta x_i \to \Delta y} = \frac{\Delta y}{\sum_j \Delta x_j} \cdot \Delta x_i
  $$

* **RevealCancel Rule**：用于捕捉输入之间的非线性相互作用，通过对比正负部分的独立贡献来分配权重。


## 3. 简单代码示例

我们使用 `captum`（PyTorch 的可解释性库）来演示 DeepLIFT 的用法。

```python
import torch
import torch.nn as nn
from captum.attr import DeepLift

# 1. 定义一个简单的模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNet()

# 2. 构造输入
inputs = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
baseline = torch.zeros_like(inputs)  # 参考输入 baseline=0

# 3. 使用 DeepLIFT 解释
deeplift = DeepLift(model)
attributions = deeplift.attribute(inputs, baselines=baseline)

print("Inputs:", inputs)
print("Attributions (feature contributions):", attributions)
```

输出结果类似于：

```
Inputs: tensor([[1., 2., 3.]], grad_fn=<...>)
Attributions: tensor([[ 0.12,  0.45,  0.33]], grad_fn=<...>)
```

这里的 `Attributions` 就是 DeepLIFT 分配给每个特征的贡献值，它们的和等于预测值相对于 baseline 的差。


## 4. 总结

* **定义**：DeepLIFT 通过比较输入与参考输入的差异来分配预测贡献。
* **数学公式**：

  $$
  f(x) - f(x') = \sum_i C_{\Delta x_i \to \Delta y}
  $$
* **优势**：解决了梯度方法在饱和区间失效的问题。
* **代码**：可通过 `captum` 直接调用 `DeepLift`。


