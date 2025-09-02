# LoHA (Low-rank Hadamard Product Approximation) 微调方法

## 1. 定义

**LoHA** 是一种参数高效微调 (PEFT, Parameter-Efficient Fine-Tuning) 方法，它和 **LoRA** 类似，但在低秩分解时引入了 **Hadamard 逐元素乘积**，从而在保持低秩更新的同时增强表示能力。

* **LoRA**：对权重矩阵 $W \in \mathbb{R}^{d \times k}$，采用低秩分解

  $$
  \Delta W = B A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \; r \ll \min(d,k)
  $$

* **LoHA**：在低秩分解的基础上，引入 Hadamard 乘积（逐元素乘法），增强参数化表达能力：

  $$
  \Delta W = (B A) \odot (D C)
  $$

  其中：

  * $B, A$ 是第一组低秩分解参数；
  * $D, C$ 是第二组低秩分解参数；
  * $\odot$ 表示逐元素乘法 (Hadamard product)。

这样，LoHA 相比 LoRA 在相似参数规模下，能表示更复杂的变化。


## 2. 数学公式

设原始权重为 $W$，LoHA 训练时冻结 $W$，仅训练 $\Delta W$：

1. **有效权重**：

$$
W^{\text{eff}} = W + \Delta W
$$

2. **LoHA 低秩更新**：

$$
\Delta W = (B A) \odot (D C)
$$

其中：

* $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$
* $D \in \mathbb{R}^{d \times r}, C \in \mathbb{R}^{r \times k}$
* $\odot$ 为逐元素乘积。

训练时只更新 $(A, B, C, D)$，原始权重 $W$ 保持冻结。


## 3. 最简代码例子

用 **PyTorch** 实现一个极简 LoHA 线性层：

```python
import torch
import torch.nn as nn

class LoHALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        # 冻结的原始权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # LoHA 参数 (两组低秩分解)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.C = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.D = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        # LoHA 低秩更新
        delta_W = (self.B @ self.A) * (self.D @ self.C)  # Hadamard product
        W_eff = self.weight + delta_W
        return x @ W_eff.T  # 线性层计算

# ===== 测试 =====
x = torch.randn(2, 10)   # 输入
layer = LoHALinear(10, 5, rank=4)
out = layer(x)
print("输出形状:", out.shape)
```

运行结果：`输出形状: torch.Size([2, 5])`，说明 LoHA 线性层正常工作。



✅ 总结：

* **LoRA**：低秩加法更新 $\Delta W = BA$。
* **LoHA**：低秩 + Hadamard 更新 $\Delta W = (BA) \odot (DC)$，表示能力更强。


