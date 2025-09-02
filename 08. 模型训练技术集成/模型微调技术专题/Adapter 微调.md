# Adapter 微调

## 1. 定义

**Adapter 微调** 是一种高效参数微调方法，常用于大规模预训练模型 (如 BERT、GPT 等)。
核心思想：

* **冻结原始预训练模型的参数**，避免大规模更新；
* 在每一层 Transformer **插入一个小型的瓶颈结构（Adapter 模块）**，仅训练这些 Adapter 参数；
* Adapter 一般是：**降维 → 激活 → 升维**，即：

  * 把输入维度 $d$ 降到一个小的瓶颈维度 $r$；
  * 经过非线性激活（如 ReLU / GELU）；
  * 再升回原维度 $d$，并与残差连接。

这样做能大幅减少需要训练的参数量，同时保持模型性能。


## 2. 数学公式

设：

* Transformer 隐状态向量：$h \in \mathbb{R}^d$
* Adapter 降维矩阵：$W_{down} \in \mathbb{R}^{r \times d}$
* Adapter 升维矩阵：$W_{up} \in \mathbb{R}^{d \times r}$
* 激活函数：$\sigma(\cdot)$

**Adapter 前向计算**：

$$
h' = h + W_{up} \, \sigma(W_{down} h)
$$

其中：

* $W_{down}$：将 $d$-维向量压缩到低维 $r$；
* $\sigma$：非线性映射（如 ReLU）；
* $W_{up}$：再映射回 $d$-维；
* 最终用残差连接加回 $h$。

训练时 **只更新 $W_{down}, W_{up}$**，其余模型参数保持冻结。


## 3. 最简代码例子

用 **PyTorch** 写一个最小化的 Adapter 层并插入到模型里：

```python
import torch
import torch.nn as nn

# ===== Adapter 模块 =====
class Adapter(nn.Module):
    def __init__(self, d_model, r=16):
        super().__init__()
        self.down = nn.Linear(d_model, r, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Linear(r, d_model, bias=False)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))  # 残差连接

# ===== 在 Transformer 层里用 Adapter =====
class ToyTransformerLayer(nn.Module):
    def __init__(self, d_model=128, adapter_r=16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.adapter = Adapter(d_model, r=adapter_r)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.ffn(x)
        x = self.adapter(x)  # 插入 Adapter
        return x

# ===== 简单测试 =====
x = torch.randn(10, 32, 128)  # [seq_len, batch_size, hidden_dim]
layer = ToyTransformerLayer()
out = layer(x)
print("输出维度:", out.shape)
```

## 解释

1. **Adapter**：两层全连接，形成降维–升维瓶颈。
2. **残差连接**：保证不破坏原有模型结构。
3. **训练**：实际应用中冻结所有预训练参数，只训练 Adapter 层的参数。
