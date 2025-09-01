
# IA³ 微调（Infused Adapter by Inhibiting and Amplifying Inner Activations）

## 1. 定义

**IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)** 是一种 **参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）** 方法，由 Liu et al. (2022) 提出。

它的核心思想是：

* 不更新原始预训练模型参数 $\theta$。
* 在 **Transformer 的注意力和前馈层** 中引入 **可训练的标量向量**，对激活值进行 **缩放（inhibit/amplify）**。
* 这样仅需训练少量参数，却能高效适配下游任务。

👉 简单理解：IA³ 给每一层的注意力和值投影增加 **逐通道的缩放因子**，像旋钮一样调节信号强度。



## 2. 数学描述

### 2.1 Transformer 中的注意力计算

标准 **Scaled Dot-Product Attention**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中：

* $Q = XW_Q$
* $K = XW_K$
* $V = XW_V$

### 2.2 IA³ 修改后的注意力

IA³ 在注意力和前馈层中插入 **逐通道缩放向量** $l_k, l_v, l_{ff}$：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q (K \odot l_k)^T}{\sqrt{d_k}}\right)(V \odot l_v)
$$

$$
\text{FFN}(X) = (X W_1 \odot l_{ff}) W_2
$$

其中：

* $\odot$ 表示逐元素相乘（广播）。
* $l_k, l_v, l_{ff}$ 是 **可训练参数向量**，维度分别与 $K$、$V$、FFN 中间层相同。

### 2.3 损失函数

只训练这些缩放参数：

$$
\mathcal{L} = \mathcal{L}_{task}(f(x; \theta, l_k, l_v, l_{ff}))
$$

$\theta$ 固定（冻结），只更新 $l_k, l_v, l_{ff}$。



## 3. 简单代码示例（PyTorch）

这里用 PyTorch 写一个 **简化版 IA³ 注意力层**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IA3Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(IA3Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        # 原始投影层（冻结）
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

        for p in self.parameters():
            p.requires_grad = False

        # IA³ 可训练缩放参数
        self.l_k = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.l_v = nn.Parameter(torch.ones(1, 1, embed_dim))

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x) * self.l_k  # 缩放 Keys
        V = self.W_V(x) * self.l_v  # 缩放 Values

        # 分多头
        Q = Q.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1,2)

        # 注意力
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1,2).reshape(x.size(0), -1, self.embed_dim)
        return out

# 测试
x = torch.rand(2, 5, 16)  # batch=2, seq_len=5, embed_dim=16
attn = IA3Attention(embed_dim=16, num_heads=2)
out = attn(x)
print("Output shape:", out.shape)  # (2, 5, 16)
```

在真实 Transformer 中，还会在 FFN（前馈层）中加上类似的缩放参数 $l_{ff}$。



## 4. 总结

* **定义**：IA³ 是一种参数高效微调方法，通过在注意力和前馈层中引入缩放向量调节激活值。
* **公式**：

  $$
  \text{Attn}(Q,K,V) = \text{softmax}\Big(\frac{Q (K \odot l_k)^T}{\sqrt{d_k}}\Big)(V \odot l_v)
  $$

  $$
  \text{FFN}(X) = (X W_1 \odot l_{ff}) W_2
  $$
* **特点**：

  * 只训练少量参数（缩放向量）。
  * 不修改预训练权重，显存/存储开销小。
  * 效果接近全量微调，尤其适合大模型适配。
* **代码**：只需在注意力和 FFN 中加入可训练缩放向量。



