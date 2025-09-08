# FlashAttention
它不是一种新的注意力机制（不像 MHA/MQA/LHA），而是一种 **高效的注意力计算方法**，解决了 Transformer 在长序列上的计算和显存瓶颈问题。


## 1. 定义

**FlashAttention** 是一种 **显存优化、I/O 感知的注意力算法**，核心思想是：

* 普通注意力需要先算出完整的 $QK^T$ 矩阵（规模 $n \times n$），再 softmax，再乘 $V$，这需要 **$O(n^2)$ 显存**。
* FlashAttention **不显式存储整个注意力矩阵**，而是：

  * 把输入序列分块 (tiling)；
  * 在每个块内逐步计算 softmax（保持数值稳定性）；
  * 边算边归一化、边与 $V$ 相乘，结果直接写回，不保存中间大矩阵。

这样：

* 时间复杂度仍然是 $O(n^2)$，但是 **显存复杂度从 $O(n^2)$ 降到 $O(n)$**；
* 可以在 GPU 上处理更长的序列（几千到几万 token）。



## 2. 数学描述

标准注意力：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

FlashAttention 的改进点：

1. 不直接形成整个矩阵 $S = QK^\top / \sqrt{d_k}$，而是把它分块：

   $$
   S_{i,j} = \frac{Q_i K_j^\top}{\sqrt{d_k}}
   $$

   按 $i$ 和 $j$ 的分块迭代计算。

2. 逐块更新 softmax：

   $$
   \text{softmax}(S_{i,:}) = \frac{\exp(S_{i,j} - m_i)}{\sum_j \exp(S_{i,j} - m_i)}
   $$

   其中 $m_i$ 是该行的最大值（数值稳定性）。

3. 在计算 softmax 的同时，立即与 $V$ 相乘并累加：

   $$
   O_i = \sum_j \text{softmax}(S_{i,j}) V_j
   $$

因此 **最终不需要存储 $S$**，只保留归一化因子和当前块的输出。

---

## 3. 最简代码实现

这里是PyTorch 官方 FlashAttention API 使用示例。从 PyTorch 2.0 开始，`torch.nn.functional.scaled_dot_product_attention` 已经支持 **FlashAttention kernel**，在 GPU 上会自动调用优化实现。


### 1. 基本用法

```python
import torch
import torch.nn.functional as F

# 模拟输入
batch, n, d, h = 2, 128, 64, 8   # batch=2, 序列长度=128, d_model=64, 8个头
d_k = d // h

Q = torch.randn(batch, h, n, d_k, device="cuda")
K = torch.randn(batch, h, n, d_k, device="cuda")
V = torch.randn(batch, h, n, d_k, device="cuda")

# 调用 PyTorch 的 FlashAttention API
out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False)

print(out.shape)  # (batch, h, n, d_k)
```


### 2. 参数说明

* `Q, K, V`：输入张量，形状为 `(batch, heads, seq_len, d_k)`
* `attn_mask`：可选，支持 padding mask 或 causal mask
* `dropout_p`：注意力 dropout 概率
* `is_causal`：若为 `True`，则启用 **自回归因果掩码**（只看过去 token）
* 返回值：形状和 `Q` 一样 `(batch, heads, seq_len, d_k)`



### 3. 与 Multi-Head Attention 结合

可以直接在 `nn.Module` 里替换原来的注意力实现：

```python
class FlashMHA(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # 线性变换
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # FlashAttention
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0)

        # 拼接 heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_o(out)

# 测试
x = torch.randn(2, 128, 64, device="cuda")
mha = FlashMHA(64, 8).cuda()
y = mha(x)
print(y.shape)  # (2, 128, 64)
```


✅ 总结：

* 只要用 **`scaled_dot_product_attention`**，在 CUDA 上就会自动调用 **FlashAttention**（如果条件满足，比如序列长度足够）。
* 对用户来说几乎“零改动”，只要替换 `softmax(QK^T)V` 的那一行即可。


