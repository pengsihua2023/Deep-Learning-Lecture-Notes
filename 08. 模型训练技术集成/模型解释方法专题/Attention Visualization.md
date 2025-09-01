# Attention Visualization

## 1. 定义

**Attention Visualization** 是一种 **模型解释方法**，它通过可视化深度学习模型中的 **注意力权重（Attention Weights）**，帮助我们理解模型在预测时“关注了哪些输入部分”。

* 在 **NLP（自然语言处理）** 中：展示词与词之间的注意关系，例如机器翻译模型在翻译某个词时主要依赖哪些上下文词。
* 在 **CV（计算机视觉）** 中：展示图像区域的注意力分布，例如视觉 Transformer 在识别“猫”时，主要关注猫的脸部区域。


## 2. 数学描述

注意力权重的计算公式来自 **Scaled Dot-Product Attention**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：

* $Q$ = Queries
* $K$ = Keys
* $V$ = Values
* $\frac{QK^T}{\sqrt{d_k}}$ = 相似度矩阵（相关性分数）
* $\text{softmax}(\cdot)$ = 将分数转为注意力权重，范围在 $[0,1]$，且和为 1

👉 **可视化的核心**：把注意力权重矩阵

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

显示为 **热力图 / 箭头图 / 叠加图**，直观展示模型的关注模式。



## 3. 代码示例

### 3.1 在 NLP 中的 Attention 可视化（热力图）

```python
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

# 定义 Multi-Head Attention
mha = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)

# 输入 (batch=1, seq_len=5, d_model=16)
x = torch.rand(1, 5, 16)

# 自注意力 (Q=K=V=x)
out, attn_weights = mha(x, x, x)

print("Attention weights shape:", attn_weights.shape)  # [1, num_heads, seq_len, seq_len]

# 可视化第一头的注意力
sns.heatmap(attn_weights[0, 0].detach().numpy(), cmap="viridis", annot=True)
plt.xlabel("Key positions")
plt.ylabel("Query positions")
plt.title("Attention Heatmap (Head 1)")
plt.show()
```

👉 运行后得到一个 **热力图**，横轴表示 **Key（被关注的词）**，纵轴表示 **Query（当前词）**，颜色深浅表示注意力权重大小。



### 3.2 在 CV 中的 Attention 可视化（叠加图）

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 模拟注意力权重 (假设图像被分成 4x4 patch)
attn_weights = torch.rand(1, 1, 16, 16)  # [batch, head, patch_num, patch_num]
attn_map = attn_weights[0, 0].mean(0).reshape(4,4).detach().numpy()

# 原图 (随机模拟一张 64x64 图像)
img = np.random.rand(64,64)

# 可视化
plt.imshow(img, cmap="gray")
plt.imshow(attn_map, cmap="jet", alpha=0.5, extent=(0,64,64,0))  # 叠加注意力
plt.title("Attention Overlay on Image")
plt.colorbar()
plt.show()
```

👉 运行后得到一个图像，叠加了 **注意力热力图**，展示模型在图像分类时主要关注的区域。



## 4. 总结

* **Attention Visualization 定义**：通过展示注意力权重，解释模型在预测时“看哪里”。
* **数学描述**：基于

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$
* **代码示例**：

  * NLP → 用热力图展示词与词之间的注意关系。
  * CV → 用热力图叠加到图像上，展示模型的关注区域。



