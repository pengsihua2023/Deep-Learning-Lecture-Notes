* 这里的 **$d_k$** 表示 **Key 向量的维度**（通常也等于 Query 的维度）。

$d_k = \frac{d_{\text{model}}}{h}, \quad \text{缩放因子} = \sqrt{d_k}$

---

# Attention 机制的数学描述

Attention 机制的核心思想是：**在信息序列中为不同元素分配不同的权重，从而突出“重要”信息，抑制“无关”信息**。



## 1. 输入表示

给定输入向量序列：

$$
X = [x_1, x_2, \dots, x_n], \quad x_i \in \mathbb{R}^d
$$

通过线性变换映射为 **查询（Query）、键（Key）、值（Value）** 向量：

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

其中：

* $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ 为可学习参数；
* $Q, K, V \in \mathbb{R}^{n \times d_k}$。



## 2. 注意力打分函数

计算 **相似度得分（score）** 来衡量 Query 与 Key 的相关性：

$$
\text{score}(q_i, k_j) = \frac{q_i \cdot k_j^\top}{\sqrt{d_k}}
$$

其中 $\sqrt{d_k}$ 是缩放因子，防止数值过大。



## 3. 权重分布（Softmax）

将所有得分转化为概率分布：

$$
\alpha_{ij} = \frac{\exp(\text{score}(q_i, k_j))}{\sum_{l=1}^n \exp(\text{score}(q_i, k_l))}
$$

其中 $\alpha_{ij}$ 表示第 $i$ 个 Query 对第 $j$ 个 Key 的注意力权重。



## 4. 上下文向量（加权求和）

根据注意力权重对 Value 加权：

$$
z_i = \sum_{j=1}^n \alpha_{ij} v_j
$$

得到最终的上下文表示 $z_i$。



## 5. 矩阵形式（Scaled Dot-Product Attention）

将上面步骤写成紧凑的矩阵形式：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$



## 6. 多头注意力（Multi-Head Attention）

为了捕捉不同子空间的信息，使用 $h$ 个独立的注意力头：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

其中：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$



## 7. 总结

* **Query–Key**：决定关注什么；
* **Softmax 权重**：分配注意力；
* **Value**：承载信息；
* **最终输出**：输入的加权表示。

公式核心是：

$$
\boxed{  \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V  }
$$

---


## Attention-based Models (注意力机制模型, 扩展 Transformer)
<div align="center">
<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/a78ff1d6-3d30-43e6-b8e2-40acad211a7f" />
 </div> 
 
- 重要性：
Transformer 的注意力机制是现代深度学习的基石，衍生出如 BERT、GPT 等模型，驱动了 NLP 和多模态任务。  
高级班可以深入探讨注意力机制的变种（如多头注意力、自我注意力）。  
- 核心概念：
注意力机制让模型聚焦输入中最重要的部分（如句子中的关键词），通过“查询-键-值”机制计算权重。  
- 应用：聊天机器人（如 Grok）、机器翻译、文本摘要。

 
深度学习中的注意力机制（Attention Mechanism）是一种模仿人类视觉和认知系统的方法，它允许神经网络在处理输入数据时集中注意力于相关的部分。通过引入注意力机制，神经网络能够自动地学习并选择性地关注输入中的重要信息，提高模型的性能和泛化能力。  
上面这张图可以较好地去理解注意力机制，其展示了人类在看到一幅图像时如何高效分配有限注意力资源的，其中红色区域表明视觉系统更加关注的目标，从图中可以看出：人们会把注意力更多的投入到人的脸部
## 代码
添加注意力权重的可视化，使用一个热图来展示第一个样本的注意力权重矩阵，帮助直观理解Attention机制如何关注不同词之间的关系。代码仍基于IMDb数据集，并使用PyTorch实现简单的 Scaled Dot-Product Attention。由于您要求结果可视化，将生成一个热图，显示注意力权重。   

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super(SimpleAttention, self).__init__()
        self.dim = dim
    
    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

def yield_tokens(dataset):
    for example in dataset:
        yield example['text'].lower().split()

def plot_attention_weights(attention_weights, tokens, title="Attention Weights Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png")
    plt.close()
    print("Attention heatmap saved as 'attention_heatmap.png'")

def main():
    # 加载IMDb数据集
    dataset = load_dataset("imdb", split="train[:1000]")  # 使用前1000条评论
    batch_size = 32
    max_length = 20  # 缩短序列长度以便可视化
    embed_dim = 64

    # 构建词汇表
    vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=['<pad>', '<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # 创建词嵌入层
    embedding = nn.Embedding(len(vocab), embed_dim)

    # 将文本转换为索引
    def text_pipeline(text):
        tokens = text.lower().split()[:max_length]
        tokens += ['<pad>'] * (max_length - len(tokens))
        return [vocab[token] for token in tokens]

    input_ids = torch.tensor([text_pipeline(example['text']) for example in dataset], dtype=torch.long)
    
    # 获取词嵌入
    embedded = embedding(input_ids)  # [num_samples, max_length, embed_dim]
    
    # 初始化Attention模型
    model = SimpleAttention(embed_dim)
    
    # 分批处理
    outputs = []
    attention_weights_list = []
    
    for i in range(0, len(dataset), batch_size):
        batch = embedded[i:i+batch_size]
        output, attention_weights = model(batch, batch, batch)
        outputs.append(output)
        attention_weights_list.append(attention_weights)
    
    outputs = torch.cat(outputs, dim=0)
    attention_weights = torch.cat(attention_weights_list, dim=0)
    
    # 打印基本信息
    print("Dataset size:", len(dataset))
    print("Sample text:", dataset[0]['text'][:100] + "...")
    print("Output shape:", outputs.shape)
    print("Attention weights shape:", attention_weights.shape)
    
    # 可视化第一个样本的注意力权重
    first_attention = attention_weights[0].detach().numpy()  # [max_length, max_length]
    first_tokens = dataset[0]['text'].lower().split()[:max_length]
    first_tokens += ['<pad>'] * (max_length - len(first_tokens))
    plot_attention_weights(first_attention, first_tokens)

if __name__ == "__main__":
    main()
```

### 简要说明：
1. **数据集**：继续使用IMDb数据集的前1000条评论，序列长度缩短至20，以便热图更易读。
2. **可视化**：添加`plot_attention_weights`函数，使用`seaborn`绘制第一个样本的注意力权重热图，保存为`attention_heatmap.png`。
3. **热图内容**：
   - X轴和Y轴显示输入句子的词（或`<pad>`）。
   - 颜色深浅表示注意力权重大小（通过`viridis`颜色映射）。
   - 热图直观展示哪些词在Attention机制中对其他词的关注程度更高。
4. **依赖**：需安装`datasets`、`torchtext`、`matplotlib`和`seaborn`（`pip install datasets torchtext matplotlib seaborn`）。

### 运行结果：
- 程序将处理1000条IMDb评论，输出数据集信息、输出张量形状和注意力权重形状。
- 生成一个热图文件`attention_heatmap.png`，展示第一个评论的注意力权重矩阵。
- 热图中的每个单元格表示query词对key词的注意力权重，颜色越亮表示权重越大。

### 注意：
- 热图文件保存在运行目录下，可用图像查看器打开。
- 由于序列长度限制为20，热图显示前20个词的注意力关系，适合直观分析。



```
