# P-Tuning v2微调

## 📖 1. P-Tuning v2 的定义

**P-Tuning v2** 是清华大学提出的 **参数高效微调（PEFT）方法**，是对 P-Tuning 的改进版本。

* **核心思想**：在模型 **每一层 Transformer 的输入** 都插入一组 **可学习的连续提示向量（prefix vectors）**。
* **区别于 Prompt Tuning**：Prompt Tuning 只在输入层加 soft prompt，而 P-Tuning v2 在 **多层前缀注入（prefix-tuning-like）**，增强表达能力。
* **特点**：

  * 与全参数微调效果接近。
  * 不依赖复杂的 LSTM/MHA 结构（比 P-Tuning v1 简洁）。
  * 对各种 PLM（GPT、BERT、T5）均适用。



## 📖 2. 数学描述

设：

* 预训练 Transformer 模型为 $f_\theta$，参数 $\theta$ 冻结。
* 输入序列 embedding：

$$
X = (x_1, x_2, \dots, x_n), \quad x_i \in \mathbb{R}^d
$$

在每一层 $l \in \{1, \dots, L\}$，引入 **prefix 向量**：

$$
P^l = (p^l_1, p^l_2, \dots, p^l_m), \quad p^l_j \in \mathbb{R}^d
$$

在注意力计算中，将 prefix 拼接到 query/key/value 的输入：

$$
\text{SelfAttn}(X^l) = \text{Softmax}\left( \frac{[X^l W_Q; P^l_Q][X^l W_K; P^l_K]^T}{\sqrt{d_k}} \right) [X^l W_V; P^l_V]
$$

其中：

* $W_Q, W_K, W_V$ 是冻结的模型权重。
* $P^l_Q, P^l_K, P^l_V$ 是 prefix 向量投影后的结果。

训练目标函数：

$$
\mathcal{L}(\{P^l\}) = - \sum_{(X, y)} \log p(y \mid X, \theta, \{P^l\})
$$

即：冻结模型参数 $\theta$，只训练 prefix 参数 $\{P^l\}$。



## 📖 3. 简单代码演示

下面是一个基于 Hugging Face 的 **P-Tuning v2 简化实现**（只展示核心逻辑）：

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class PrefixEncoder(nn.Module):
    """用 MLP 生成 prefix embeddings"""
    def __init__(self, prefix_length, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(prefix_length, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, batch_size):
        prefix_tokens = torch.arange(self.embedding.num_embeddings).to(self.embedding.weight.device)
        prefix_embeds = self.embedding(prefix_tokens)  # [m, d]
        prefix_embeds = self.mlp(prefix_embeds)
        return prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, m, d]

class PTuningV2Model(nn.Module):
    def __init__(self, model_name="bert-base-uncased", prefix_length=10, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # 冻结预训练参数

        self.prefix_encoder = PrefixEncoder(prefix_length, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.prefix_length = prefix_length

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)

        # 原始词嵌入
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)

        # prefix embeddings
        prefix_embeds = self.prefix_encoder(batch_size)

        # 拼接 prefix + 原始输入
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        # 扩展 attention mask
        prefix_mask = torch.ones(batch_size, self.prefix_length).to(attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 输入 BERT
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]

        # 分类
        logits = self.classifier(cls_output)
        return logits

# 示例
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("P-Tuning v2 is powerful!", return_tensors="pt")

model = PTuningV2Model()
logits = model(**inputs)
print(logits)
```



## 📖 总结

* **定义**：P-Tuning v2 在 Transformer **各层** 注入 prefix 向量，只训练 prefix 参数，冻结模型。
* **数学形式**：在每层自注意力加入 prefix Q/K/V，优化 $\{P^l\}$。
* **代码实现**：用一个 `PrefixEncoder` 生成 prefix embedding，拼接到输入 embedding 和注意力计算中。


