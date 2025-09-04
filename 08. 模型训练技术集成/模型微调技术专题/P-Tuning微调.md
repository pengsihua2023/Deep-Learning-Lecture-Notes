# P-Tuning微调

<img width="105" height="140" alt="image" src="https://github.com/user-attachments/assets/7baa59a8-490f-4c58-b11b-757bbecfcb69" />  
  
P-Tuning 最早是由 复旦大学邱锡鹏团队（FudanNLP, Qiu Xipeng 等） 在 2021 年提出的。  

## 📖 1. P-Tuning 的定义

**P-Tuning**（Prompt Tuning 的一种变体）是一种 **参数高效微调方法**，它通过在输入序列前插入 **可学习的连续提示向量**（continuous prompts），来引导预训练语言模型完成下游任务。

* 原始模型参数保持冻结。
* 只训练提示向量（通常是几百到几千个参数），从而大幅减少微调开销。

与传统 “离散 prompt” 不同，P-Tuning 的提示是 **连续可微的向量**，可以通过梯度下降直接优化。



## 📖 2. 数学描述

假设：

* 预训练语言模型为 $f_\theta$，参数 $\theta$ 冻结。
* 原始输入序列为：

  $$
  X = (x_1, x_2, \dots, x_n), \quad x_i \in \mathbb{R}^d
  $$

  其中 $d$ 是词向量维度。

我们定义 **可学习的提示向量**：

$$
P = (p_1, p_2, \dots, p_m), \quad p_j \in \mathbb{R}^d
$$

新的输入序列为：

$$
X' = (p_1, p_2, \dots, p_m, x_1, x_2, \dots, x_n)
$$

训练时，我们只优化 $P$，目标函数（例如分类任务的交叉熵）为：

$$
\mathcal{L}(P) = - \sum_{(X, y)} \log p(y \mid X'; \theta, P)
$$

其中：

* $\theta$ 固定（模型参数冻结）。
* 仅通过反向传播更新 $P$。



## 📖 3. 简单代码示例

下面给出一个 Hugging Face `transformers` 上的 **P-Tuning 简化实现**（文本分类为例）：

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class PTuningModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", prompt_length=10, num_labels=2):
        super().__init__()
        # 加载预训练 BERT
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # 冻结预训练参数
        
        # 定义可学习的 prompt 向量
        self.prompt_length = prompt_length
        self.hidden_size = self.bert.config.hidden_size
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, self.hidden_size))
        
        # 分类头
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # 原始输入 embedding
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)

        # 扩展 batch 维度的 prompt
        batch_size = input_ids.size(0)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # 拼接 prompt + 原始 embedding
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)

        # 注意力 mask 也要扩展
        prompt_mask = torch.ones(batch_size, self.prompt_length).to(attention_mask.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # 送入 BERT
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        # 分类
        logits = self.classifier(pooled_output)
        return logits

# 测试
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("P-Tuning is amazing!", return_tensors="pt", padding=True, truncation=True)
model = PTuningModel()

logits = model(**inputs)
print(logits)
```


## 📖 总结

* **定义**：在输入序列前插入连续可学习的提示向量，冻结预训练模型，只优化提示参数。
* **数学形式**：$\mathcal{L}(P) = - \sum \log p(y \mid [P; X]; \theta)$。
* **代码**：通过 `nn.Parameter` 定义提示 embedding，并拼接到输入 embedding 前。


