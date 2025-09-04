# Prompt Tuning微调
<img width="182" height="238" alt="image" src="https://github.com/user-attachments/assets/8454aee4-3658-4429-81d4-c07e0d728ea6" />

  Prompt Tuning 最早是 Google Research 的研究团队 在 2021 年提出的。第一作者：Brian Lester。  

## 📖 1. Prompt Tuning 的定义

**Prompt Tuning** 是一种 **参数高效微调（PEFT）** 方法。

* 它通过在输入序列前插入一段 **可学习的连续提示向量**（soft prompt），来适配下游任务。
* 与 **P-Tuning** 不同：

  * Prompt Tuning 只在 **embedding 层**加提示，不在模型中间层插入。
  * P-Tuning v1 曾使用 LSTM/MHA 来建模 prompt，更复杂。
* 训练时，**仅更新 soft prompt 参数**，而 **冻结整个预训练语言模型**。



## 📖 2. 数学描述

设：

* 预训练语言模型为 $f_\theta$，参数 $\theta$ 冻结。
* 原始输入为 token 序列：

  $$
  X = (x_1, x_2, \dots, x_n), \quad x_i \in \mathbb{R}^d
  $$

  其中 $d$ 是 embedding 维度。

定义 soft prompt 向量：

$$
P = (p_1, p_2, \dots, p_m), \quad p_j \in \mathbb{R}^d
$$

拼接后得到输入：

$$
X' = (p_1, p_2, \dots, p_m, x_1, x_2, \dots, x_n)
$$

任务目标函数（例如分类任务的交叉熵）：

$$
\mathcal{L}(P) = - \sum_{(X, y)} \log p(y \mid X'; \theta, P)
$$

即：冻结 $\theta$，仅优化 $P$。



## 📖 3. 简单代码演示

下面用 Hugging Face + PyTorch 给出一个简化的 **Prompt Tuning** 实现（文本分类为例）：

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class PromptTuningModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", prompt_length=20, num_labels=2):
        super().__init__()
        # 1. 预训练模型
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # 冻结 BERT 参数
        
        # 2. 定义 soft prompt (可学习参数)
        self.prompt_length = prompt_length
        self.hidden_size = self.bert.config.hidden_size
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, self.hidden_size))
        
        # 3. 分类头
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # 获取原始词嵌入
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)

        # 扩展 batch 维度
        batch_size = input_ids.size(0)
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # 拼接 soft prompt 和原始输入
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        # 调整 attention mask
        prompt_mask = torch.ones(batch_size, self.prompt_length).to(attention_mask.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # 输入到 BERT
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        # 分类
        logits = self.classifier(cls_output)
        return logits

# 使用示例
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Prompt tuning is efficient!", return_tensors="pt")

model = PromptTuningModel()
logits = model(**inputs)
print(logits)
```



## 📖 总结

* **Prompt Tuning 定义**：在输入序列前添加 **可学习的 soft prompt**，冻结模型，仅训练 prompt 参数。
* **数学形式**：$\mathcal{L}(P) = - \sum \log p(y \mid [P; X]; \theta)$。
* **代码实现**：用 `nn.Parameter` 定义 soft prompt，并拼接到 embedding 前端。



