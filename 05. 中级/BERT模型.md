## BERT模型
### BERT 网络介绍、原理和具体问题的实现

#### **介绍**
BERT（Bidirectional Encoder Representations from Transformers）是 Google 在 2018 年提出的预训练语言模型，广泛应用于自然语言处理（NLP）任务，如文本分类、问答、命名实体识别等。BERT 的核心在于使用 **双向 Transformer Encoder**，通过大规模无监督预训练捕获深层语义信息，然后微调以适配特定任务。

#### **原理**
1. **核心思想**：
   - BERT 使用 Transformer 的 Encoder 部分，通过双向自注意力机制（Multi-Head Self-Attention）建模单词的上下文。
   - **预训练**：
     - **Masked Language Model (MLM)**：随机掩盖 15% 的输入词，预测这些词，学习语义表示。
     - **Next Sentence Prediction (NSP)**：预测两个句子是否连续，学习句子间关系。
   - **微调**：替换最后一层（如全连接层），在特定任务数据集上微调模型。
   - 输入包括 token、段落和位置嵌入，输出为每个 token 的上下文表示。

2. **网络结构**：
   - **输入**：
     - 文本通过分词器（Tokenizer）转换为 token IDs，添加 `[CLS]`（分类 token）和 `[SEP]`（分隔 token）。
     - 嵌入层：Token Embedding（词向量） + Segment Embedding（区分句子） + Position Embedding（位置信息）。
   - **Transformer Encoder**：多层（BERT-base 为 12 层），包含自注意力机制和前馈网络。
   - **输出**：`[CLS]` token 的表示用于分类任务；每个 token 的表示用于序列任务（如 NER）。

3. **公式**：
   - 输入嵌入：
     \[
     \text{Input} = \text{Token Embedding} + \text{Segment Embedding} + \text{Position Embedding}
     \]
   - 自注意力：
     \[
     \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
     \]
     其中 \( Q, K, V \) 是查询、键、值向量，\( d_k \) 是维度。
   - 输出：最后一层 `[CLS]` 表示用于分类，或 token 表示用于序列标注。

4. **优点**：
   - 双向上下文建模，捕获深层语义。
   - 预训练 + 微调，适配多种 NLP 任务。
   - 预训练模型开箱即用，性能强大。

5. **缺点**：
   - 计算成本高，参数量大（BERT-base：110M 参数）。
   - 微调需要小心调参，避免过拟合。

6. **适用场景**：
   - 文本分类、问答、NER、翻译等。

---

#### **具体问题实现：文本分类 + 数据集加载 + 注意力可视化**
以下是一个完整的 PyTorch 代码示例，使用 Hugging Face 的 `transformers` 库实现 BERT 模型，解决**文本分类任务**（以情感分析为例，使用 IMDb 数据集），包括：
- 数据集加载（IMDb 数据集，简化版本）。
- BERT 模型微调。
- 注意力权重可视化（显示 `[CLS]` token 的注意力分布）。

##### **代码示例**
```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# 1. 自定义数据集
class IMDbDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.texts = dataset['text']
        self.labels = dataset['label']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. 加载数据集和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = load_dataset('imdb', split='train[:1000]')  # 使用 IMDb 数据集前 1000 条
train_dataset = IMDbDataset(dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 3. 加载 BERT 模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 二分类：正/负情感
model.train()

# 4. 训练（微调）模型
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.4f}")

# 5. 测试和注意力可视化
model.eval()
test_text = "This movie is fantastic!"
inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 获取注意力权重
model_with_attentions = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=True)
model_with_attentions.load_state_dict(model.state_dict())
model_with_attentions.to(device)
model_with_attentions.eval()

with torch.no_grad():
    outputs = model_with_attentions(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    attentions = outputs.attentions  # 注意力权重 (num_layers, batch_size, num_heads, seq_len, seq_len)

# 预测结果
labels = ['Negative', 'Positive']
pred_label = labels[torch.argmax(probs, dim=1).item()]
print(f"Text: {test_text}")
print(f"Predicted sentiment: {pred_label}")
print(f"Probabilities: {probs.tolist()}")

# 6. 可视化注意力权重（最后一层，第 1 个注意力头）
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
attn = attentions[-1][0, 0].detach().cpu().numpy()  # 最后一层，第 1 头
plt.figure(figsize=(10, 8))
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
plt.title('Attention Weights (Last Layer, Head 1)')
plt.xlabel('Tokens')
plt.ylabel('Tokens')
plt.show()
```

##### **代码说明**
- **数据集加载**：
  - 使用 `datasets` 库加载 IMDb 数据集（情感分析，二分类：正/负），取前 1000 条数据以简化。
  - 自定义 `IMDbDataset` 类，将文本编码为 `input_ids` 和 `attention_mask`，并提供标签。
  - `DataLoader` 批量加载数据，`batch_size=8`。
- **模型**：
  - `BertForSequenceClassification`：预训练 BERT 模型，附加分类头（全连接层），`num_labels=2` 表示正/负分类。
  - 使用 `bert-base-uncased`（12 层，768 维，110M 参数）。
- **训练**：
  - 使用 Adam 优化器，学习率 `2e-5`（BERT 微调常用值）。
  - 训练 3 个 epoch，计算交叉熵损失。
- **测试**：
  - 对示例文本 “This movie is fantastic!” 进行预测，输出情感和概率。
- **注意力可视化**：
  - 加载带 `output_attentions=True` 的模型，获取注意力权重。
  - 绘制最后一层第一个注意力头的权重热图，显示 token 间的注意力分布。
- **输出**：
  - 预测情感（正/负）。
  - 注意力热图，显示 `[CLS]` 和其他 token 的注意力权重。

#### **注意事项**
1. **环境**：
   - 安装依赖：`pip install transformers datasets torch matplotlib seaborn`.
   - GPU 加速：将模型和数据移到 GPU（`model.to(device)`）。
2. **数据集**：
   - IMDb 数据集需下载（`datasets` 库会自动处理）。
   - 可替换为其他数据集（如 SST-2、GLUE）。
3. **微调**：
   - 实际应用中，建议用完整数据集（如 IMDb 的 25,000 条训练数据）。
   - 可冻结部分层以加速：`model.bert.encoder.layer[:8].requires_grad = False`。
4. **注意力可视化**：
   - 热图显示 token 间的注意力权重，`[CLS]` token 通常聚合全局信息。
   - 可扩展到多头或多层注意力分析。
5. **计算资源**：
   - BERT 计算密集，建议 GPU 运行。
   - 批大小和序列长度（`max_length`）需根据硬件调整。

#### **扩展**
1. **多任务**：
   - 问答：使用 `BertForQuestionAnswering`，输入问题和上下文，输出答案跨度。
   - NER：使用 `BertForTokenClassification`，为每个 token 预测标签。
2. **更复杂可视化**：
   - 绘制多头注意力：循环 `attentions[-1][0, i]`（i 为注意力头）。
   - 分析特定 token 的注意力：提取 `attn[:, 0, :]`（`[CLS]` 的注意力）。
3. **反问题**：
   - 结合观测数据估计 BERT 参数（如注意力权重）。

#### **总结**
BERT 通过双向 Transformer 建模语义，预训练后微调适配任务。上述代码展示了在 IMDb 数据集上的文本分类实现，包括数据加载、微调和注意力可视化。如果需要更复杂的任务（如问答、NER）或特定可视化（如多头注意力）。
