### PEFT 库简介

**PEFT**（Parameter-Efficient Fine-Tuning）是一个开源 Python 库，由 Hugging Face 开发，专注于参数高效的微调方法。它旨在降低对大型预训练模型（如 Transformer、BERT、GPT 等）进行微调的计算和存储成本，特别适合资源受限场景或需要快速适配多个下游任务的情况。PEFT 提供了多种参数高效微调技术的实现，包括 **LoRA**（Low-Rank Adaptation）、**Prefix Tuning**、**Prompt Tuning** 等。

以下是 PEFT 库的详细介绍，涵盖其核心功能、支持的微调方法、优势、应用场景以及与之前讨论的 Prefix Tuning 相关的代码示例。

---

### 核心功能
PEFT 库的主要目标是让用户以极低的参数成本微调大型预训练模型，同时保持与全参数微调（Full Fine-Tuning）相当的性能。其核心功能包括：
1. **参数高效微调**：仅更新少量新增参数（通常 <1% 的模型参数），冻结原始模型权重，减少内存和计算需求。
2. **模块化设计**：支持多种高效微调方法，用户可根据任务选择合适的策略。
3. **与 Hugging Face 生态无缝集成**：兼容 `transformers` 和 `datasets` 库，支持主流预训练模型（如 BERT、RoBERTa、T5、LLaMA 等）。
4. **跨任务支持**：适用于分类、生成、问答、序列标注等多种自然语言处理（NLP）任务。
5. **模型保存与加载**：高效存储仅包含微调参数的模型，显著减少存储需求。
6. **易用性**：提供简单 API，只需少量代码即可实现复杂微调逻辑。

---

### 支持的微调方法
PEFT 库支持以下主要参数高效微调方法：
1. **LoRA（Low-Rank Adaptation）**：
   - 通过在权重矩阵中引入低秩分解（Low-Rank Updates）来微调模型。
   - 只更新少量低秩矩阵参数，适合多种任务。
   - 优点：高效、性能接近全参数微调。
2. **Prefix Tuning**：
   - 在 Transformer 的注意力层添加可学习的前缀向量，影响键（Key）和值（Value）的计算。
   - 适合生成任务和序列分类任务（之前讨论的重点）。
3. **Prompt Tuning**：
   - 在输入层添加可学习的虚拟 token（提示），仅优化这些提示参数。
   - 适合小规模任务或数据受限场景。
4. **P-Tuning v2**：
   - Prompt Tuning 的改进版，增加了前缀参数的灵活性，适用于更复杂任务。
5. **Adapter 方法**：
   - 在每层 Transformer 中插入小型全连接模块（Adapter），仅优化这些模块。
   - 优点：模块化，可针对不同任务切换 Adapter。
6. **IA³（Infused Adapter by Inhibiting and Amplifying Inner Activations）**：
   - 通过缩放 Transformer 内部激活值来微调，参数效率极高。
7. **LoHA（Low-Rank Hadamard Adaptation）**和 **LoKr（Low-Rank Kronecker Adaptation）**：
   - LoRA 的变体，使用不同矩阵分解方式进一步优化效率。

---

### 优势
- **低资源需求**：只需微调少量参数（通常几十 KB 到几 MB），适合在普通 GPU 或 CPU 上运行。
- **快速适配**：不同任务可使用独立的可学习参数，切换任务无需重新训练整个模型。
- **存储高效**：保存的模型仅包含微调参数，大幅减少存储空间（相比全参数微调的 GB 级模型）。
- **性能可比**：在许多任务上（如情感分析、文本生成），性能接近甚至优于全参数微调。
- **开源与社区支持**：PEFT 是开源项目（Apache 2.0 许可证），与 Hugging Face 生态深度整合，社区活跃。

---

### 应用场景
- **NLP 任务**：情感分析、文本分类、机器翻译、问答、文本生成等。
- **多任务学习**：通过为每个任务训练独立的微调参数（LoRA 或 Adapter），实现高效任务切换。
- **边缘设备部署**：在资源受限设备上微调和部署大型模型。
- **领域适配**：将通用预训练模型快速适配到特定领域（如医疗、法律）。
- **研究与实验**：快速测试不同微调策略，探索参数高效方法的效果。

---

### 与 Prefix Tuning 的关系
你在之前的讨论中重点关注了 **Prefix Tuning**，这是 PEFT 库支持的一种方法。PEFT 通过 `PrefixTuningConfig` 和 `get_peft_model` 函数简化了 Prefix Tuning 的实现，自动将可学习前缀融入 Transformer 的注意力机制。以下是一个基于 IMDB 数据集的真实例子，展示如何使用 PEFT 实现 Prefix Tuning。

---

### 代码示例：使用 PEFT 进行 Prefix Tuning（IMDB 数据集）
以下代码基于之前讨论的 IMDB 数据集，展示如何使用 PEFT 库进行 Prefix Tuning 实现文本分类（情感分析）。

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from peft import PrefixTuningConfig, get_peft_model
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 1. 加载 IMDB 数据集
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

# 为加速演示，使用前 2000 条训练数据和 1000 条测试数据
train_texts, train_labels = train_texts[:2000], train_labels[:2000]
test_texts, test_labels = test_texts[:1000], test_labels[:1000]

# 2. 自定义 Dataset
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 3. 加载 Tokenizer 和 DataLoader
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 4. 配置 Prefix Tuning
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
prefix_config = PrefixTuningConfig(
    task_type="SEQ_CLS",  # 序列分类任务
    num_virtual_tokens=20,  # 前缀长度
    prefix_projection=True  # 启用投影层
)
model = get_peft_model(model, prefix_config)

# 打印可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.2f}% of total)")
print(f"Total parameters: {total_params}")

# 5. 设置优化器和调度器
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
num_epochs = 3
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 6. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 7. 训练函数
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 8. 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    preds, true_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=["Negative", "Positive"])
    avg_loss = total_loss / len(dataloader)
    return accuracy, report, avg_loss

# 9. 训练循环
for epoch in range(num_epochs):
    avg_train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
    test_accuracy, test_report, test_loss = evaluate(model, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
    print("Classification Report:\n", test_report)

# 10. 保存模型
model.save_pretrained("prefix_tuned_bert_imdb_peft")
tokenizer.save_pretrained("prefix_tuned_bert_imdb_peft")
print("Model and tokenizer saved!")

# 11. 推理示例
loaded_model = BertForSequenceClassification.from_pretrained("prefix_tuned_bert_imdb_peft")
loaded_tokenizer = BertTokenizer.from_pretrained("prefix_tuned_bert_imdb_peft")
loaded_model.to(device)
loaded_model.eval()

test_text = "This movie was absolutely fantastic and thrilling!"
encoding = loaded_tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
encoding = {k: v.to(device) for k, v in encoding.items()}

with torch.no_grad():
    outputs = loaded_model(**encoding)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    label_map = {0: "Negative", 1: "Positive"}
    print(f"Predicted sentiment: {label_map[predicted_class]}")
```

---

### 代码说明
1. **数据集**：
   - 使用 Hugging Face 的 `datasets` 库加载 IMDB 数据集。
   - 为加速演示，限制训练数据为 2000 条，测试数据为 1000 条（可移除限制使用全部 50,000 条）。
2. **PEFT 配置**：
   - 使用 `PrefixTuningConfig` 配置 Prefix Tuning，设置 `num_virtual_tokens=20` 和 `prefix_projection=True`。
   - 通过 `get_peft_model` 自动将前缀参数融入 BERT 模型。
3. **训练与评估**：
   - 训练 3 轮，使用 AdamW 优化器和线性学习率调度。
   - 评估指标包括准确率、分类报告和测试损失。
4. **推理**：
   - 展示如何加载保存的模型进行情感预测。
5. **依赖**：
   ```bash
   pip install torch transformers peft datasets scikit-learn tqdm
   ```

---

### PEFT 库的安装与使用
- **安装**：
  ```bash
  pip install peft
  ```
- **文档**：PEFT 官方文档（https://huggingface.co/docs/peft）提供详细配置说明和示例。
- **支持模型**：支持 Hugging Face `transformers` 中几乎所有 Transformer 模型。
- **版本要求**：建议使用最新版本（截至 2025 年 8 月，推荐 PEFT>=0.5.0，transformers>=4.30.0）。

---

### 扩展与进一步需求
如果你需要以下内容，请告诉我，我可以进一步定制：
1. **其他 PEFT 方法**：如 LoRA 或 Prompt Tuning 的实现。
2. **其他数据集**：如 Yelp、SST-2、Twitter 等。
3. **多任务支持**：为多个任务配置独立的 PEFT 参数。
4. **Snakemake 整合**：将 PEFT 训练流程融入 Snakemake 工作流（结合你之前的讨论）。
5. **数学推导**：深入讲解 Prefix Tuning 或 LoRA 的数学原理。
6. **部署**：将训练好的模型部署到 GitHub 或云平台。


