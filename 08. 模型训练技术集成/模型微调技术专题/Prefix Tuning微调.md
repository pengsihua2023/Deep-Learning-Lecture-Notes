# Prefix Tuning 微调


## 📖 1. 定义

**Prefix Tuning** 是一种参数高效微调 (PEFT, Parameter-Efficient Fine-Tuning) 方法。
核心思想：

* 冻结预训练语言模型的所有参数；
* 在每一层 Transformer 的 **注意力机制输入前，插入一小段可训练的“前缀向量 (prefix tokens)”**；
* 训练时只更新这部分 prefix 参数，而不改动原始模型参数。

这样，Prefix Tuning 能极大减少需要训练的参数量，并且在不同任务间只需存储一小段前缀即可完成迁移。



## 📖 2. 数学公式

设 Transformer 的注意力层输入为 **query**、**key**、**value**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

在 Prefix Tuning 中：

* 对每一层 $l$，引入 prefix key 和 prefix value：

$$
K' = [P_k^l; K], \quad V' = [P_v^l; V]
$$

其中：

* $P_k^l, P_v^l$ 是可训练的前缀参数，通常通过一个小的 MLP 从 prefix embedding 生成；
* “;” 表示拼接操作。

因此，注意力机制变为：

$$
\text{Attention}(Q, K', V') = \text{softmax}\left(\frac{Q {K'}^T}{\sqrt{d_k}}\right) V'
$$

训练时只更新 $\{P_k^l, P_v^l\}$，冻结原始 $W_Q, W_K, W_V$。


## 📖 3. 最简代码例子

用 **PyTorch** 写一个最小化 Prefix Tuning 的示例（在一个 Transformer 层里加 prefix）：

```python
import torch
import torch.nn as nn

class PrefixTuningAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, prefix_len=5):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.prefix_len = prefix_len

        # 可训练的 prefix 参数 (prefix_len 个前缀 token)
        self.prefix_key = nn.Parameter(torch.randn(prefix_len, 1, d_model))
        self.prefix_value = nn.Parameter(torch.randn(prefix_len, 1, d_model))

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        seq_len, batch, d_model = x.shape

        # 将 prefix 扩展到 batch 维度
        pk = self.prefix_key.expand(-1, batch, -1)  # [prefix_len, batch, d_model]
        pv = self.prefix_value.expand(-1, batch, -1)

        # 生成原始 Q,K,V
        q = k = v = x

        # 拼接 prefix 到 K,V 上
        k = torch.cat([pk, k], dim=0)
        v = torch.cat([pv, v], dim=0)

        # 注意力计算
        out, _ = self.attn(q, k, v)
        return out

# ===== 测试 =====
x = torch.randn(10, 2, 128)  # [seq_len=10, batch=2, hidden=128]
layer = PrefixTuningAttention()
out = layer(x)
print("输出形状:", out.shape)
```

输出结果：

```
输出形状: torch.Size([10, 2, 128])
```

说明 Prefix Tuning 层正常运行。



## 📖 总结

* **Prefix Tuning**：在注意力层前引入 prefix key/value，不修改原始权重。
* **优点**：极大降低训练参数量，方便多任务共享预训练模型。
* **核心公式**：

$$
K' = [P_k; K], \quad V' = [P_v; V]
$$

---


## 📖 Prefix Tuning with Hugging Face PEFT
下面使用 Hugging Face PEFT 在 BERT 上进行 Prefix Tuning 微调 的完整示例。我们用一个小型文本分类任务（SST-2 情感分类）来演示。  
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, PrefixTuningConfig, TaskType

# 1. 加载预训练 BERT 模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 定义 Prefix Tuning 配置
peft_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_CLS,   # 任务类型：序列分类
    num_virtual_tokens=20,        # prefix 的长度（可调）
    encoder_hidden_size=768       # BERT 的 hidden dim
)

# 3. 把模型包装成 Prefix Tuning 模型
model = get_peft_model(model, peft_config)

# 4. 加载数据集 (SST-2 情感分类)
dataset = load_dataset("glue", "sst2")

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = encoded_dataset["train"].shuffle(seed=42).select(range(2000))  # 小样本演示
eval_dataset = encoded_dataset["validation"].shuffle(seed=42).select(range(500))

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./prefix_tuning_out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-4,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=50,
    save_strategy="no",
    fp16=True
)

# 6. Trainer API 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# 7. 测试推理
text = "The movie was fantastic!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=-1).item()
print(f"输入: {text} → 预测类别: {pred}")
```


## 📖 说明

1. **Prefix Tuning 配置**

   * `num_virtual_tokens=20` 表示在每层注意力前注入 20 个 prefix tokens。
   * 训练时只更新 prefix 参数，其余 BERT 权重保持冻结。

2. **数据集**

   * 使用 `GLUE/SST-2` 数据集（二分类：positive / negative）。
   * 这里只取一小部分数据训练，方便快速测试。

3. **训练**

   * 只更新 prefix 参数，显存和参数量大幅减少。
   * 训练几轮即可看到验证集准确率上升。

