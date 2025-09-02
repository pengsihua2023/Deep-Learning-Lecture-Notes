# Adapter模块（Adapter Modules）微调

## 1. 定义

**Adapter 微调** 是一种参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）方法。
核心思想：

* 冻结预训练模型的大部分参数；
* 在 Transformer 的每一层（通常在前馈层 FFN 或注意力层后）插入一个 **小型瓶颈结构（Adapter）**；
* 只训练 Adapter 模块的参数，而不更新原始模型的权重。

**优点**：训练参数量大幅减少（往往 <5%），同时保持模型性能。



## 2. 数学描述

设 Transformer 隐状态向量为 $h \in \mathbb{R}^d$，Adapter 的核心结构是一个 **降维—非线性—升维** 的瓶颈：

$$
h' = h + W_{up}\,\sigma(W_{down}\,h)
$$

其中：

* $W_{down} \in \mathbb{R}^{r \times d}$：降维映射，$r \ll d$；
* $W_{up} \in \mathbb{R}^{d \times r}$：升维映射；
* $\sigma(\cdot)$：非线性激活函数（如 ReLU, GELU）；
* 残差连接保证适配器不会破坏原始表示。

训练时只更新 $\{W_{down}, W_{up}\}$，冻结原始 Transformer 参数。



## 3. 最简代码例子

用 **PyTorch** 写一个极简 Adapter 模块，并在 Transformer 层中使用：

```python
import torch
import torch.nn as nn

# ===== Adapter 模块 =====
class Adapter(nn.Module):
    def __init__(self, d_model, r=16):
        super().__init__()
        self.down = nn.Linear(d_model, r, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Linear(r, d_model, bias=False)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))  # 残差连接

# ===== 在 Transformer 层里插入 Adapter =====
class ToyTransformerLayer(nn.Module):
    def __init__(self, d_model=128, adapter_r=16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.adapter = Adapter(d_model, r=adapter_r)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.ffn(x)
        x = self.adapter(x)  # 插入 Adapter
        return x

# ===== 测试 =====
x = torch.randn(10, 32, 128)  # [seq_len, batch, hidden_dim]
layer = ToyTransformerLayer()
out = layer(x)
print("输出形状:", out.shape)
```

运行结果：

```
输出形状: torch.Size([10, 32, 128])
```

说明 Adapter 模块正常工作。


## 总结：

* **定义**：在 Transformer 层插入小型瓶颈层，只训练 Adapter。
* **公式**：$h' = h + W_{up}\,\sigma(W_{down}\,h)$。
* **代码**：几行 PyTorch 就能实现最小化 Adapter 模块。

---

Hugging Face Transformers + PEFT Adapter 微调的完整示例，在 BERT 上做文本分类（比如 SST-2 情感分类）。


## 用 Hugging Face PEFT 进行 Adapter 微调

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, AdapterConfig, TaskType

# 1. 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 定义 Adapter 配置
adapter_config = AdapterConfig(
    task_type=TaskType.SEQ_CLS,   # 任务类型：序列分类
    r=16,                         # Adapter 的瓶颈维度 (降维)
    alpha=16,                     # 缩放系数
    dropout=0.1
)

# 3. 包装成 Adapter 模型
model = get_peft_model(model, adapter_config)
print(model)  # 你会看到模型中注入了 adapter 模块

# 4. 加载数据集 (GLUE/SST-2)
dataset = load_dataset("glue", "sst2")

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = encoded_dataset["train"].shuffle(seed=42).select(range(2000))  # 小样本演示
eval_dataset = encoded_dataset["validation"].shuffle(seed=42).select(range(500))

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./adapter_out",
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

# 6. Hugging Face Trainer 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# 7. 测试推理
text = "The movie was absolutely wonderful!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=-1).item()
print(f"输入: {text} → 预测类别: {pred}")
```

## 说明

1. **AdapterConfig**：定义瓶颈维度 `r=16`，意味着每个 Adapter 层只有少量可训练参数。
2. **冻结大模型**：PEFT 库会自动冻结 BERT 权重，只训练 Adapter 参数。
3. **数据集**：SST-2 情感分类（二分类：正面 / 负面）。
4. **效率**：相比全量微调，Adapter 微调的参数量 <5%，非常适合多任务迁移。

---

## Adapter / LoRA / Prefix Tuning 对比

| 方法                | 核心思想                                                 | 更新公式                                           | 训练参数量                   | 优点                            | 适用场景                      |
| ----------------- | ---------------------------------------------------- | ---------------------------------------------- | ----------------------- | ----------------------------- | ------------------------- |
| **Adapter**       | 在每层 Transformer 中插入一个小型瓶颈网络（降维→激活→升维），只训练 Adapter 参数 | $h' = h + W_{up}\,\sigma(W_{down} h)$          | $O(d r)$ （其中 $r \ll d$） | 稳定，易于多任务迁移，每个任务只需存储小型 Adapter | NLP 任务广泛应用，文本分类、序列标注      |
| **LoRA**          | 冻结原始权重，只在权重矩阵上添加低秩更新                                 | $\Delta W = B A$，有效权重 $W^{eff} = W + \Delta W$ | $O(d r + r k)$          | 参数量极小，推理高效，和量化结合（QLoRA）非常常见   | 大语言模型 (LLM) 微调，指令微调，聊天机器人 |
| **Prefix Tuning** | 在注意力机制的 key/value 前注入可训练的虚拟 prefix token             | $K' = [P_k; K], \; V' = [P_v; V]$              | 与 prefix 长度成正比，和模型参数无关  | 参数量固定，与模型大小解耦，适合大模型           | 生成类任务（NLG、对话、翻译），多任务快速切换  |


## 总结

* **Adapter** → 类似“外挂小网络”，稳定性好，适合分类 / 序列标注任务。
* **LoRA** → 在权重矩阵上做低秩近似，超高效，尤其在 LLM 上成为主流方法。
* **Prefix Tuning** → 在注意力前加虚拟 tokens，和模型参数规模解耦，适合大模型生成任务。



