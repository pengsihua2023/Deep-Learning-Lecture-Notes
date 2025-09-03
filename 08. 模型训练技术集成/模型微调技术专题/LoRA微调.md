# LoRA 微调（Low-Rank Adaptation）

## 📖  1. 定义

**LoRA** 是一种 **参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）** 方法，由微软在 2021 年提出。

核心思想：

* **冻结预训练模型参数** $W$，不直接更新它们。
* 在每个大权重矩阵的更新项中，引入一个 **低秩分解参数** $\Delta W = BA$。
* 在微调时 **只训练低秩矩阵 $A, B$**，而原始权重保持不变。

👉 好处：

* 极大减少可训练参数量（因为秩 $r \ll d,k$ ）。
* 训练后可选择“合并” $\Delta W$ 到原模型，或以“插件”方式加载。



## 📖 2. 数学描述

设原始权重矩阵：

$$
W \in \mathbb{R}^{d \times k}
$$

在微调过程中，LoRA 将权重修改为：

$$
W' = W + \Delta W, \quad \Delta W = B A
$$

其中：

* $A \in \mathbb{R}^{r \times k}$ ， $B \in \mathbb{R}^{d \times r}$ ， $r \ll \min(d,k)$ 。
* 初始时 $BA = 0$ ，确保不会影响预训练模型。
* 只训练 $A, B$ ，冻结 $W$ 。

应用在 Transformer 中，LoRA 通常作用在 **注意力层的投影矩阵** $W_Q, W_V$。


## 📖 3. 简单代码示例（PyTorch 实现）

下面用 PyTorch 写一个 **线性层 + LoRA** 的实现：

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super(LoRALayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))  
        self.weight.requires_grad = False  # 冻结原始权重

        # LoRA 参数 (低秩分解)
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)

    def forward(self, x):
        W_eff = self.weight + self.B @ self.A  # 有效权重 = 冻结权重 + LoRA 增量
        return x @ W_eff.T

# 测试
x = torch.randn(2, 10)  # batch=2, 输入维度=10
layer = LoRALayer(in_dim=10, out_dim=6, rank=2)
y = layer(x)
print("Output shape:", y.shape)  # (2, 6)
```

在实际应用中，LoRA 通常插入到 Transformer 的 **Q (query projection)** 和 **V (value projection)** 矩阵中，用少量参数实现高效适配。


## 📖 4. 总结

* **定义**：LoRA 通过低秩分解，仅训练增量参数矩阵，而冻结原始权重。
* **公式**：

$$
W' = W + BA, \quad A \in \mathbb{R}^{r \times k}, \; B \in \mathbb{R}^{d \times r}, \; r \ll \min(d,k)
$$

* **特点**：

  * 显著减少可训练参数量。
  * 训练后可“合并”或“插件式”使用。
  * 与 Transformer 架构高度兼容。
* **代码**：PyTorch 可以通过 `LoRALayer` 类来实现。

---

# 📖 在 Transformer 注意力层中集成 LoRA

## 1. 思路

* 在 **Self-Attention** 里，通常有三个线性投影：

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$
  
* **LoRA** 只作用在 $W_Q, W_V$（也可以扩展到 $W_K, W_O$）。
* 替换方式：

$$
W_Q' = W_Q + B_Q A_Q, \quad W_V' = W_V + B_V A_V
$$

* 这样只需训练 **低秩矩阵 $A, B$**，冻结原始参数。



## 2. PyTorch 示例代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super(LoRALinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.weight.requires_grad = False  # 冻结原始权重

        # LoRA 参数
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)

    def forward(self, x):
        W_eff = self.weight + self.B @ self.A
        return x @ W_eff.T


class LoRAAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rank=4):
        super(LoRAAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 投影矩阵：Q、K、V（其中Q和V使用LoRA）
        self.W_Q = LoRALinear(embed_dim, embed_dim, rank)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = LoRALinear(embed_dim, embed_dim, rank)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out)


# 测试
x = torch.randn(2, 5, 16)  # batch=2, seq_len=5, embed_dim=16
attn = LoRAAttention(embed_dim=16, num_heads=2, rank=2)
y = attn(x)
print("Output shape:", y.shape)  # (2, 5, 16)
```

---

## 📖 3. 总结

* **Q、V 投影矩阵** 被替换成 **LoRA 版本**，只训练低秩矩阵 $A, B$。
* **参数量显著减少**：比如 $d=1024, k=1024, r=8$，LoRA 参数只有 $16k$，远小于全量参数 $1M+$。
* **兼容性强**：LoRA 可以无缝插入现有 Transformer 模型。

---


# 📖 Hugging Face Transformers + LoRA 示例

## 1. 安装依赖

```bash
pip install transformers datasets peft accelerate
```



## 2. 加载数据 & 模型

我们用 **Hugging Face Datasets** 加载 `sst2` 情感分类任务，并加载 `bert-base-uncased`。

```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 数据集
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)

# 模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```



## 3. 配置 LoRA

`peft` 库提供了简洁的接口。

```python
from peft import LoraConfig, get_peft_model

# LoRA 配置
config = LoraConfig(
    r=8,                      # 低秩分解维度
    lora_alpha=16,            # 缩放因子
    target_modules=["query", "value"],  # 作用在 Self-Attention 的 Q,V 矩阵
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

# 注入 LoRA
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

输出示例：

```
trainable params: 590,848 || all params: 109,483,778 || trainable%: 0.54
```

👉 只需训练 **不到 1% 的参数**。



## 4. 训练

我们用 Hugging Face 的 `Trainer` 来进行微调。

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-bert",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-4,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(5000)),  # 小样本演示
    eval_dataset=encoded_dataset["validation"].select(range(1000)),
    tokenizer=tokenizer
)

trainer.train()
```



## 5. 推理 & 保存 LoRA

```python
# 推理
text = "The movie was absolutely wonderful!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=-1).item()
print("Prediction:", "Positive" if pred == 1 else "Negative")

# 只保存 LoRA 参数
model.save_pretrained("./lora-bert")
```

---

## 📖 总结

* **LoRA 在 Hugging Face 中的实现**非常方便，只需用 `peft.LoraConfig` 注入即可。
* 只训练 **Q、V 矩阵**的低秩更新，大大减少参数量。
* 可用于 BERT、GPT-2、T5 等大模型的 **高效微调**。






