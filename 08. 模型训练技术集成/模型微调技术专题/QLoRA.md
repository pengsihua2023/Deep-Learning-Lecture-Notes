# QLoRA (Quantized Low-Rank Adaptation) 微调方法


## 📖 1. 定义

**QLoRA (Quantized Low-Rank Adaptation)** 是 2023 年提出的一种高效微调大语言模型 (LLM) 的方法，它结合了 **权重量化 (Quantization)** 和 **LoRA (Low-Rank Adaptation)**：

* **权重量化**：把预训练模型的权重从 16/32-bit 浮点数压缩成 4-bit 表示（通常是 NF4 格式），显著减少显存占用。
* **LoRA**：在量化权重的基础上添加低秩矩阵（通常 rank=4\~64），仅训练这些小矩阵。
* **结果**：

  * 可以在单张 GPU 上微调百亿级别的模型；
  * 几乎不损失性能；
  * 保持高效和低显存开销。


## 📖 2. 数学公式

设：

* 原始权重矩阵： $W \in \mathbb{R}^{d \times k}$
* 量化后的权重： $\hat{W} = Q(W)$，其中 $Q(\cdot)$ 表示量化函数（如 4-bit NF4）
* LoRA 参数： $A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}, r \ll \min(d,k)$

**QLoRA 有效权重表示**：

$$
W^{\text{eff}} = \hat{W} + BA
$$

其中：

* $\hat{W}$ 是冻结的量化权重；
* $BA$ 是可训练的低秩增量。

训练时仅更新 $A,B$，保持 $\hat{W}$ 不变，从而减少内存和计算开销。


## 📖 3. 最简代码例子

用 Hugging Face **PEFT** 库进行 QLoRA 微调的最小示例（假设模型是一个小型 Transformers 模型）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. 加载一个预训练模型，并使用 4-bit 量化
model_name = "facebook/opt-125m"  # 小模型示例
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,                # 启用4-bit量化
    device_map="auto",                # 自动分配到GPU
    torch_dtype=torch.float16
)

# 2. 让模型支持 k-bit 训练（冻结量化权重）
model = prepare_model_for_kbit_training(model)

# 3. 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,                       # LoRA rank
    lora_alpha=16,             # 缩放因子
    target_modules=["q_proj","v_proj"],  # 指定在哪些层插入LoRA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. 给模型加上 LoRA 层 (得到 QLoRA)
model = get_peft_model(model, lora_config)

# 5. 示例输入
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("QLoRA is great because", return_tensors="pt").to("cuda")

# 6. 前向传播
outputs = model(**inputs)
print("Logits shape:", outputs.logits.shape)
```


## 📖 解释

1. **`load_in_4bit=True`** → 模型权重加载为 4-bit，显著节省显存。
2. **`prepare_model_for_kbit_training`** → 冻结量化权重，并启用必要的梯度支持。
3. **`LoraConfig`** → 配置 LoRA 参数，比如秩 `r=8`，目标模块是 Transformer 里的 `q_proj, v_proj`。
4. **`get_peft_model`** → 把 LoRA 插入到模型里，形成 QLoRA。
5. **训练时** → 只更新 LoRA 参数 (`A, B`)，量化的权重保持冻结。

下面是一个 **完整的 QLoRA 微调训练循环** 示例，我们用 Hugging Face 的 🤗 PEFT + Transformers，在一个小数据集（比如 `tiny_shakespeare` 或者一小段自定义文本）上跑几步，直观演示 **loss 下降**。

---

## 📖 QLoRA 微调完整示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. 加载模型和分词器 (小模型示例)
model_name = "facebook/opt-125m"  # 也可以换成 LLaMA/OPT/GPT-NeoX 等
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,           # 开启4-bit量化
    device_map="auto",           # 自动分配GPU
    torch_dtype=torch.float16
)

# 2. 让模型支持 k-bit 训练
model = prepare_model_for_kbit_training(model)

# 3. 配置 LoRA (QLoRA = 4-bit量化 + LoRA)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Transformer注意力层常用的LoRA插入点
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 构造一个小数据集 (tiny_shakespeare.txt 或者自定义文本)
with open("tiny_shakespeare.txt", "w") as f:
    f.write("To be, or not to be, that is the question.\n"
            "Whether 'tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune...")

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="tiny_shakespeare.txt",
    block_size=64
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./qlora_out",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=1,
    logging_steps=5,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs"
)

# 6. Hugging Face Trainer 训练
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()

# 7. 测试生成
inputs = tokenizer("To be, or not to", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```



## 📖 代码说明

1. **量化加载**：`load_in_4bit=True` → 使用 4-bit NF4 权重。
2. **LoRA 配置**：仅在注意力层的 `q_proj, v_proj` 上插入低秩适配器。
3. **数据集**：这里用一个简化版的 `tiny_shakespeare` 文本。
4. **训练循环**：使用 `Trainer` 封装，loss 会逐渐下降。
5. **推理**：训练完成后可以生成 Shakespeare 风格的文本。


