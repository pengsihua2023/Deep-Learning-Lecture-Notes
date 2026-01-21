# RAG技术

# 一、RAG 的基本原理（信息论视角）

传统 LLM：

$$
P(y \mid x)
$$

> 在给定问题 (x) 时，直接生成答案 (y)

RAG：

$$
P(y \mid x) = \sum_{d \in \mathcal{D}} P(y \mid x, d) \cdot P(d \mid x)
$$

含义是：

> **答案来自于所有可能文档的加权期望**

其中：

* (x)：用户问题
* (d)：检索到的文档
* (P(d|x))：检索模型（embedding + 向量检索）
* (P(y|x,d))：生成模型（LLM）

---

# 二、RAG 的数学结构（完整分解）

### 1. 向量空间映射

Embedding 本质是一个映射：

$$
f: \text{text} \rightarrow \mathbb{R}^n
$$

比如：

$$
v_q = f(x), \quad v_d = f(d)
$$

---

### 2. 相似度检索

最常用是余弦相似度：

$$
\text{sim}(q,d) = \frac{v_q \cdot v_d}{|v_q||v_d|}
$$

检索：

$$
d^* = \arg\max_d \text{sim}(q,d)
$$

---

### 3. 生成模型条件概率

最终模型是：


<img width="122" height="38" alt="image" src="https://github.com/user-attachments/assets/8f28643b-e0a2-427d-b415-91566291e1a7" />



也就是：

> 在 **文档作为额外上下文约束下** 的语言模型。

---

### 4. 信息论解释（最关键）

RAG 的作用其实是降低熵：

$$
H(Y|X, D) < H(Y|X)
$$

也就是：

> 给模型喂真实文档，相当于给答案空间加了强先验约束。

---

# 三、RAG 的概率图模型结构

```
   x (query)
      |
      v
  Retriever
      |
      v
   d1 d2 d3
      |
      v
  Generator
      |
      v
      y
```

这是一个典型的 **latent variable model**：

* d 是隐变量（不可直接观察）
* 用检索器近似后验分布

---

# 四、最小可运行 RAG 实现（50 行核心代码）

不依赖 LangChain / FAISS
纯粹数学结构的工程映射版

## 依赖

```bash
pip install sentence-transformers scikit-learn openai
```

---

## 代码

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

client = OpenAI()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===== 知识库 D =====
docs = [
    "RAG 是 Retrieval-Augmented Generation 的缩写。",
    "RAG 通过检索外部知识减少模型幻觉。",
    "RAG 常用于企业私有知识库问答。",
]

# 向量化 D
doc_vecs = embed_model.encode(docs)

# ===== P(d|x) 近似 =====
def retrieve(query, k=2):
    q_vec = embed_model.encode([query])
    sims = cosine_similarity(q_vec, doc_vecs)[0]
    idx = np.argsort(sims)[-k:][::-1]
    return [docs[i] for i in idx]

# ===== P(y|x,d) =====
def rag_answer(query):
    contexts = retrieve(query)
    context = "\n".join(contexts)

    prompt = f"""
只能基于以下资料回答：

{context}

问题：{query}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


print(rag_answer("RAG 有什么作用？"))
```

---

# 五、这个最小实现对应的数学映射

| 数学对象   | 工程实现               |                   |
| ------ | ------------------ | ----------------- |
| (f(x)) | embed_model.encode |                   |
| (P(d   | x))                | cosine_similarity |
| argmax | np.argsort         |                   |
| (P(y   | x,d))              | GPT forward       |
| Σ 积分   | top-k 近似           |                   |

---

# 六、为什么说 RAG 是“系统级而非模型级创新”

本质上：

> RAG 没有引入任何新的神经结构
> 它只是把 **搜索引擎变成了模型的一部分**

数学上属于：

$$
\text{Bayesian inference + Approximate posterior}
$$

而不是深度学习创新。

---

# 七、最重要的一句话总结（本质）

> **RAG = 把语言模型从“无条件生成”变成“有证据约束的条件生成”。**

或者更学术一点：

> RAG 是一个 **带隐变量的条件语言模型的近似推断系统**。

---

