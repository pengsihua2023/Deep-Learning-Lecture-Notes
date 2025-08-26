对比学习（Contrastive Learning）是一类通过 **相似样本“拉近”、不相似样本“推远”** 来学习表示的机器学习方法。下面我给你一个数学化的描述：

---

## 1. 数据与表示

设有数据点集合

$$
\{x_i\}_{i=1}^N, \quad x_i \in \mathcal{X}
$$

通过一个编码器（如神经网络) $f_\theta: \mathcal{X} \to \mathbb{R}^d$ 将输入映射到特征空间：

$$
z_i = f_\theta(x_i) \in \mathbb{R}^d
$$

通常我们会对 $z_i$ 做归一化：

$$
\tilde{z}_i = \frac{z_i}{\|z_i\|}
$$

---

## 2. 正样本与负样本

对比学习的核心是 **构造正样本对和负样本对**：

* **正样本对（positive pair）**：如 $(x_i, x_i^+)$，通常是同一个样本的两种增强视图（data augmentation），例如图片的两种随机裁剪。
* **负样本对（negative pair）**：如 $(x_i, x_j^-)$，通常是来自不同样本的数据。

目标是让 $f_\theta(x_i)$ 更接近 $f_\theta(x_i^+)$，同时远离 $f_\theta(x_j^-)$。

---

## 3. 相似度度量

常用 **余弦相似度**：

$$
s(z_i, z_j) = \frac{z_i^\top z_j}{\|z_i\|\|z_j\|}
$$

---

## 4. 损失函数（Contrastive Loss）

最常见的形式是 **InfoNCE loss**：

$$
\mathcal{L}_i = - \log \frac{\exp\big( s(z_i, z_i^+) / \tau \big)}{\sum_{j=1}^N \exp\big( s(z_i, z_j) / \tau \big)}
$$

其中：

* $z_i^+$ 是与 $z_i$ 配对的正样本
* 其他 $z_j$ （$j \neq i$）视为负样本
* $\tau > 0$ 是温度系数，控制分布的平滑性

总损失是所有样本的平均：

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_i
$$

---

## 5. 直观解释

* 分子：鼓励正样本相似度 $s(z_i, z_i^+)$ 高
* 分母：抑制负样本相似度 $s(z_i, z_j)$ 高
* 结果：模型学习到的特征空间中，相似数据会聚在一起，不相似数据会被推开

