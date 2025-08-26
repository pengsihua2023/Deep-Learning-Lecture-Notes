

**1. 优化判别器：**

* 固定生成器 $G$，用真实样本 $x \sim p_{\text{data}}$ 和生成样本 $G(z) \sim p_z$ 训练判别器，最大化：

$$
V(D) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
$$

* 通常使用梯度上升更新 $D$ 的参数。

---

**2. 优化生成器：**

* 固定判别器 $D$，用噪声 $z \sim p_z$ 生成样本 $G(z)$，最小化：

$$
V(G) = \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
$$

* 实践中，常优化等价形式

$$
\max_G \mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$

因为原始形式梯度可能不稳定（尤其当 $D(G(z)) \approx 0$ 时）。



