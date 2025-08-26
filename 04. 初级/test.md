
$$
\min_G \max_D V(D, G) = \mathbb{E}_ {x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

**解释：**

* $\mathbb{E}_ {x \sim p_{\text{data}}(x)}[\log D(x)]$：判别器试图最大化对真实样本的正确分类概率。

* $\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$：判别器试图最大化对生成样本的拒绝概率，而生成器试图让 $D(G(z))$ 接近 1（即欺骗判别器）。

* 生成器 $G$ 希望最小化 $\log(1 - D(G(z)))$，使生成样本尽可能接近真实数据。


