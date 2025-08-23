$\mathbb{E}_{q_\phi(z \mid x)}[\cdot]$：在编码器分布 $q_\phi(z \mid x)$ 下的期望； 

$_{q_\phi(z \mid x)}[\cdot]$：在编码器分布 $q_\phi(z \mid x)$ 下的期望； 

### (1) Kullback–Leibler Divergence (KL 散度)

* **$P(x)$**：真实分布（target / data distribution），表示在事件 $x$ 上的真实概率。
* **$Q(x)$**：近似分布或模型分布（approximation / model distribution），表示在事件 $x$ 上模型的估计概率。
* **解释**： $D_{\mathrm{KL}}(P\|Q)$ 衡量在用 $Q$ 来近似 $P$ 时，信息损失的大小。

---

### (2) Jensen–Shannon Divergence (JS 散度)

* **$P, Q$**：两个概率分布。
* **$M = \frac{1}{2}(P+Q)$**：混合分布，即对两个分布取平均。
* **解释**：JS 散度是基于 KL 散度的对称化版本，保证了**对称性**（即 $D_{JS}(P\|Q) = D_{JS}(Q\|P)$）和**有界性**（取值在 \[0, 1] 之间，若使用 log base 2）。

---

### (3) Wasserstein Distance (Earth Mover’s Distance, EMD)

* **$P, Q$**：两个概率分布。
* **$\Pi(P, Q)$**：所有可能的联合分布（couplings），边缘分布分别为 $P$ 和 $Q$。
* **$E(x,y) \sim \gamma$**：表示在联合分布 $\gamma$ 下的期望。
* **$\|x-y\|$**：从点 $x$ 移动到点 $y$ 的“距离”（通常是欧氏距离）。
* **解释**：Wasserstein 距离衡量将一个分布“搬运”成另一个分布所需的最小代价。

---

👉 简单总结：

* **KL 散度**：不对称，衡量分布差异，常用于信息论与概率建模。
* **JS 散度**：对称化后的 KL 散度，常用于比较两个分布的相似性。
* **Wasserstein 距离**：考虑“搬运代价”的分布差异度量，常用于生成对抗网络（WGAN）。


