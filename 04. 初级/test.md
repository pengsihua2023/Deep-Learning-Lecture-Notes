

* **全局最优**：当 $p_g = p_{\text{data}}$，目标函数 $V(D,G)$ 达到全局最优，判别器输出 $D(x) = 0.5$。

* **JS散度**：GAN 的优化可以看作最小化生成分布 $p_g$ 和真实分布 $p_{\text{data}}$ 之间的 Jensen–Shannon 散度：

$$
JS(p_{\text{data}} \parallel p_g) = \frac{1}{2} KL\left(p_{\text{data}} \parallel \frac{p_{\text{data}} + p_g}{2}\right) + \frac{1}{2} KL\left(p_g \parallel \frac{p_{\text{data}} + p_g}{2}\right)
$$

---

* **挑战**：

  * **模式崩塌**：生成器可能只生成有限的样本模式，忽略真实数据的多样性。
  * **训练不稳定**：由于对抗性目标，梯度可能震荡或消失。
  * **梯度消失**：当判别器过强，生成器可能无法有效学习。



