## PU学习的数学描述

### 1. 问题设定

设特征空间 $\mathcal{X} \subseteq \mathbb{R}^d$，真实标签 $y \in \{+1, -1\}$。定义：

- 正类条件密度： $p_p(x) := p(x \mid y = +1)$
- 负类条件密度： $p_n(x) := p(x \mid y = -1)$（**从不可观测**）
- 类别先验： $\pi := P(y = +1)$，$0 < \pi < 1$
- 混合密度（即"未标记"数据的真实分布）：

$$
p(x) = \pi\, p_p(x) + (1-\pi)\, p_n(x)
$$

**观测到的数据：**

$$
\{x_i^p\}_{i=1}^{n_p} \overset{\text{iid}}{\sim} p_p(x), \qquad \{x_j^u\}_{j=1}^{n_u} \overset{\text{iid}}{\sim} p(x)
$$

注意：U 集合并不是从 $p_n(x)$ 采样，而是从**混合分布**采样——这是 PU 学习和普通二分类的本质区别。

### 2. 标记机制假设（SCAR）

引入一个"是否被标记"的隐变量 $s \in \{0,1\}$。标准假设 **SCAR**（Selected Completely At Random）：

$$
P(s = 1 \mid y = +1, x) = P(s=1 \mid y=+1) = c \quad (\text{与 } x \text{ 无关})
$$

即已标记的正样本是全体正样本中的一个完全随机子集。这个假设保证了后面风险重写的等式成立。

在这一假设下，Elkan & Noto (2008) 给出一个有用的恒等式：若 $f(x) := P(s=1\mid x)$ 是从"$s$ vs 其余"训练出的分类器，则

$$
P(y=1 \mid x) = \frac{f(x)}{c}
$$

### 3. 目标风险

我们真正想最小化的是标准分类风险：

$$
R(g) = \pi\, \mathbb{E}_{x \sim p_p}\big[\ell(g(x), +1)\big] + (1-\pi)\, \mathbb{E}_{x \sim p_n}\big[\ell(g(x), -1)\big]
$$

问题：第二项需要从 $p_n$ 采样，而我们没有负样本。

### 4. 无偏风险重写（uPU）

利用混合分布的定义 $p(x) = \pi p_p(x) + (1-\pi) p_n(x)$，可得

$$
(1-\pi)\, p_n(x) = p(x) - \pi\, p_p(x)
$$

代入负类项：

$$
(1-\pi)\,\mathbb{E}_{p_n}[\ell(g(x),-1)] = \mathbb{E}_{p(x)}[\ell(g(x),-1)] - \pi\,\mathbb{E}_{p_p}[\ell(g(x),-1)]
$$

于是目标风险可以**完全用 P 和 U 的分布表示**：

$$
R(g) = \pi\, \mathbb{E}_{p_p}\big[\ell(g(x),+1) - \ell(g(x),-1)\big] + \mathbb{E}_{p(x)}\big[\ell(g(x),-1)\big]
$$

对应的经验风险估计量（du Plessis et al., 2014/2015）：

$$
\widehat{R}_{\text{pu}}(g) = \frac{\pi}{n_p}\sum_{i=1}^{n_p}\ell(g(x_i^p),+1) \;-\; \frac{\pi}{n_p}\sum_{i=1}^{n_p}\ell(g(x_i^p),-1) \;+\; \frac{1}{n_u}\sum_{j=1}^{n_u}\ell(g(x_j^u),-1)
$$

可以证明 $\mathbb{E}\big[\widehat{R}_{\text{pu}}(g)\big] = R(g)$，即这是一个**无偏估计**。

### 5. 深度模型下的问题：负风险

把上式拆成正类项和"负类整体"两块：

$$
\widehat{R}_{\text{pu}}(g) = \pi\, \widehat{R}_p^{+}(g) + \underbrace{\left[\widehat{R}_u^{-}(g) - \pi\, \widehat{R}_p^{-}(g)\right]}_{=: \widehat{R}_{\text{pu}}^{-}(g)}
$$

其中 $\widehat{R}_p^{+}(g) = \frac{1}{n_p}\sum \ell(g(x_i^p),+1)$，$\widehat{R}_p^{-}, \widehat{R}_u^{-}$ 类似定义。

理论上  $\widehat{R}_{\text{pu}}^{-}(g) \ge 0$ （因为它逼近  $(1-\pi)\mathbb{E}_{p_n}[\ell(\cdot,-1)] \ge 0$ ） ，但当 $g$ 的假设空间足够灵活（深度网络）时，训练过程会让  $\widehat{R}_{\text{pu}}^{-}(g)$  **在有限样本上跑到负值**并被无限压低——模型通过过拟合让经验风险发散到  $-\infty$ ，导致严重过拟合。

### 6. 非负修正（nnPU）

Kiryo et al. (2017) 的修正：直接对负类分风险施加非负约束

$$
\widetilde{R}_{\text{pu}}(g) = \pi\, \widehat{R}_p^{+}(g) + \max\Big(0,\; \widehat{R}_u^{-}(g) - \pi\, \widehat{R}_p^{-}(g)\Big)
$$

当  $\widehat{R}_u^{-}(g) - \pi \widehat{R}_p^{-}(g) < 0$ 时，实践中常用**梯度反转**：对该 batch 反向传播 $-\big(\widehat{R}_u^{-}-\pi\widehat{R}_p^{-}\big)$ 的梯度而非直接截断为 0，以避免梯度消失、把模型往回拉。这是目前深度学习场景下 PU 学习的标准做法。

### 7. 类别先验 $\pi$ 的估计

上述所有推导都假设 $\pi$ 已知，但实际中通常未知，需要单独估计（这是 PU 学习里独立的一个子问题，称为 **mixture proportion estimation**），常见方法：
- KM 估计器（Ramaswamy et al., 2016，基于核嵌入距离）
- 基于 $\text{AUC}$/决策函数分布尖峰的启发式方法（Elkan & Noto 的 $c$ 估计）
- 假设 $p_n$ 与 $p_p$ 的支撑集不完全重叠，利用密度比的下界估计 $\pi$

$\pi$ 的估计误差会直接线性地传递到 $\widehat{R}_{\text{pu}}(g)$ 的偏差中，是实践中效果不稳定的主要来源之一。

---

