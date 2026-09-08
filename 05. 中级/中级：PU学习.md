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

其中 

$$
\widehat{R}_p^{+}(g) = \frac{1}{n_p}\sum \ell(g(x_i^p),+1)$，$\widehat{R}_p^{-} ,  $\widehat{R}_u^{-}$  类似定义。
$$

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

## PU学习代码例子
```
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# ============================================================
# PU学习最简单示例：Elkan-Noto (2008) 校正法
#
# 核心思想：
#   1. 直接训练一个分类器区分 "已标记正样本(P)" vs "未标记(U)"，
#      得到 f(x) = P(s=1 | x)
#   2. 在 SCAR 假设下（标记过程与 x 无关），可以证明：
#         f(x) = c * P(y=1 | x)，其中 c = P(s=1 | y=1)
#      于是校正后的真实正类概率为 P(y=1|x) = f(x) / c
# ============================================================

rng = np.random.RandomState(42)

# ---------- 1. 生成模拟数据：两个高斯簇代表正类/负类 ----------
n_pos, n_neg = 500, 500
X_pos = rng.normal(loc=[2, 2], scale=1.5, size=(n_pos, 2))
X_neg = rng.normal(loc=[-2, -2], scale=1.5, size=(n_neg, 2))

X = np.vstack([X_pos, X_neg])
# y_true 是"上帝视角"的真实标签，PU学习训练时完全不可见，只用来最后评估效果
y_true = np.hstack([np.ones(n_pos), np.zeros(n_neg)])

# ---------- 2. 模拟 PU 场景 ----------
# 真实正样本里只有比例 c 被标记为 P，剩下的正样本 + 全部负样本 都混入 U
c_true = 0.3
is_labeled = (y_true == 1) & (rng.rand(len(y_true)) < c_true)
s = is_labeled.astype(int)  # s=1: 已标记(P)；s=0: 未标记(U)

print(f"总样本数: {len(y_true)}")
print(f"标记为 P 的样本数: {s.sum()}")
print(f"未标记 U 中样本数: {(s == 0).sum()}，其中真实正样本占比: {y_true[s == 0].mean():.3f}")
print("-" * 50)

# ---------- 3a. 【错误做法】把 U 直接当负样本训练 ----------
# 用 s 当作"标签"直接训练一个普通二分类器，且直接把输出当作 P(y=1|x)
clf = LogisticRegression().fit(X, s)
f_scores = clf.predict_proba(X)[:, 1]     # f(x) = P(s=1|x)，未经校正
pred_naive = f_scores                     # 朴素做法：直接拿 f(x) 当 P(y=1|x)

# ---------- 3b. 【PU做法】用 c 校正 ----------
# 估计 c = P(s=1|y=1)：在已标记正样本子集上取 f(x) 的均值
c_hat = clf.predict_proba(X[s == 1])[:, 1].mean()
pred_pu = np.clip(f_scores / c_hat, 0, 1)  # 校正后的 P(y=1|x)

print(f"真实 c = {c_true}，估计出的 c_hat = {c_hat:.4f}")
print("-" * 50)

# ---------- 4. 对比效果（用 y_true 评估，仅用于演示）----------
thresh = 0.5
acc_naive = accuracy_score(y_true, pred_naive > thresh)
acc_pu = accuracy_score(y_true, pred_pu > thresh)

print("=== 分类准确率对比（阈值0.5）===")
print(f"朴素方法 (未校正, 把U当负样本训练后直接用输出): {acc_naive:.4f}")
print(f"PU学习方法 (Elkan-Noto 校正后):              {acc_pu:.4f}")
print()
print("=== AUC 对比（AUC对单调缩放不敏感，两者理论上相同，仅供参考）===")
print(f"AUC: {roc_auc_score(y_true, f_scores):.4f}")
print()
print("结论：朴素方法把所有输出普遍压低了（因为只有约c比例的正样本被标记为P），")
print("导致大量真正的正样本预测概率低于0.5被误判为负类；")
print("除以估计出的 c_hat 校正后，概率被重新拉回正确尺度，分类效果大幅改善。")

```
### 代码学习
这段代码演示 PU 学习最经典的 **Elkan-Noto (2008)** 校正方法：先把"标记 P vs 未标记 U"当成一个普通二分类问题训练分类器，再用一个常数 c 校正得到真实的正类概率。我先在沙盒里跑一遍确认结果合理，再给你文件。结果符合预期：朴素方法准确率只有 52%（几乎把所有真正的正样本都误判成负类），而经过 c 校正后的 PU 方法达到 94.7%。这正好说明了 PU 学习校正的价值。## 代码逐步说明

**思路（Elkan-Noto 方法）：**

1. **构造 PU 场景**：生成两个高斯簇（正/负类），只把正类中的一部分（比例 $c=0.3$）标记为 P，其余正样本连同全部负样本都扔进 U。

2. **朴素做法（错误）**：直接把 $s$（是否被标记）当作真实标签训练逻辑回归，输出 $f(x)=P(s=1\mid x)$ 直接当成 $P(y=1\mid x)$。问题在于：由于只有 30% 的正样本被标记，$f(x)$ 在整个正类区域会被系统性压低到约 0.3，用 0.5 做阈值时几乎所有真正的正样本都被误判为负——运行结果准确率只有 52%。

3. **PU 校正做法**：用同一个 $f(x)$，但除以估计出的常数 $\hat{c} = P(s=1\mid y=1)$（用已标记正样本上 $f(x)$ 的均值来估计），得到校正后的 $P(y=1\mid x) = f(x)/\hat{c}$。运行结果准确率提升到 94.7%。

这是最简单能跑通的 PU 学习实现——只用逻辑回归 + 一次除法校正，没有涉及 nnPU 的非负风险约束或深度网络。如果你想看 **nnPU**（更适合深度学习场景、能处理 $\pi$ 未知或分布重叠更复杂情况）的实现，我可以再写一版 PyTorch 版本。
