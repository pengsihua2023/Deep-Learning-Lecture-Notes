# 多任务学习的数学描述

## 1. 单任务学习的基本形式

给定数据集：

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N,
$$

* $x_i \in \mathcal{X}$：第 $i$ 个样本的输入特征。
* $y_i \in \mathcal{Y}$：第 $i$ 个样本对应的监督信号（标签）。
* $N$：训练样本数量。

我们训练一个参数为 $\theta$ 的模型：

$$
f_\theta : \mathcal{X} \to \mathcal{Y},
$$

目标是最小化期望损失：

${\min_\theta \ \mathbb{E}_{(x,y)\sim \mathcal{D}} }$        

${\left[ \mathcal{L}(f_\theta(x), y) \right].}$


![loss function](https://latex.codecogs.com/svg.latex?\min_{\theta}\;\mathbb{E}_{(x,y)\sim D}[L(f_{\theta}(x),y)])


* $\mathcal{L}(\cdot, \cdot)$：损失函数（如均方误差、交叉熵）。
* $\mathbb{E}_{(x,y)\sim \mathcal{D}}[\cdot]$：对训练数据分布的期望。
* 含义：学习一个模型，使得它在整体数据分布上预测结果尽可能接近真实标签。

---

## 2. 多任务学习的扩展形式

假设有 $T$ 个任务，每个任务 $t$ 的数据集为：

$\mathcal{D}_t = \{(x_i^t, y_i^t)\}_{i=1}^{N_t},$

* $x_i^t$：任务 $t$ 的输入。
* $y_i^t$：任务 $t$ 的标签。
* $N_t$：任务 $t$ 的样本数量。

每个任务对应损失函数 $\mathcal{L}_t$。多任务学习优化目标是：

$$
\min_\theta \ \sum_{t=1}^T \lambda_t \, \mathbb{E}_{(x,y)\sim \mathcal{D}_t} \Big[ \mathcal{L}_t(f_\theta(x), y) \Big].
$$

* $\lambda_t$：任务权重，控制不同任务在整体目标中的重要性。
* 含义：用一个共享参数 $\theta$ 的模型，同时在多个任务上表现良好。

---

## 3. 参数共享的结构化表示

实际中常用 **共享表示层 + 任务专用输出层**：

1. **共享表示层**：

$$
h = \phi_{\theta_s}(x),
$$

* $\phi_{\theta_s}$：特征抽取器（如神经网络的前几层），参数 $\theta_s$ 在所有任务中共享。
* $h$：共享的隐含表示（latent representation）。

2. **任务专用输出层**：

$$
\hat{y}^t = f^t_{\theta_t}(h),
$$

* $f^t_{\theta_t}$：任务 $t$ 的预测器，参数 $\theta_t$ 仅供任务 $t$ 使用。
* $\hat{y}^t$：模型对任务 $t$ 的预测。

整体优化目标：

$$
\min_{\theta_s, \{\theta_t\}_{t=1}^T} \ \sum_{t=1}^T \lambda_t \, \mathbb{E}_{(x,y)\sim \mathcal{D}_t} \left[ \mathcal{L}_t\big(f^t_{\theta_t}(\phi_{\theta_s}(x)), y\big) \right].
$$

* $\theta_s$：所有任务共享的参数（捕捉共性）。
* $\theta_t$：任务 $t$ 的专用参数（捕捉个性）。

---

## 4. 矩阵/正则化视角

若假设任务参数矩阵为：

$$
W = [\theta_1, \dots, \theta_T] \in \mathbb{R}^{d \times T},
$$

* $d$：每个任务参数的维度。
* $T$：任务数。
* $W_t$：矩阵 $W$ 的第 $t$ 列，对应任务 $t$ 的参数。

则可在损失函数外加正则化约束：

### (a) 低秩约束

$$
\min_W \ \sum_{t=1}^T \mathcal{L}_t(W_t) + \lambda \|W\|_*,
$$

* $\|W\|_*$：核范数（矩阵奇异值之和），促使 $W$ 的秩较低，表示任务共享一个低维子空间。

### (b) 图正则化

$$
\min_W \ \sum_{t=1}^T \mathcal{L}_t(W_t) + \gamma \sum_{(i,j)\in E} \|W_i - W_j\|^2,
$$

* $E$：任务关系图的边集合，表示哪些任务彼此相似。
* $\|W_i - W_j\|^2$：鼓励相似任务的参数接近。

---

## 5. 贝叶斯视角

引入任务参数的先验分布：

$$
p(\theta_1, \dots, \theta_T | \alpha) = \prod_{t=1}^T p(\theta_t | \alpha),
$$

* $\alpha$：共享的超参数，控制所有任务的先验分布。
* 含义：通过 $\alpha$ 在不同任务间引入统计上的耦合，从而利用共享知识。

---

## 总结

多任务学习的数学建模有三种主要思路：

1. **加权损失函数**（任务简单相加，带权重 $\lambda_t$）；
2. **参数共享**（共享层 $\theta_s$ + 任务专用头 $\theta_t$）；
3. **正则化 / 概率建模**（通过核范数、图正则化或共享先验建模任务关系）。

---

<img width="1000" height="380" alt="image" src="https://github.com/user-attachments/assets/3f6e0222-c92a-4f1d-bdcb-44bd1b8378ae" />


* **Input $x$** → **Shared representation $\phi_{\theta_s}(x)$**
* 分别进入任务头 $f^t_{\theta_t}(h)$，得到预测 $\hat{y}^t$
* 每个任务计算自己的损失 $\mathcal{L}_t$
* 再乘以任务权重 $\lambda_t$
* 最后加总得到 **总目标 $\sum_{t=1}^T \lambda_t \mathcal{L}_t$**



