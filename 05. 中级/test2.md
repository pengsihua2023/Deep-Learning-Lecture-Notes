
$$
\mathcal{D}_t = \{(x_i^t, y_i^t)\}_{i=1}^{N_t} ,
$$

$\mathcal{D}_t = $  ${ (x_i^t, y_i^t) \}_{i=1}^{N_t} ,$


多任务学习（Multi-Task Learning, MTL）的数学描述通常建立在**机器学习的优化问题**框架下，可以从单任务学习推广而来。下面给出常见的数学形式：

---

## 1. 单任务学习的基本形式

给定数据集

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N,
$$

其中 $x_i \in \mathcal {X}$ 为输入，$y_i \in \mathcal{Y}$ 为标签。我们训练一个模型 $f_\theta(x)$，参数为 $\theta$，目标是最小化期望损失：

$\min_{\theta} \mathbb{E}_ {(x,y)\sim \mathcal{D}} \big[ \mathcal{L}(f_{\theta}(x), y) \big].$

---

## 2. 多任务学习的扩展形式

假设有 $T$ 个任务，每个任务 $t \in \{1,\dots,T\}$ 对应数据集

$$
\mathcal{D}_t = \{(x_i^t, y_i^t)\}_{i=1}^{N_t} ,
$$

损失函数为 $\mathcal{L}_t$。
多任务学习的目标是**同时优化多个任务的损失**，一般形式为：

$\min_{\theta} 
\sum\nolimits_{t=1}^T \lambda_t 
\mathbb{E}_{(x,y)\sim \mathcal{D}_t}
\left[ \mathcal{L}_t\big(f\theta(x), y\big) \right],$


其中 $\lambda_t \geq 0$ 是任务权重，控制不同任务的重要性。

---

## 3. 参数共享的结构化表示

在实际中，MTL常采用**共享表示 + 任务专用头部**的形式：

* 共享表示层：

$$
h = \phi_{\theta_s}(x),
$$

其中 $\theta_s$ 是共享参数。

* 每个任务的专用输出层：

$$
\hat{y}^t = f^t_{\theta_t}(h),
$$

其中 $\theta_t$ 是任务 $t$ 的专用参数。

优化目标变为：

$$
\min_{\theta_s, \{\theta_t\}_{t=1}^T} \ \sum_{t=1}^T \lambda_t \, \mathbb{E}_{(x,y)\sim \mathcal{D}_t} \left[ \mathcal{L}_t\big(f^t_{\theta_t}(\phi_{\theta_s}(x)), y\big) \right].
$$

---

## 4. 矩阵/正则化视角

如果任务间参数存在相关性，可以引入**结构化正则化**。例如：

* 假设任务参数矩阵

$$
W = [\theta_1, \dots, \theta_T] \in \mathbb{R}^{d \times T},
$$

可以施加低秩约束（捕捉共享子空间）：

$$
\min_W \ \sum_{t=1}^T \mathcal{L}_t(W_t) + \lambda \|W\|_*,
$$

其中 $\|W\|_*$ 是核范数。

* 或者通过图正则化建模任务关系：

$$
\min_W \ \sum_{t=1}^T \mathcal{L}_t(W_t) + \gamma \sum_{(i,j)\in E} \|W_i - W_j\|^2,
$$

其中 $E$ 表示任务关系图的边。

---

## 5. 贝叶斯视角

在贝叶斯MTL中，可以假设任务参数 $\theta_t$ 来自某个共享先验 $p(\theta_t | \alpha)$，如：

$$
p(\theta_1, \dots, \theta_T | \alpha) = \prod_{t=1}^T p(\theta_t | \alpha),
$$

从而通过共享超参数 $\alpha$ 建模任务间关系。

---

总结：多任务学习的数学描述主要有三种思路：  

1. **加权损失函数**：直接对所有任务的损失加权求和。
2. **参数共享**：公共表示层 + 任务专用层。
3. **结构化正则化/概率建模**：通过矩阵分解、图正则化或贝叶斯先验来表达任务关系。


