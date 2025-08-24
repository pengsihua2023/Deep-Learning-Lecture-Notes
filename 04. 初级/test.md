
---


## 循环神经网络 (RNN) 数学描述

给定输入序列：

$$
X = \{x_1, x_2, \dots, x_T\}, \quad x_t \in \mathbb{R}^{d_x}
$$

RNN 的核心是 **隐藏状态递推**，它能捕捉序列依赖关系。  

---

### (1) 隐藏状态更新

$$
h_t = \phi\!\left(W_h h_{t-1} + W_x x_t + b_h \right), \quad t=1,2,\dots,T
$$

其中：  
- $h_t \in \mathbb{R}^{d_h}$：时刻 $t$ 的隐藏状态  
- $W_h \in \mathbb{R}^{d_h \times d_h}$：隐藏层到隐藏层的权重矩阵  
- $W_x \in \mathbb{R}^{d_h \times d_x}$：输入到隐藏层的权重矩阵  
- $b_h \in \mathbb{R}^{d_h}$：偏置项  
- $\phi(\cdot)$：激活函数（常用 $\tanh$ 或 ReLU）  

初始条件：

$$
h_0 = \mathbf{0} \quad (\text{零向量}) \quad \text{或可学习的参数向量}
$$

---

### (2) 输出层

$$
\hat{y}_t = \text{softmax}\!\left(W_y h_t + b_y \right), \quad \hat{y}_t \in \mathbb{R}^{d_y}
$$

其中：  
- $W_y \in \mathbb{R}^{d_y \times d_h}$：隐藏层到输出层的权重矩阵  
- $b_y \in \mathbb{R}^{d_y}$：输出层偏置  

---

### (3) 损失函数
对序列任务（如分类/语言建模），常用交叉熵损失：  

$$
\mathcal{L} = - \sum_{t=1}^{T} y_t^\top \log \hat{y}_t
$$

其中 $y_t$ 为真实标签的 one-hot 向量，$\hat{y}_t$ 为预测概率分布。  

---

### (4) 参数更新
通过 **时间反向传播 (Backpropagation Through Time, BPTT)** 计算梯度：  

$$
\theta \leftarrow \theta - \eta \, \frac{\partial \mathcal{L}}{\partial \theta}
$$

其中：  

$$
\theta \in \{W_x, W_h, W_y, b_h, b_y\}, \quad \eta: \text{学习率}
$$

---

## 总结
RNN 的核心公式：
$$
\begin{aligned}
h_t &= \phi(W_h h_{t-1} + W_x x_t + b_h) \\
\hat{y}_t &= \text{softmax}(W_y h_t + b_y)
\end{aligned}
$$
```

---



