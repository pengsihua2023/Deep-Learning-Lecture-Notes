

### (4) 输出门
$$
O_t = \sigma\left(Z_t W_o^\top + \mathbf{1} b_o^\top \right)
$$

$$
H_t = O_t \odot \tanh(C_t)
$$

---

### (5) 输出层
对于分类问题：

$$
\hat{Y}_t = \text{softmax}\left(H_t W_y^\top + \mathbf{1} b_y^\top\right)
$$

其中

$\hat{Y}_t \in \mathbb{R}^{B \times d_y}, \quad
W_y \in \mathbb{R}^{d_y \times H}, \quad
b_y \in \mathbb{R}^{d_y}$

---

### (6) 损失函数
交叉熵损失在批量上的平均：

$$
\mathcal{L} = - \frac{1}{B} \sum_{b=1}^{B} \sum_{t=1}^{T} y_{b,t}^\top \log \hat{y}_{b,t}
$$

---

 
