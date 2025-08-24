## 4. 数学形式化（小批量版本）

假设输入批量大小为 $B$，隐藏层维度为 $H$，输入维度为 $d_x$。  
输入矩阵 $X_t \in \mathbb{R}^{B \times d_x}$，隐藏状态矩阵 $H_{t-1} \in \mathbb{R}^{B \times H}$。  
定义拼接矩阵：
\[
Z_t = \begin{bmatrix} H_{t-1} & X_t \end{bmatrix} \in \mathbb{R}^{B \times (H+d_x)}
\]

---

### (1) 遗忘门
\[
F_t = \sigma\!\left(Z_t W_f^\top + \mathbf{1} b_f^\top \right)
\]
其中 $W_f \in \mathbb{R}^{H \times (H+d_x)}, \; b_f \in \mathbb{R}^{H} $

---

### (2) 输入门
\[
I_t = \sigma\!\left(Z_t W_i^\top + \mathbf{1} b_i^\top \right)
\]

\[
\tilde{C}_t = \tanh\!\left(Z_t W_c^\top + \mathbf{1} b_c^\top \right)
\]

---

### (3) 细胞状态更新
\[
C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t
\]

---

### (4) 输出门
\[
O_t = \sigma\!\left(Z_t W_o^\top + \mathbf{1} b_o^\top \right)
\]

\[
H_t = O_t \odot \tanh(C_t)
\]

---

### (5) 输出层
对于分类问题：
\[
\hat{Y}_t = \text{softmax}\!\left(H_t W_y^\top + \mathbf{1} b_y^\top\right)
\]

其中 $\hat{Y}_t \in \mathbb{R}^{B \times d_y}$，$W_y \in \mathbb{R}^{d_y \times H}, \; b_y \in \mathbb{R}^{d_y}$。

---

### (6) 损失函数
交叉熵损失在批量上的平均：
\[
\mathcal{L} = - \frac{1}{B} \sum_{b=1}^{B} \sum_{t=1}^{T} y_{b,t}^\top \log \hat{y}_{b,t}
\]

---

### (7) 参数更新
使用 BPTT 更新参数：
\[
\theta \leftarrow \theta - \eta \, \frac{\partial \mathcal{L}}{\partial \theta}
\]
其中 
\[
\theta \in \{ W_f, W_i, W_c, W_o, W_y, \, b_f, b_i, b_c, b_o, b_y \}
\]

---

## 5. 参数（小批量形式）

- $W_f, W_i, W_c, W_o \in \mathbb{R}^{H \times (H+d_x)}$: 权重矩阵  
- $b_f, b_i, b_c, b_o \in \mathbb{R}^{H}$: 偏置向量  
- $W_y \in \mathbb{R}^{d_y \times H}, \; b_y \in \mathbb{R}^{d_y}$: 输出层参数  
- $H_t \in \mathbb{R}^{B \times H}$: 隐藏状态  
- $C_t \in \mathbb{R}^{B \times H}$: 细胞状态  
- $\hat{Y}_t \in \mathbb{R}^{B \times d_y}$: 模型预测  
- $\mathcal{L}$: 平均交叉熵损失  
