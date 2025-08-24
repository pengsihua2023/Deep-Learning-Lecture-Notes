
## 4. 数学公式化

### (1) 遗忘门

$$
f_t = \sigma \big( W_f [h_{t-1}, x_t] + b_f \big)
$$

*控制前一时刻细胞状态保留的比例。*

### (2) 输入门

$$
i_t = \sigma \big( W_i [h_{t-1}, x_t] + b_i \big)
$$

$$
\tilde{c_t} = \tanh \big( W_c [h_{t-1}, x_t] + b_c \big)  
$$

*决定哪些新信息被存入细胞状态。*

### (3) 细胞状态更新

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

*通过结合保留信息和新信息来更新细胞状态。*

### (4) 输出门

$$
o_t = \sigma \big( W_o [h_{t-1}, x_t] + b_o \big)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

*控制隐藏状态的输出。*

---

## 5. 参数

* \$W\_f, W\_i, W\_c, W\_o\$: 权重矩阵
* \$b\_f, b\_i, b\_c, b\_o\$: 偏置项
* \$h\_t\$: 隐藏状态
* \$c\_t\$: 细胞状态

---

