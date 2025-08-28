#基于PIKAN解SEIR动力学方程

## 1. SEIR动力学方程

$$
\begin{aligned}
\frac{dS}{dt} &= -\beta S I, \\
\frac{dE}{dt} &= \beta S I - \sigma E, \\
\frac{dI}{dt} &= \sigma E - \gamma I, \\
\frac{dR}{dt} &= \gamma I.
\end{aligned}
$$


## 2. 神经网络近似解

假设 PIKAN / PINN 网络输出为：

$$
\hat{S}(t), \hat{E}(t), \hat{I}(t), \hat{R}(t).
$$

并通过自动微分得到它们的导数：

$$
\frac{d\hat{S}}{dt},\quad \frac{d\hat{E}}{dt},\quad \frac{d\hat{I}}{dt},\quad \frac{d\hat{R}}{dt}.
$$



## 3. 方程残差定义

把预测值代入 SEIR 方程，得到残差：

$$
\begin{aligned}
r_S(t) &= \frac{d\hat{S}}{dt} + \beta \hat{S}(t)\hat{I}(t), \\
r_E(t) &= \frac{d\hat{E}}{dt} - \big(\beta \hat{S}(t)\hat{I}(t) - \sigma \hat{E}(t)\big), \\
r_I(t) &= \frac{d\hat{I}}{dt} - \big(\sigma \hat{E}(t) - \gamma \hat{I}(t)\big), \\
r_R(t) &= \frac{d\hat{R}}{dt} - \gamma \hat{I}(t).
\end{aligned}
$$



## 4. 物理残差损失函数

对采样的时间点 $\{t_i\}_{i=1}^N$，物理残差的损失是：

$$
L_{\text{physics}} = \frac{1}{N} \sum_{i=1}^N \Big( r_S(t_i)^2 + r_E(t_i)^2 + r_I(t_i)^2 + r_R(t_i)^2 \Big).
$$



## 5. 总损失

完整的 PIKAN 损失函数可以写为：

$$
L = L_{\text{data}} + \lambda_{\text{phys}} L_{\text{physics}} + \lambda_{\text{cons}} L_{\text{conservation}},
$$

其中守恒约束：

$$
L_{\text{conservation}} = \frac{1}{N} \sum_{i=1}^N \Big( \hat{S}(t_i)+\hat{E}(t_i)+\hat{I}(t_i)+\hat{R}(t_i)-1 \Big)^2.
$$




要不要我帮你把这个公式直接翻译成 **PyTorch 代码的 `loss_fn`**，和你前面跑的训练脚本一一对应？
