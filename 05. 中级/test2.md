

* **Input $x$** → **Shared representation $\phi_{\theta_s}(x)$**
* 分别进入任务头 $f^t_{\theta_t}(h)$，得到预测 $\hat{y}^t$
* 每个任务计算自己的损失 $\mathcal{L}_t$
* 再乘以任务权重 $\lambda_t$
* 最后加总得到 **总目标 $\sum_{t=1}^T \lambda_t \mathcal{L}_t$**

这样可以同时看清 **前向传播** 和 **损失加权聚合**。

