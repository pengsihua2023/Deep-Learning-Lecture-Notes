

---


1. **PDE 问题定义**

我们要求解偏微分方程 (PDE)：

$ \mathcal{N}\[u(\mathbf{x}, t)] = f(\mathbf{x}, t), \quad \mathbf{x} \in \Omega, ; t \in \[0, T] $

边界条件和初始条件为：

$ \mathcal{B}\[u(\mathbf{x}, t)] = g(\mathbf{x}, t), \quad \mathbf{x} \in \partial\Omega $

$ u(\mathbf{x}, 0) = u\_0(\mathbf{x}) $

---

2. **传统 PINN 的损失函数**

\$ \mathcal{L}(\theta) = \mathcal{L}*{\text{PDE}} + \mathcal{L}*{\text{BC}} + \mathcal{L}\_{\text{IC}} \$

其中：

\$ \mathcal{L}*{\text{PDE}} = \frac{1}{N\_r} \sum*{i=1}^{N\_r} \big( \mathcal{N}\[u\_\theta(\mathbf{x}\_i, t\_i)] - f(\mathbf{x}\_i, t\_i) \big)^2 \$

\$ \mathcal{L}*{\text{BC}} = \frac{1}{N\_b} \sum*{i=1}^{N\_b} \big( \mathcal{B}\[u\_\theta(\mathbf{x}\_i, t\_i)] - g(\mathbf{x}\_i, t\_i) \big)^2 \$

\$ \mathcal{L}*{\text{IC}} = \frac{1}{N\_0} \sum*{i=1}^{N\_0} \big( u\_\theta(\mathbf{x}\_i, 0) - u\_0(\mathbf{x}\_i) \big)^2 \$

---

3. **ARSPINN 的残差分割**

\$ \mathcal{L}*{\text{PDE}} = \sum*{k=1}^K w\_k , \mathcal{L}\_{\text{PDE}, k} \$

其中：

\$ \mathcal{L}*{\text{PDE}, k} = \frac{1}{N*{r,k}} \sum\_{(\mathbf{x}*i, t\_i) \in \Omega\_k} \big( \mathcal{N}\[u*\theta(\mathbf{x}\_i, t\_i)] - f(\mathbf{x}\_i, t\_i) \big)^2 \$

\$ w\_k = \frac{\exp(\alpha , \mathcal{L}*{\text{PDE}, k})}{\sum*{j=1}^K \exp(\alpha , \mathcal{L}\_{\text{PDE}, j})} \$

---

4. **总损失函数**

\$ \mathcal{L}(\theta) = \sum\_{k=1}^K w\_k , \mathcal{L}*{\text{PDE}, k} + \lambda\_b , \mathcal{L}*{\text{BC}} + \lambda\_0 , \mathcal{L}\_{\text{IC}} \$

---


---

要不要我帮你把 **代码注释里的公式** 也换成这种 `$...$` 格式，这样代码和文档风格就完全统一？


