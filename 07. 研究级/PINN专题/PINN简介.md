<img width="614" height="614" alt="image" src="https://github.com/user-attachments/assets/ed51a778-660f-4ebc-9a96-ef4cb5180f95" />  
  
George Em Karniadakis

---

# PINN 简介

**Physics-Informed Neural Networks (PINNs)** 是一种结合 **深度学习** 与 **物理方程** 的数值方法。它的核心思想是：

* 用 **神经网络** 近似 PDE/ODE 的解函数 $u(x,t,...)$。
* 在训练过程中，不仅最小化数据误差，还把 **物理方程残差**（PDE、边界条件、初始条件）作为损失函数的一部分，强制神经网络满足物理规律。

这样，PINN 既利用了数据驱动的优势，又融入了物理约束，能在数据稀缺或噪声较大时获得更可靠的解。

---

## 1. PINN 的基本思想

设 PDE 为：

---math
\mathcal{N}[u](x,t) = 0, \quad (x,t)\in \Omega,
```

其中 $\mathcal{N}$ 是微分算子。

* 神经网络：用 $u_\theta(x,t)$（参数 $\theta$）近似解。
* 残差定义：

$$
r_\theta(x,t) = \mathcal{N}[u_\theta](x,t).
$$

* 损失函数：

$$
\mathcal{L}(\theta) = \underbrace{\frac{1}{N_f}\sum |r_\theta(x_f,t_f)|^2}_{\text{PDE 残差}}
+ \underbrace{\frac{1}{N_b}\sum |u_\theta(x_b,t_b)-g_b|^2}_{\text{边界条件}}
+ \underbrace{\frac{1}{N_0}\sum |u_\theta(x_0,0)-g_0|^2}_{\text{初始条件}}
+ \underbrace{\frac{1}{N_d}\sum |u_\theta(x_d,t_d)-u^\text{obs}|^2}_{\text{观测数据 (可选)}}.
$$

训练就是优化神经网络参数 $\theta$，让解尽可能同时满足物理规律与数据约束。

---

## 2. PINN 的典型应用

1. **正问题 (Forward Problem)**

   * 已知 PDE 和参数，预测解 $u(x,t)$。
   * 例如：求解热传导方程、Navier–Stokes 方程。

2. **逆问题 (Inverse Problem)**

   * 已知部分观测数据，通过 PINN 同时学习解函数和未知参数（或系数）。
   * 例如：从温度观测反演热导率，从流场数据反演粘性系数。

3. **数据同化 (Data Assimilation)**

   * 在有稀疏、带噪声数据时，PINN 可以利用 PDE 物理规律进行补全和去噪。

---

## 3. PINN 的优势

* **物理一致性**：保证解符合 PDE，而不是仅靠拟合数据。
* **少样本需求**：只需少量数据点，就能训练出合理解。
* **通用性强**：适用于 ODE、PDE（包括高维复杂 PDE）。
* **自然处理逆问题**：无需修改数值格式，直接通过损失函数反演。

---

## 4. PINN 的挑战

* **训练难**：损失函数非凸，高阶 PDE 时梯度传播不稳定。
* **收敛慢**：需要较多训练时间，相比传统数值方法（如有限元/有限差分）计算开销大。
* **多尺度问题困难**：在高频解或强非线性 PDE 中效果不稳定。
* **参数选择敏感**：采样点数量、网络深度、激活函数都会影响结果。

---

## 5. PINN 与传统数值方法的对比

| 特点   | PINN                   | 有限元/有限差分        |
| ---- | ---------------------- | --------------- |
| 维度   | 高维问题仍可处理（克服“维数灾难”一定程度） | 高维很难扩展          |
| 数据   | 可结合实验/观测数据             | 纯物理计算           |
| 计算效率 | 训练代价高，推理快              | 通常更快更稳定         |
| 应用场景 | 稀疏数据、逆问题、复杂几何          | 工程仿真、工业级 PDE 求解 |

---

## 总结：
**PINN 是一种融合物理规律与深度学习的神经网络方法，适用于 PDE/ODE 的正问题与逆问题，尤其在数据稀缺时有优势，但训练过程较难，仍处于研究与探索阶段。**

---


Links to the Original Papers

[Part I on arXiv (2017)] – Physics Informed Deep Learning (Part I): Data‑driven Solutions of Nonlinear Partial Differential Equations 

[Part II on arXiv (2017)] – Physics Informed Deep Learning (Part II): Data‑driven Discovery of Nonlinear Partial Differential Equations 

[Journal Publication (2019)] – Physics‑informed neural networks: A deep learning framework… in Journal of Computational Physics 
