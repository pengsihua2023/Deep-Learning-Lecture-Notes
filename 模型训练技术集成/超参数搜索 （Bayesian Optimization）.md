## 超参数搜索 （Bayesian Optimization）
### 什么是超参数搜索（Bayesian Optimization）？

贝叶斯优化（Bayesian Optimization）是一种基于概率模型的全局优化算法，常用于机器学习中的超参数搜索（Hyperparameter Tuning）。它特别适用于目标函数评估代价高昂的场景，如训练一个深度学习模型需要大量计算资源。不同于网格搜索（Grid Search）或随机搜索（Random Search），贝叶斯优化通过构建一个代理模型（通常是高斯过程，Gaussian Process）来模拟目标函数的分布，并使用采集函数（Acquisition Function，如预期改进Expected Improvement）来智能选择下一个要评估的超参数点，从而高效地探索超参数空间。

#### 核心原理
- **代理模型**：使用高斯过程或其他概率模型来拟合已评估点的性能，预测未评估点的均值和不确定性。
- **采集函数**：基于代理模型计算每个潜在点的“价值”，平衡探索（不确定性高的点）和利用（预测性能好的点）。
- **迭代过程**：
  1. 初始化：随机采样几个超参数点，评估目标函数（如模型准确率）。
  2. 更新模型：用新数据更新代理模型。
  3. 选择下一步：用采集函数选出下一个超参数点。
  4. 重复直到收敛或达到迭代上限。
- **优势**：在高维空间中更高效，通常只需少量评估即可找到近似最优解；适用于连续或离散超参数。
- **缺点**：代理模型的计算开销在极高维时可能增加；需要定义合理的超参数边界。

贝叶斯优化广泛应用于XGBoost、神经网络等模型的调优，能显著缩短搜索时间并提升性能。

---

### Python代码示例

以下是一个使用Python实现的贝叶斯优化超参数搜索示例，基于`skopt`库（Scikit-Optimize）的`BayesSearchCV`。示例针对XGBoost分类器在手写数字数据集（digits）上调优超参数，如学习率、最大深度等。注意：运行此代码需安装`skopt`和`xgboost`（`pip install scikit-optimize xgboost`），并确保有`numpy`、`sklearn`等基础库。

```python
import numpy as np
from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split

# 步骤1: 加载数据集并拆分
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤2: 定义超参数搜索空间
# 使用元组定义范围：(min, max, '分布类型')，如均匀或对数均匀
param_space = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),  # 学习率，对数均匀分布
    'max_depth': (1, 50),                         # 最大深度，整数均匀分布
    'gamma': (1e-9, 0.6, 'log-uniform'),          # 正则化参数
    'n_estimators': (50, 100),                    # 树的数量
    'degree': (1, 6),                             # 核函数度数（如果适用）
    'kernel': ['linear', 'rbf', 'poly']           # 离散类别参数
}

# 步骤3: 初始化贝叶斯优化器
# 使用BayesSearchCV，指定模型、搜索空间、评分指标、交叉验证和迭代次数
optimizer = BayesSearchCV(
    estimator=XGBClassifier(n_jobs=1),  # XGBoost分类器
    search_spaces=param_space,         # 搜索空间
    scoring='accuracy',                # 评估指标：准确率
    cv=3,                              # 3折交叉验证
    n_iter=50,                         # 总迭代次数（评估50个点）
    random_state=42                    # 随机种子，确保可复现
)

# 步骤4: 拟合模型并搜索最优超参数
optimizer.fit(X_train, y_train)

# 步骤5: 输出结果
best_params = optimizer.best_params_
best_score = optimizer.best_score_
print(f"最佳超参数: {best_params}")
print(f"最佳准确率: {best_score:.4f}")

# 可选: 在测试集上评估
test_score = optimizer.score(X_test, y_test)
print(f"测试集准确率: {test_score:.4f}")
```

#### 代码说明
1. **数据准备**：加载sklearn的digits数据集，拆分为训练/测试集。
2. **搜索空间**：定义超参数的范围和分布类型（连续、离散）。
3. **优化器初始化**：`BayesSearchCV`封装了贝叶斯优化过程，使用高斯过程作为代理模型。
4. **拟合与结果**：运行优化，自动迭代评估模型性能。`n_iter=50`表示评估50个超参数组合。
5. **输出**：打印最佳超参数和分数。实际运行中，最佳准确率可能达到0.98以上。

这个示例展示了贝叶斯优化在实际超参数调优中的应用。
