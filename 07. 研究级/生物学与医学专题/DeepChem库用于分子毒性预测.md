## DeepChem库用于分子毒性预测
以下是一个使用DeepChem库的真实数据例子。该例子基于Tox21数据集（Toxicology in the 21st Century），这是一个真实的公共数据库，包含约12,000种化合物的毒性测量数据（针对12个生物靶点），来源于2014年的Tox21 Data Challenge。 该数据集使用SMILES表示分子结构，常用于分子毒性预测的多任务分类问题。

这个例子演示了如何加载Tox21数据集、使用两种模型（GraphConvModel：基于图卷积的模型；RobustMultitaskClassifier：鲁棒多任务分类器）进行训练和评估。代码是可运行的（假设已安装DeepChem、TensorFlow等依赖包），来源于DeepChem的教程和文档。 我已将代码组合成完整脚本，并添加了注释。

### 完整代码例子
```python
import deepchem as dc
from deepchem.models import GraphConvModel, RobustMultitaskClassifier
from deepchem.metrics import roc_auc_score  # 使用ROC-AUC作为评估指标

# 第一部分：使用GraphConv featurizer加载Tox21数据集（图形表示）
tasks_graph, datasets_graph, transformers_graph = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset_graph, valid_dataset_graph, test_dataset_graph = datasets_graph

# 定义并训练GraphConvModel（图卷积模型，用于处理分子图结构）
gcn_model = GraphConvModel(n_tasks=len(tasks_graph), mode='classification', dropout=0.2)
gcn_model.fit(train_dataset_graph, nb_epoch=50)  # 训练50个epoch

# 评估GraphConvModel
train_score_gcn = gcn_model.evaluate(train_dataset_graph, [roc_auc_score], transformers_graph)
test_score_gcn = gcn_model.evaluate(test_dataset_graph, [roc_auc_score], transformers_graph)
print("GraphConvModel AUC-ROC metrics:")
print(f"Training set score: {train_score_gcn}")
print(f"Test set score: {test_score_gcn}")

# 第二部分：使用ECFP featurizer加载Tox21数据集（循环指纹表示，非图形模型）
tasks_ecfp, datasets_ecfp, transformers_ecfp = dc.molnet.load_tox21(featurizer='ECFP')
train_dataset_ecfp, valid_dataset_ecfp, test_dataset_ecfp = datasets_ecfp

# 定义并训练RobustMultitaskClassifier（鲁棒多任务分类器）
mtc_model = RobustMultitaskClassifier(n_tasks=len(tasks_ecfp), n_features=1024, layer_sizes=[1000], dropout=0.5)
mtc_model.fit(train_dataset_ecfp, nb_epoch=50)  # 训练50个epoch

# 评估RobustMultitaskClassifier
train_score_mtc = mtc_model.evaluate(train_dataset_ecfp, [roc_auc_score], transformers_ecfp)
test_score_mtc = mtc_model.evaluate(test_dataset_ecfp, [roc_auc_score], transformers_ecfp)
print("RobustMultitaskClassifier AUC-ROC metrics:")
print(f"Training set score: {train_score_mtc}")
print(f"Test set score: {test_score_mtc}")
```

### 代码说明
- **数据加载**：使用`dc.molnet.load_tox21()`直接从MoleculeNet加载Tox21数据集。`featurizer`参数指定分子特征化方式：'GraphConv'用于图卷积模型，'ECFP'（Extended-Connectivity Fingerprints）用于非图模型。数据集自动分割为训练、验证和测试集。
- **模型训练**：GraphConvModel适合处理分子图结构；RobustMultitaskClassifier用于多任务学习（12个毒性靶点）。训练使用50个epoch，可根据需要调整。
- **评估**：使用ROC-AUC分数评估模型性能，适用于二分类问题（毒性/非毒性）。
- **运行要求**：需安装DeepChem（pip install deepchem）。运行时会自动下载Tox21数据（如果未缓存）。预期输出包括训练和测试的AUC-ROC分数，例如训练集分数可能在0.9以上，测试集在0.7-0.8左右（取决于随机种子）。
- **应用**：这个例子可用于量子化学或药物发现中的分子毒性预测，帮助筛选潜在有害化合物。
