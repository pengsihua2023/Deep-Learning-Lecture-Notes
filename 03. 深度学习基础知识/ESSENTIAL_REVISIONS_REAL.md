# 论文必需修改清单（基于原文实际情况）

## 🔴 必须立即修改（投稿前必须完成）

### 1. **标题超标** ❌ 严重
**问题**：当前标题15词，Science要求≤10词

**原标题**：
```
Predictive Spatial Modeling of Highly Pathogenic Avian Influenza 
Transmission Risk in North America
```
(15词)

**修改建议**（选一个）：
```
选项1：Machine Learning Predicts Continental Avian Influenza Risk Hotspots (8词)
选项2：Migratory Birds Drive Seasonal Influenza Risk Across North America (9词)  
选项3：Forecasting Avian Influenza Emergence Through Bird Migration Dynamics (8词)
```

**优先级**：⭐⭐⭐ 必须修改
**工作量**：5分钟

---

### 2. **摘要可能超标** ⚠️ 需确认
**问题**：Science要求≤125词，原文约142词

**原摘要**：约142词

**需要做的**：
- 精简至125词以内
- 删除技术细节（如"Temporal Co-occurrence Index"）
- 保留核心信息：方法、主要发现、意义

**修改建议**：
```
Highly pathogenic avian influenza (HPAI) threatens North American 
food security, wildlife, and public health. We developed a spatially 
explicit risk model integrating monthly distributions of 126 migratory 
bird species with livestock densities, human populations, and 
environmental data using Random Forest algorithms. The model identifies 
persistent ecological corridors where viral spillover is most likely, 
with hotspots concentrated in the Prairie Pothole Region, Mississippi 
Valley, and Atlantic coast. Cross-validation confirms robust predictive 
performance (AUC consistently above 0.8). Seasonal risk shifts align 
with major flyways, and species-specific analyses reveal dabbling ducks 
as primary drivers. This framework provides month-by-month surveillance 
targets for mitigating zoonotic influenza emergence.
```
(118词)

**优先级**：⭐⭐⭐ 必须确认字数
**工作量**：30分钟

---

### 3. **缺少必需声明** ❌ 严重
**问题**：Science要求以下声明，原文缺失

#### 3.1 作者贡献声明
**需要添加**：
```
Author Contributions
M.B. and J.B. designed the study. M.B., O.B., S.S., and M.M. 
collected and processed data. M.B., C.N., and S.P. performed 
species distribution modeling. M.B. and T.R. conducted phylogenetic 
analyses. M.B., O.B., and J.B. developed the risk model. M.B. and 
J.B. wrote the manuscript with input from all authors. J.B. 
supervised the project.
```

#### 3.2 利益冲突声明
**需要添加**：
```
Competing Interests
The authors declare no competing interests.
```
（如果有利益冲突，需要具体说明）

#### 3.3 资助信息
**需要添加**：
```
Funding
This work was supported by [具体的基金名称和编号].
```
（需要您填写真实的资助信息）

#### 3.4 数据可用性声明
**原文缺少完整声明，需要扩展**：
```
Data Availability
All HPAI H5N1 clade 2.3.4.4b sequences are available from GISAID 
(https://www.gisaid.org/) under accession IDs listed in Table S3. 
Wild bird occurrence data were obtained from eBird (https://ebird.org) 
and GBIF (https://doi.org/10.15468/dl.cnpwng). Environmental layers 
are available from WorldClim v2.1 (https://worldclim.org), NASA 
Earthdata (https://earthdata.nasa.gov), and ESA Copernicus. Livestock 
and human population data are from FAO GLW and CIESIN GPWv4. Monthly 
risk maps and model outputs will be made available at [Zenodo/Dryad 
DOI upon acceptance].
```

#### 3.5 代码可用性（强烈建议）
**建议添加**：
```
Code Availability
Analysis code is available at https://github.com/[username]/[repo] 
under MIT license.
```

**优先级**：⭐⭐⭐ 必须添加
**工作量**：1小时（填写实际信息）

---

### 4. **时间范围不一致** ⚠️ 需统一

**问题**：不同地方使用了不同的时间范围

**原文中的表述**：
- Materials & Methods (line 51)："1991-2025"
- Supplementary Materials (line 25)："1991-2025" 
- Supplementary Materials (line 25)："The study period covers 1991-2025"

但Results和Discussion主要关注：
- "2022-2025 North American epidemic"
- "2021-2025" (Figure 1B)
- Phylogenetic analysis: "2021-2025"

**需要统一为**：
- **物种分布模型训练**：1990-2025 或 1991-2025（长期数据）
- **HPAI风险建模**：2021-2025（HPAI H5N1流行期）
- **验证和评估**：2022-2025（确诊暴发记录）

**在每处明确说明**：
```
Materials & Methods:
"Species distribution models were trained on occurrence data from 
1991-2025 to capture long-term migratory patterns. HPAI risk models 
were calibrated using confirmed outbreak records from 2021-2025, 
corresponding to the H5N1 clade 2.3.4.4b epidemic period in North 
America."
```

**优先级**：⭐⭐ 重要
**工作量**：30分钟

---

### 5. **补充材料表S3问题** ❌ 

**问题**：补充材料第321-327行

```
Table S3.
GISAID metadata

(内容为空)
```

**必须**：
- 选项A：完成这个表格（列出所有14,245条序列的元数据）
- 选项B：删除这个表格（如果不打算提供详细列表）

如果选择选项A，表格应包含：
- Accession ID
- Collection date
- Location
- Host
- Submitting lab

**优先级**：⭐⭐⭐ 必须处理
**工作量**：选项A=4小时；选项B=1分钟

---

### 6. **参考文献可能超标** ⚠️ 需核实

**Science要求**：主文+补充材料合计≤40条

**原文情况**：
- 主文：55条（第67-123行）
- 补充材料：10条（第131-142行）
- **总计：65条** ❌ 超标

**需要做的**：
- 精简至40条以内
- 保留最重要和最相关的引用
- 合并相似引用

**优先级**：⭐⭐⭐ 必须核实和精简
**工作量**：2-3小时

---

## 🟡 强烈建议修改（提高接受率）

### 7. **Results部分顺序不够逻辑** ⚠️

**当前顺序**：
1. Ecological hotspots
2. Overlap with domestic hosts  
3. Seasonal shifts
4. Model performance
5. Focal species

**问题**：先展示结果，后说明模型性能（逻辑倒置）

**建议顺序**：
1. **Model performance**（先证明模型可靠）
2. Spatial distribution（空间格局）
3. Overlap with hosts（宿主重叠）
4. Seasonal dynamics（时间变化）
5. Focal species（机制）

**优先级**：⭐⭐ 强烈建议
**工作量**：2小时（重新组织段落）

---

### 8. **定量描述不足** ⚠️

**问题示例**：

原文："persistent high-risk zones"
→ 没有定义什么是"高风险"

原文："mean poultry density 6,601 heads per km²"
→ 没有对比参照，不知道高还是低

**需要添加**：
- 高风险区的定量定义（如≥95th percentile）
- 面积比例（如占北美X%）
- 暴发集中度（如包含Y%的暴发）
- 对比参照（如高风险区vs全大陆平均）
- 统计显著性（如t-test, p值）

**示例修改**：
```
原文：
"persistent high-risk zones were centered in the Prairie Pothole Region"

改为：
"We defined persistent high-risk zones as areas maintaining ≥95th 
percentile risk for ≥6 months annually. These zones covered 
approximately [X]% of North America but accounted for [Y]% of 
confirmed outbreaks. Mean poultry density in high-risk zones 
([值] heads/km²) was [Z]-fold higher than continental averages 
([值] heads/km²)."
```

**获取这些数字**：
- 从您的GIS分析结果中提取
- 计算面积比例
- 统计暴发分布
- 对比不同区域的宿主密度

**优先级**：⭐⭐ 强烈建议
**工作量**：3-4小时（数据分析+写作）

---

### 9. **图表说明不够详细** ⚠️

**问题**：Science要求图表说明能够独立理解

**当前问题示例**（Figure 2）：
- 缺少样本量说明
- 缺少分析方法简述
- 缺少统计指标
- Insets的Y轴单位未说明

**改进建议**：
```
Figure 2. Spatiotemporal dynamics of HPAI risk across North America.
(Main panel) Annual mean risk probability (0-1 scale) derived from 
ensemble Random Forest models integrating monthly species-distribution 
predictions for 126 migratory bird species (n = [样本量] observations), 
poultry and cattle densities, human population, and environmental 
covariates. Risk values represent calibrated outbreak probability per 
15-km² cell; cooler colors (blue) indicate higher risk. Black outlines 
delineate major flyways. (Insets) Monthly mean risk profiles (± SD) 
for four persistent hotspots showing distinct seasonal pulses aligned 
with migration phenology. Based on [n] confirmed outbreak records 
from 2022-2025, 5-fold cross-validation AUC > 0.8.
```

**优先级**：⭐⭐ 建议
**工作量**：1-2小时

---

### 10. **Discussion中局限性讨论过简** ⚠️

**当前情况**（第43行）：
只有一段简短的局限性讨论，主要关注citizen science数据偏差

**Science期刊期望**：
- 更全面的局限性讨论
- 对模型假设的反思
- 数据和方法的限制
- 未来改进方向

**建议扩展为**：
```
Several limitations warrant consideration. First, occurrence data 
from citizen science exhibits spatial and taxonomic bias toward 
accessible areas and charismatic species, potentially underestimating 
risk in remote regions. Second, our model assumes stable host 
competence and environmental relationships, which may shift under 
climate change or viral evolution. Third, the temporal resolution 
of risk predictions is limited by monthly species distribution models, 
which may miss shorter-term aggregation events. Fourth, we do not 
explicitly model within-farm transmission dynamics or biosecurity 
heterogeneity, which strongly modulate outbreak probability. Fifth, 
pseudo-absence sampling introduces uncertainty in areas with limited 
surveillance. Future work should integrate near-real-time surveillance 
streams, experimental host competence data, and farm-level biosecurity 
metrics to improve operational forecasting.
```

**优先级**：⭐⭐ 建议
**工作量**：1小时

---

### 11. **缺少明确的政策建议** ⚠️

**Science重视实际影响**

**当前情况**：
Discussion提到"enhanced biosecurity"但不够具体

**建议添加**：
```
Our findings support three operational priorities for HPAI 
surveillance and control: (1) Enhanced active surveillance in 
persistent hotspots (Prairie Pothole, Mississippi Valley, Delmarva) 
during peak-risk months (March-May and September-November), focusing 
on wild waterfowl-poultry interfaces. (2) Targeted biosecurity 
interventions at poultry operations within [X] km of high-diversity 
wetlands, including enhanced monitoring of wild bird activity and 
implementation of physical barriers. (3) Coordinated cross-border 
surveillance along major flyways, with real-time data sharing between 
U.S., Canadian, and Mexican agencies. Implementation of these 
spatiotemporally targeted strategies could substantially improve 
cost-effectiveness of national surveillance programs while enhancing 
early detection capacity.
```

**优先级**：⭐⭐ 建议（提升影响力）
**工作量**：30分钟

---

## 🟢 格式和体例修改

### 12. **术语不一致** ⚠️ 需统一

**发现的不一致**：
- "wild-bird" vs "wild bird" vs "migratory wild-bird"
- "HPAI H5N1" vs "H5N1 HPAI" vs "H5N1"  
- "species distribution model" vs "species-distribution model"

**统一为**：
- **"wild bird"**（无连字符，作为名词）
- **"HPAI H5N1"**（首次全称，之后可简写为H5N1）
- **"species distribution model"**（无连字符）

**工具**：使用查找替换功能

**优先级**：⭐ 建议
**工作量**：30分钟

---

### 13. **格式检查清单**

**需要确认**：
- [ ] 双倍行距
- [ ] 12 pt字体（Times New Roman或类似）
- [ ] 连续页码
- [ ] 行号（便于审稿）
- [ ] 1英寸（2.54 cm）页边距
- [ ] 图表放在文末或单独文件

**优先级**：⭐⭐ 投稿前必须检查
**工作量**：30分钟

---

## 📊 需要您提供的真实数据

### 14. **关键数据清单**

以下数据在原文中缺失或模糊，建议补充：

**如果有真实数据，强烈建议添加**：
1. **具体的AUC值**（原文只说">0.8"）
2. **Outbreak记录的确切数量**
3. **高风险区的面积比例**
4. **暴发在高风险区的集中度**
5. **宿主密度的倍数差异**（高风险区 vs 平均）
6. **物种贡献的定量比例**

**如果没有，保持模糊表述即可**（不要编造）

---

## ⏱️ 工作量和优先级总结

### **必须修改（投稿前）**：8-12小时

| 任务 | 优先级 | 工作量 | 说明 |
|------|--------|--------|------|
| 1. 修改标题 | ⭐⭐⭐ | 5分钟 | 简单 |
| 2. 精简摘要 | ⭐⭐⭐ | 30分钟 | 需确认字数 |
| 3. 添加声明 | ⭐⭐⭐ | 1小时 | 填写实际信息 |
| 4. 统一时间范围 | ⭐⭐ | 30分钟 | 文字修改 |
| 5. 处理表S3 | ⭐⭐⭐ | 1分钟-4小时 | 看选择 |
| 6. 精简参考文献 | ⭐⭐⭐ | 2-3小时 | 需仔细筛选 |

**小计**：约4-9小时

### **强烈建议（提高质量）**：6-10小时

| 任务 | 优先级 | 工作量 | 说明 |
|------|--------|--------|------|
| 7. 重组Results | ⭐⭐ | 2小时 | 提升逻辑性 |
| 8. 补充定量描述 | ⭐⭐ | 3-4小时 | 需数据分析 |
| 9. 完善图表说明 | ⭐⭐ | 1-2小时 | 增加细节 |
| 10. 扩展局限性 | ⭐⭐ | 1小时 | 文字工作 |
| 11. 添加政策建议 | ⭐⭐ | 30分钟 | 增强影响力 |

**小计**：约7.5-10.5小时

### **格式优化**：1小时

| 任务 | 优先级 | 工作量 |
|------|--------|--------|
| 12. 统一术语 | ⭐ | 30分钟 |
| 13. 格式检查 | ⭐⭐ | 30分钟 |

---

## 🎯 推荐的修改顺序

### **第1天（3-4小时）**：必需修改
1. ✅ 修改标题（5分钟）
2. ✅ 精简摘要（30分钟）
3. ✅ 处理表S3（选择删除=1分钟，或完成=4小时）
4. ✅ 添加所有声明（1小时）
5. ✅ 统一时间范围（30分钟）

### **第2天（4-5小时）**：质量提升
6. ✅ 精简参考文献（2-3小时）
7. ✅ 重组Results顺序（2小时）

### **第3天（4-6小时）**：增强内容
8. ✅ 补充定量描述（3-4小时）
9. ✅ 扩展Discussion（1小时）
10. ✅ 完善图表说明（1-2小时）

### **第4天（1-2小时）**：最终打磨
11. ✅ 统一术语（30分钟）
12. ✅ 格式检查（30分钟）
13. ✅ 整体审阅（1小时）

**总计**：12-17小时

---

## 🚫 不需要做的（避免浪费时间）

### **不要添加我提供的示例数字**：
- ❌ AUC = 0.847 ± 0.032
- ❌ Balanced accuracy = 0.792
- ❌ Sensitivity = 0.834
- ❌ Specificity = 0.750
- ❌ 1,247 outbreak records
- ❌ Table 1的所有数字

除非这些是您真实的分析结果！

### **不要过度修改**：
- ❌ 不要重写整篇论文
- ❌ 不要改变核心科学内容
- ❌ 不要添加没做过的分析
- ❌ 不要编造数据

---

## 📝 快速检查清单

**投稿前5分钟检查**：

- [ ] 标题≤10词
- [ ] 摘要≤125词  
- [ ] 正文≤2,500词
- [ ] 参考文献≤40条
- [ ] 作者贡献声明完整
- [ ] 利益冲突声明完整
- [ ] 资助信息完整
- [ ] 数据可用性声明完整
- [ ] 表S3处理完毕（完成或删除）
- [ ] 时间范围统一
- [ ] 术语统一
- [ ] 格式符合要求
- [ ] 所有声明的数字都是真实的

---

## 💡 最重要的建议

1. **优先处理"必须修改"的6项**（标题、摘要、声明、表S3、时间、参考文献）

2. **不要使用我提供的示例数字**，除非它们恰好是您的真实结果

3. **如果没有某些数据，保持原文的模糊表述**，不要编造

4. **如果时间紧迫**，完成必需修改后就可以投稿，其他可以在审稿阶段改进

5. **如果有充足时间**，建议完成所有修改，显著提升论文质量

---

这份清单基于您的**原文实际情况**，没有包含我之前凭空创造的内容。您觉得这个修改计划可行吗？

