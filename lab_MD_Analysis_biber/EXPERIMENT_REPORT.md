# MD分析可行性验证实验报告

## Multi-Dimensional Analysis Feasibility Testing — Experiment Report

---

**实验日期**：2026-07-17  
**实验目的**：验证基于Biber(1988)风格的15个语言特征+4维度PCA框架能否有效分析腐蚀科学论文摘要  
**数据集**：5,788篇腐蚀科学英文摘要（6个期刊，2015-2024年）  
**结论**：当前框架对学术摘要内部差异的解释力不足，不建议继续优化

---

## 1. 实验背景

本项目（腐蚀领域学术摘要文体特征与影响力关联分析）处于第二阶段探索性研究。此前baseline阶段使用6个基础文体特征（ASL/MWL/LD/LC/JD/HD）未发现与NCC的线性关系。本次实验引入Biber(1988)多维分析框架，试图构建更高层次的语言功能维度，探索其与学术影响力的关联。

## 2. 技术架构

```
Abstract Corpus (5821篇)
        |
        v
spaCy NLP Pipeline (tokenization/POS/dependency/lemma)
        |
        v
15 Linguistic Features (POS统计 + 依存句法 + 词典匹配)
        |
        v
StandardScaler + PCA(4) + Promax Rotation
        |
        v
4 Dimension Scores per Abstract
        |
        +---> Experiment A: Dimensions vs NCC (Regression)
        +---> Experiment B: High/Low Impact Classification
        +---> Experiment C: Journal Style Classification
```

### 2.1 15个语言特征

| #   | 特征                       | 检测方式                      | 归属维度 |
| --- | ------------------------ | ------------------------- | ---- |
| 1   | past_tense_ratio         | POS: VBD统计                | Dim1 |
| 2   | present_tense_ratio      | POS: VBP+VBZ统计            | Dim1 |
| 3   | passive_ratio            | dep: auxpass统计            | Dim1 |
| 4   | stance_verb_ratio        | stance verb + that clause | Dim1 |
| 5   | mental_verb_ratio        | 词典匹配(45词)                 | Dim1 |
| 6   | modal_possibility_ratio  | 词典匹配(28词)                 | Dim2 |
| 7   | modal_prediction_ratio   | 词典匹配(28词)                 | Dim2 |
| 8   | relative_clause_ratio    | dep: relcl统计              | Dim2 |
| 9   | communication_verb_ratio | 词典匹配(47词)                 | Dim3 |
| 10  | suasive_verb_ratio       | 词典匹配(34词)                 | Dim3 |
| 11  | noun_modifier_ratio      | NOUN+NOUN序列检测             | Dim4 |
| 12  | abstract_noun_ratio      | 词典匹配(85词)                 | Dim4 |
| 13  | nominalization_ratio     | 后缀检测(-tion等)              | Dim4 |
| 14  | human_noun_ratio         | 词典匹配(60词)                 | 辅助   |
| 15  | word_length              | 平均字符长度                    | 辅助   |

### 2.2 PCA结果

| 维度          | 解释方差   | 累计     |
| ----------- | ------ | ------ |
| Dimension 1 | 15.00% | 15.00% |
| Dimension 2 | 12.19% | 27.19% |
| Dimension 3 | 9.47%  | 36.66% |
| Dimension 4 | 7.75%  | 44.41% |

- **累计解释方差 44.41%**，超过参考论文的38.143%
- **KMO = 0.517**，Bartlett p ≈ 0（极显著）
- KMO低于0.6表明因子结构偏弱——特征间相关性不够强

### 2.3 开发中修复的Bug

1. **spaCy `attribute_ruler`被误禁用**：导致`token.pos_`返回空字符串，`noun_modifier_ratio`始终为0。修复：从禁用列表中移除`attribute_ruler`。
2. **中文引号冲突**：`""`与Python字符串定界符冲突导致SyntaxError。修复：替换为`「」`。
3. **Windows GBK终端编码**：emoji和中文字符导致`UnicodeEncodeError`。修复：用ASCII字符和`PYTHONIOENCODING=utf-8`。

---

## 3. 实验A：维度得分 → NCC（回归分析）

### 3.1 问题
4个MD维度得分能否预测论文的规范化引用计数(NCC)？

### 3.2 方法覆盖（9种）

| 序号  | 方法                       | 目的        |
| --- | ------------------------ | --------- |
| 1   | 描述性统计                    | 分布形态确认    |
| 2   | Pearson/Spearman相关       | 线性/单调关系检测 |
| 3   | OLS线性回归                  | 基线预测力     |
| 4   | Ridge/Lasso回归(5-fold CV) | 正则化+特征选择  |
| 5   | 随机森林                     | 非线性关联     |
| 6   | 梯度提升(GBRT)               | 更强非线性拟合   |
| 7   | 分位数分析                    | 极端组对比     |
| 8   | K-Means聚类                | 语言风格群体差异  |
| 9   | 交互效应                     | 维度间协同作用   |

### 3.3 结果

| 方法 | 最佳指标 | 解读 |
|------|---------|------|
| Pearson相关 | Dim1 r=+0.059*** | 统计显著但效应极小 |
| OLS回归 | R^2=0.0063 | 4个维度仅解释0.6%的NCC方差 |
| Ridge CV | R^2=0.004 | 正则化后几乎无预测力 |
| Lasso CV | R^2=0.004 | 同上 |
| Random Forest CV | R^2=0.010 | 略好但仍<1% |
| GBRT CV | R^2=-0.007 | 过拟合（负R^2） |
| Q4 vs Q1 (Dim1) | Cohen's d=+0.196*** | 仅有的有意义效应 |
| K-Means (K=2) | ANOVA p=0.072 | 类间NCC差异不显著 |
| 交互效应 | 全部DeltaR^2<0.002 | 无意义的交互项 |

### 3.4 结论

**MD维度得分不能预测论文引用数。** 所有模型（线性/非线性/正则化）的R^2均接近零。这与baseline阶段用6个基础特征未发现线性关系的结论一致。

---

## 4. 实验B：维度得分 → 高/低影响力分类

### 4.1 问题
从回归改为分类——语言特征能否区分"高被引"和"低被引"论文？

### 4.2 5种分组策略

| 策略 | 高影响力定义 | 低影响力定义 | 样本量 |
|------|------------|------------|--------|
| S1 | Top 25% NCC | Bottom 25% NCC | 2,897 |
| S2 | Top 20% NCC | Bottom 20% NCC | 2,321 |
| S3 | Top 10% NCC | Bottom 50% NCC | 3,475 |
| S4 | Top 10% NCC | Bottom 10% NCC（极端组） | 1,391 |
| S5 | 有引用 | 零引用 | 5,788 |

### 4.3 3种分类器

- Logistic Regression（L2正则化）
- Random Forest（300 trees）
- Linear SVM（Calibrated）

### 4.4 结果

| 策略 | 最佳AUC | Baseline | 实际提升 | 判断 |
|------|---------|----------|---------|------|
| S1 Top25vsBot25 | 0.597 | 0.500 | +0.097 | 弱 |
| S2 Top20vsBot20 | 0.604 | 0.501 | +0.103 | 弱 |
| S3 Top10vsBot50 | 0.594 | 0.833 | **-0.239** | ❌ 模型全猜多数类 |
| S4 Top10vsBot10 | 0.632 | 0.584 | +0.048 | 弱 |
| S5 CitedvsUncited | 0.644 | 0.860 | **-0.216** | ❌ 不如盲猜 |

- S3(S3): 所有模型的F1=0.000——完全退化，只会猜"低影响力"
- 单特征最强效应：Dim1 (d=+0.251**，Top10% vs Bottom50%)
- 所有Cohen's d值在0.1-0.25范围——全部属于"小效应"

### 4.5 结论

**即使改为分类问题并采取极端组对比，信号依然太弱。** 最佳实用AUC仅0.604，且类不平衡策略(S3/S5)的"AUC虚高"是假象——模型实际预测能力不如直接猜多数类。对Copilot的启示：15个Biber特征+4个PCA维度不足以提供写作建议。

---

## 5. 实验C：维度得分 → 期刊分类

### 5.1 问题
MD维度得分能否区分不同期刊的写作风格？

### 5.2 数据集
6个腐蚀科学期刊，5,788篇论文：

| 期刊 | 缩写 | 论文数 |
|------|------|--------|
| Materials and Corrosion | Mat.Corros. | 1,809 |
| CORROSION | CORROSION | 1,262 |
| Anti-Corrosion Methods and Materials | Anti-Corr.MM. | 893 |
| Corrosion Engineering, Science and Technology | Corr.Eng.Sci. | 866 |
| Corrosion Science | Corr.Sci. | 688 |
| Corrosion and Materials Degradation | Corr.Mat.Deg. | 270 |

### 5.3 方法覆盖（7层分析）

1. ANOVA + eta^2（各维度的期刊间差异量）
2. 逐对Cohen's d（哪对期刊风格差异最大）
3. 多分类（6期刊，5-fold CV）
4. One-vs-Rest AUC（哪个期刊最独特）
5. 混淆矩阵（哪些期刊容易混淆）
6. 马氏距离（期刊风格相似度地图）
7. Per-journal LR系数（每个期刊的维度特征）

### 5.4 结果

**ANOVA**：所有4个维度均显著（p<0.001），但效应量很小：
- Dim1: eta^2=6.7%（最大）
- Dim2: eta^2=0.8%
- Dim3: eta^2=0.6%
- Dim4: eta^2=1.2%

**多分类**：RF Accuracy=0.332 vs baseline=0.313，F1_macro=0.173

**混淆矩阵揭示的真相**：

| 真实期刊 | → Anti-Corr.MM. | → 其他期刊 | → Mat.Corros. |
|---------|:---:|:---:|:---:|
| Anti-Corr.MM. | **40%** | 0% | 52% |
| CORROSION | 14% | 0% | 71% |
| Corr.Eng.Sci. | 15% | 0% | 75% |
| Corr.Sci. | 11% | 0% | 79% |
| Corr.Mat.Deg. | 13% | 0% | 75% |
| Mat.Corros. | 13% | 0% | **76%** |

**模型本质上只学会了2件事**：(1)是不是Anti-Corr.MM.？(2)不是？那全猜Mat.Corros.

**One-vs-Rest AUC**：

| 期刊 | AUC | 独特性 |
|------|-----|--------|
| **Anti-Corr.MM.** | **0.751** | 🔴 唯一可区分的期刊 |
| Corr.Sci. | 0.604 | 弱 |
| CORROSION | 0.587 | 弱 |
| Corr.Mat.Deg. | 0.578 | 弱 |
| Mat.Corros. | 0.569 | 弱 |
| Corr.Eng.Sci. | 0.554 | 几乎不可区分 |

**期刊风格地图**（马氏距离）：

```
最接近: Corr.Sci. ←→ Mat.Corros. (d=0.1174)  ← 几乎完全重叠
最遥远: Anti-Corr.MM. ←→ Corr.Sci. (d=0.8351) ← 大效应
```

- Anti-Corr.MM.在Dim1上偏离均值+0.59个标准差（更多过去时/被动/立场动词）
- Dim1是唯一的区分维度（RF重要性=0.346）

### 5.5 结论

**效果介于有信号和无信号之间。** 6个期刊中有5个几乎完全不可区分——它们的写作风格在当前的4个维度空间中高度重叠。仅Anti-Corr.MM.一个期刊展现出足够独特的语言特征(AUC=0.751)。这个结果说明"语言特征能检测期刊差异"的前提成立，但15特征+4维度的粒度不足以区分大多数腐蚀科学期刊。

---

## 6. 综合结论

### 6.1 三次实验对比

| 实验 | 问题 | 最佳指标 | 信号强度 | 判决 |
|------|------|---------|---------|------|
| A: 回归 | 维度→NCC | R^2=0.006 | 无 | ❌ 不可用 |
| B: 分类 | 维度→高/低引用 | AUC≈0.60 | 极弱 | ❌ 不可用 |
| C: 分类 | 维度→期刊 | 1/6期刊AUC=0.75 | 弱-中 | ⚠️ 部分可用 |

### 6.2 根本原因分析

**Biber的维度体系不是为这个场景设计的。** Biber(1988)的67个特征和6个维度是为了区分"口语vs书面语"、"叙事vs非叙事"等宏观语域差异。将这些特征压缩到15个并用于区分同一语域（学术摘要）内部的细微差异，本质上是用大锤做微雕——工具不对。

在此基础上再做PCA降维到4维，等于将本已粗粒度的信号进一步压缩。56%的语言变异未被4个维度捕获，而这些被丢弃的变异可能恰恰包含期刊间、影响力间的区分信息。

### 6.3 方法论收获

虽然实验结果是negative的，但方法论的探索是有价值的：

1. **多角度验证策略有效**：回归→分类→多分类三种框架相互印证，避免了单一方法的假阳性
2. **分类优于回归的判断被证伪**：即使改为极端组分类，信号依然弱
3. **KMO=0.52是一个预警信号**：如果在因子分析阶段就注意到KMO偏低，可以更早地对特征工程提出质疑

---

## 7. 技术债务与已知问题

1. **spaCy lemma警告**：禁用sentencizer后lemmatizer找不到POS标注（不影响结果但产生W108警告）
2. **factor_analyzer数值问题**：协方差矩阵接近奇异时KMO/Bartlett返回NaN（已通过过滤零方差特征解决）
3. **`noun_modifier_ratio`的POS检测依赖`attribute_ruler`**：如果禁用它则该特征失效
4. **Windows GBK终端**：所有非ASCII字符在控制台输出中乱码（HTML报告和图表不受影响）

---

## 8. 建议的后续方向

基于三次实验的失败模式分析，当前15特征+4维度框架不应继续优化。建议考虑以下替代路线（按copilot价值排序）：

| 优先级 | 路线 | 原因 |
|--------|------|------|
| 🥇 | 元话语特征(Hyland, 2005) | 有现成词典，10个可解释特征，直接对应写作建议 |
| 🥈 | 句法复杂度(L2SCA) | 有pyl2sca现成实现，14个指标，可转化指导 |
| 🥉 | 信息密度与精确性 | 简单有效，能生成实用写作建议 |
| 4 | 修辞Move分析(CARS/Swales) | 最有理论深度但需先做move标注 |
| 5 | SciBERT embeddings | 探索价值高但可解释性低 |

---

## 9. 文件清单

### 核心代码（`lab/md_analysis/`）

| 文件 | 功能 |
|------|------|
| `preprocessor.py` | spaCy文本预处理管道 |
| `feature_extractor.py` | 15个Biber风格语言特征提取 |
| `pca_analyzer.py` | StandardScaler + PCA + Promax Rotation |
| `suitability_checker.py` | KMO/Bartlett/样本量/低方差检测 |
| `report_generator.py` | 7章节HTML报告生成 |
| `pipeline.py` | 主流程编排 |
| `run_md_analysis.py` | CLI入口脚本 |

### 分析脚本（`lab/`）

| 文件 | 功能 |
|------|------|
| `dim_ncc_analysis.py` | 实验A: 维度→NCC回归（9种方法） |
| `highlow_classification.py` | 实验B: 高/低影响力分类（5策略×3模型） |
| `journal_classification.py` | 实验C: 期刊风格分类（7层分析） |

### 数据与词典（`lab/`）

| 路径 | 内容 |
|------|------|
| `md_dictionaries/*.txt` | 8个语言特征词典（358个词条） |
| `output/feature_matrix.csv` | 特征矩阵 5788×15 |
| `output/dimension_scores.csv` | 维度得分 5788×4 |
| `output/dim_ncc_analysis_results.csv` | 实验A完整数据 |
| `output/MD_analysis_report.html` | 完整MD分析HTML报告 |

### 图表（`lab/output/`）

| 文件 | 来源 | 内容 |
|------|------|------|
| `fig1_correlation_heatmap.png` | 实验A | 维度×NCC相关热力图 |
| `fig2_dim_vs_ncc_scatter.png` | 实验A | 各维度vs NCC散点+LOWESS |
| `fig3_feature_importance_comparison.png` | 实验A | OLS/RF/GBRT重要性对比 |
| `fig4_quartile_ncc.png` | 实验A | 四分位NCC均值对比 |
| `fig5_cluster_visualization.png` | 实验A | K-Means聚类可视化 |
| `fig6_model_comparison.png` | 实验A | 模型性能对比 |
| `fig7_ncc_distribution.png` | 实验A | NCC分布直方图 |
| `fig_class1_cohens_d_heatmap.png` | 实验B | 各策略×维度Cohen's d |
| `fig_class2_auc_comparison.png` | 实验B | 分类AUC对比 |
| `fig_class3_roc_curves.png` | 实验B | ROC曲线 |
| `fig_class4_feature_importance.png` | 实验B | 分类特征重要性 |
| `fig_class5_violin_distributions.png` | 实验B | 高/低组维度分布小提琴图 |
| `fig_j1_confusion_matrix.png` | 实验C | 6期刊混淆矩阵 |
| `fig_j2_ovr_auc.png` | 实验C | One-vs-Rest AUC |
| `fig_j3_journal_scatter.png` | 实验C | 期刊2D PCA散点 |
| `fig_j4_journal_profiles.png` | 实验C | 期刊风格剖面图 |
| `fig_j5_pairwise_effects.png` | 实验C | 逐对期刊Cohen's d |
| `fig_j6_feature_importance.png` | 实验C | 期刊分类特征重要性 |

---

**实验完成。当前框架验证结果为negative，后续工作应转向新的特征工程方案。**

*本报告由实验脚本自动输出的数据汇总生成。所有数值与图表均可通过运行对应脚本复现。*
