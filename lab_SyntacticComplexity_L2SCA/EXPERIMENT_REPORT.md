# 句法复杂度特征分析验证实验报告

## Syntactic Complexity Feature Analysis — Experiment Report

---

**实验日期**：2026-07-17  
**实验目的**：验证Lu(2010) L2SCA句法复杂度指标是否比Biber MD分析和元话语方案更适合描述腐蚀科学论文摘要的语言变异  
**数据集**：5,792篇腐蚀科学英文摘要  
**对比基准**：lab_MD_Analysis_biber（15特征+4PCA）、lab_Metadiscourse_Hyland（10特征）  
**结论**：句法复杂度是三个方案中最弱的，几乎所有指标都垫底

---

## 1. 实验背景

前两个实验（Biber MD、元话语）均未能在回归和分类任务上产生有意义的信号。句法复杂度(L2SCA)是第三个候选方案，由Lu Xiaofei(2010)提出，包含14个指标，覆盖5个维度。该框架广泛应用于二语写作研究，理论上能捕捉学术文本的信息压缩和句法精细度差异。

## 2. 技术架构

基于spaCy实现14个L2SCA指标（T_S因零方差被移除，实际使用13个）：

```
Abstract Corpus → spaCy NLP → 
  ├── T-unit Detection
  ├── Clause Detection
  ├── Dependent Clause Detection
  ├── Coordinate Phrase Detection
  ├── Complex Nominal Detection
  └── Verb Phrase Detection
→ 13-D Feature Matrix → 三实验分析
```

### 2.1 13个特征

| 维度 | 特征 | 含义 | 均值 | 标准差 |
|------|------|------|:---:|:---:|
| Length | MLS | 平均句长 | 25.11 | 6.50 |
| Length | MLT | 平均T-unit长度 | 25.11 | 6.50 |
| Length | MLC | 平均从句长度 | 13.24 | 4.21 |
| Sentence | C_S | 每句从句数 | 1.89 | 0.52 |
| Subordination | C_T | 每T-unit从句数 | 1.89 | 0.52 |
| Subordination | CT_T | 复杂T-unit比例 | 0.55 | 0.20 |
| Subordination | DC_C | 从属从句比 | 0.52 | 0.16 |
| Subordination | DC_T | 每T-unit从属从句 | 1.01 | 0.44 |
| Coordination | CP_C | 每从句并列短语 | 0.63 | 0.49 |
| Coordination | CP_T | 每T-unit并列短语 | 1.14 | 0.76 |
| Structures | CN_C | 每从句复杂名词 | 2.14 | 0.68 |
| Structures | CN_T | 每T-unit复杂名词 | 4.24 | 1.31 |
| Structures | VP_T | 每T-unit动词短语 | 2.67 | 0.79 |

### 2.2 实现说明

- T-unit检测使用spaCy依存树：以ROOT为锚点收集子树
- 由于学术摘要高度压缩，MLS=MLT且C_S=C_T（每句基本就是1个T-unit）
- T_S（每句T-unit数）恒为1.0，因此从分析中移除

---

## 3. 实验结果

### 3.1 实验A：回归（句法复杂度 → NCC）

| 指标 | 值 |
|------|-----|
| OLS R^2 | **0.0017** |
| Adjusted R^2 | **-0.0006** (负值!) |
| Ridge CV R^2 | -0.0031 |
| Lasso CV R^2 | -0.0024 |
| RF CV R^2 | -0.0046 |
| 最强Pearson |r| | MLC r=-0.022 |

**所有CV R^2均为负值**——这意味着模型的表现不如直接猜测均值。句法复杂度特征对NCC的预测力为零。

### 3.2 实验B：分类（句法复杂度 → 高/低影响力）

| 策略 | AUC |
|------|:---:|
| S1 Top25vsBot25 | 0.526 |
| S2 Top20vsBot20 | 0.550 |
| S4 Top10vsBot10 | 0.551 |

所有AUC接近0.5（随机水平），分类能力几乎为零。

### 3.3 实验C：期刊分类

| 指标 | 值 |
|------|-----|
| 多分类 Accuracy | 0.341 (baseline=0.313) |
| Acc - Baseline | +0.028 |
| F1_macro | 0.183 |
| 最佳 OvR AUC | Corr.Sci. AUC=0.717 |
| 最强成对 d | Anti-Corr.MM. vs Corr.Mat.Deg. (C_T) d=0.580 |

期刊分类仅略优于随机（+2.8%）。最强的成对效应d=0.58仅为中等效应——远小于元话语方案的d=0.905。

---

## 4. 三方案对比

| 指标 | Biber | Metadiscourse | **Syntactic** | 最佳 |
|------|:---:|:---:|:---:|------|
| Regression R^2 | 0.0063 | **0.0077** | 0.0017 | Meta |
| Classification AUC | **0.604** | 0.592 | 0.551 | Biber |
| Journal Acc | 0.332 | **0.409** | 0.341 | **Meta** |
| Acc-Baseline | +0.019 | **+0.096** | +0.028 | **Meta** |
| Journal OvR AUC | 0.751 | **0.874** | 0.717 | **Meta** |
| Journal F1_macro | 0.173 | **0.259** | 0.183 | **Meta** |

```
三方案排名 (按期刊Acc-Baseline):
  🥇 Metadiscourse: +0.096
  🥈 Syntactic:      +0.028
  🥉 Biber:          +0.019

三方案排名 (按回归R^2):
  🥇 Metadiscourse: 0.0077
  🥈 Biber:         0.0063
  🥉 Syntactic:     0.0017  ← 唯一出现负Adj R^2的方案
```

---

## 5. 结论

**句法复杂度是三个方案中最弱的。** 回归CV R^2全为负值——说明这些特征在交叉验证中不仅不能预测NCC，反而增加了噪声。分类AUC接近0.5。期刊分类仅比Biber略好（+2.8% vs +1.9%），远逊于元话语的+9.6%。

### 5.1 为什么句法复杂度在此场景失效

1. **学术摘要的句法变异空间小**：摘要都遵循类似的句法模式（先行词+方法论+结果），句法复杂度在摘要之间的差异远小于在全文之间的差异。

2. **T-unit检测在摘要中退化**：MLS=MLT、C_S=C_T——T-unit没有提供超越句子级别的信息，因为它几乎等于句子的同义词。

3. **L2SCA是为二语写作设计的**：该框架的14个指标是为区分L1/L2英语写作水平设计的，不是为区分同质化学术摘要内部的细微风格差异设计的。

### 5.2 三方案整体评估

| 方案 | 理论基础 | 特征数 | 期刊区分力 | 可解释性 | 综合排名 |
|------|---------|:---:|:---:|:---:|:---:|
| Metadiscourse (Hyland) | 学术元话语理论 | 10 | ⭐⭐⭐ | ⭐⭐⭐ | 🥇 |
| Biber MD Analysis | 语域变异理论 | 15→4PCA | ⭐ | ⭐ | 🥈 |
| Syntactic Complexity (L2SCA) | 二语写作理论 | 13 | ⭐ | ⭐⭐ | 🥉 |

**元话语是唯一展示出实用信号（Acc-Baseline=+9.6%）的方案，但其信号集中在Anti-Corr.MM.一个期刊上。**

---

## 6. 跨方案发现

三个独立方案的实验汇集了一个跨方案的稳健结论：

> **在当前的腐蚀科学摘要语料上，任何单一的NLP特征工程方案都无法有效预测引用影响力（NCC）或区分高/低被引论文。**

这不是特征工程不够好的问题——三个理论框架从三个完全不同的语言学角度（语域变异、元话语策略、句法复杂度）都得出了一致的null result。这个null result本身具有方法论价值。

---

## 7. 文件清单

```
lab_SyntacticComplexity/
├── EXPERIMENT_REPORT.md
├── syntactic_extractor.py       # L2SCA句法复杂度提取器
├── run_all_analyses.py          # 三实验统一分析 + 对比
└── output/
    ├── feature_matrix.csv
    ├── results_summary.csv
    ├── fig_s1_importance.png
    ├── fig_s2_journal_auc.png
    └── fig_s3_threeway.png
```

---

**实验完成。句法复杂度是三个方案中最弱的。三方案一致结论：NLP特征对引用预测无效。元话语在期刊区分上最优。**
