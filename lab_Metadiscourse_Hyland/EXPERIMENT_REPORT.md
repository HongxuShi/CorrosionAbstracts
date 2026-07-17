# 元话语特征分析验证实验报告

## Metadiscourse Feature Analysis — Experiment Report

---

**实验日期**：2026-07-17  
**实验目的**：验证Hyland(2005)元话语模型是否比Biber MD分析更适合描述腐蚀科学论文摘要的语言变异  
**数据集**：5,792篇腐蚀科学英文摘要（6个期刊，去除NCC极端值后）  
**对比基准**：lab_MD_Analysis_biber（15个Biber特征 + 4个PCA维度）  
**结论**：元话语方案在期刊区分上显著优于Biber，但对引用预测和影响力分类同样无效

---

## 1. 实验背景

Biber MD分析实验（lab_MD_Analysis_biber）的结论是：15个Biber风格特征+4个PCA维度对学术摘要内部差异的解释力不足。根因在于Biber的维度体系设计目标是区分口语vs书面语等宏观语域差异，不适合学术摘要的细粒度分析。

元话语(Metadiscourse)框架由Hyland(2005)提出，专为学术写作分析设计。其10个特征类别直接对应作者在文本中的策略选择，理论上比Biber的POS/句法特征更贴近学术摘要的分析需求。

参考：Hyland, K. (2005). *Metadiscourse: Exploring Interaction in Writing*. Continuum.

---

## 2. 技术架构

```
Abstract Corpus (5821篇)
        |
        v
Metadiscourse Lexicon Matching (10类词典, ~400词条)
（不需要spaCy NLP管道 — 仅需tokenization + 词边界匹配）
        |
        v
10-D Feature Matrix (n × 10, 无需PCA降维)
        |
        +---> 实验A: Features vs NCC (Regression)
        +---> 实验B: High/Low Impact Classification
        +---> 实验C: Journal Style Classification
        |
        v
Biber vs Metadiscourse 直接对比
```

### 2.1 10类元话语特征

| 大类 | 特征 | 词典规模 | 示例 |
|------|------|---------|------|
| **Interactive** | transition_ratio | 60词条 | however, therefore, in addition |
| （引导元话语） | frame_marker_ratio | 45词条 | first, this paper aims to, in summary |
| | endophoric_ratio | 35词条 | as shown in Fig., see Table, above |
| | evidential_ratio | 40词条 | according to, previous studies, it has been shown |
| | code_gloss_ratio | 35词条 | namely, such as, in other words |
| **Interactional** | hedge_ratio | 55词条 | may, possibly, approximately, suggest |
| （互动元话语） | booster_ratio | 50词条 | clearly, demonstrate, it is evident that |
| | attitude_ratio | 55词条 | importantly, crucial, surprisingly |
| | self_mention_ratio | 30词条 | we, our study, this paper, the author |
| | engagement_ratio | 40词条 | note that, should, consider, must |

### 2.2 元话语使用频率排名

| 排名 | 特征 | 平均密度 | 含义 |
|------|------|---------|------|
| 1 | transition_ratio | 0.0127 | 逻辑过渡是腐蚀科学摘要中最常见的元话语策略 |
| 2 | hedge_ratio | 0.0070 | 适度的模糊限制（学术写作的谨慎传统） |
| 3 | frame_marker_ratio | 0.0070 | 文本结构标记与hedge同等频率 |
| 4 | booster_ratio | 0.0055 | 增强语次之 |
| 5 | self_mention_ratio | 0.0051 | 自我指称密度中等 |

---

## 3. 实验A：元话语特征 → NCC（回归）

### 3.1 方法
同Biber实验的9种分析方法：Pearson/Spearman/OLS/Ridge/Lasso/RF/分位数Cohen's d

### 3.2 结果

| 指标 | 值 |
|------|-----|
| OLS R^2 | 0.0077 |
| Adjusted R^2 | 0.0059 |
| Ridge CV R^2 | 0.0025 ± 0.0035 |
| Lasso CV R^2 | 0.0025 ± 0.0032 |
| RF CV R^2 | 0.0034 ± 0.0064 |
| 最强Pearson |r| | frame_marker_ratio r=+0.0510 |

### 3.3 与Biber对比

| 指标 | Biber | Metadiscourse | 优胜 |
|------|-------|--------------|------|
| OLS R^2 | 0.0063 | **0.0077** | Meta (+22%) |
| 最强单特征 |r| | 0.059 (Dim1) | 0.051 (frame_marker) | Biber |

### 3.4 结论

**元话语特征同样无法预测论文引用数。** 虽然R^2比Biber高22%，但绝对值仍<1%——10个特征仅解释了不到1%的NCC方差。两个方案在此任务上都不及格。

---

## 4. 实验B：元话语特征 → 高/低影响力分类

### 4.1 方法
3种分组策略（Top25vsBot25, Top20vsBot20, Top10vsBot10）× Random Forest × 5-fold Stratified CV

### 4.2 结果

| 策略 | AUC | F1 | Baseline |
|------|-----|-----|----------|
| S1 Top25vsBot25 | 0.550 ± 0.030 | — | 0.500 |
| S2 Top20vsBot20 | 0.586 ± 0.024 | — | 0.501 |
| S4 Top10vsBot10 | 0.592 ± 0.024 | — | 0.584 |

### 4.3 与Biber对比

| 指标 | Biber | Metadiscourse | 优胜 |
|------|-------|--------------|------|
| 最佳AUC | **0.604** | 0.592 | Biber |

### 4.4 结论

**元话语特征的分类能力略逊于Biber。** 两者都达不到实用水平（AUC < 0.65），再次确认了"语言特征无法有效区分高/低被引论文"的结论。

---

## 5. 实验C：元话语特征 → 期刊分类

### 5.1 方法

7层分析同Biber实验：ANOVA、逐对Cohen's d、多分类(6期刊)、One-vs-Rest AUC、混淆矩阵、期刊风格剖面、马氏距离

### 5.2 结果

**多分类**：Random Forest Accuracy = 0.409 (baseline = 0.313)，F1_macro = 0.259

**One-vs-Rest AUC**：

| 期刊 | Metadiscourse AUC | Biber AUC | 提升 |
|------|:---:|:---:|:---:|
| Anti-Corr.MM. | **0.874** | 0.751 | **+12.3** |
| CORROSION | 0.669 | 0.587 | +8.2 |
| Corr.Eng.Sci. | 0.637 | 0.554 | +8.3 |
| Corr.Sci. | 0.634 | 0.604 | +3.0 |
| Corr.Mat.Deg. | 0.618 | 0.578 | +4.0 |
| Mat.Corros. | 0.612 | 0.569 | +4.3 |

**逐对期刊效应量**：

| 期刊对 | 最强区分特征 | Cohen's d | 效应等级 |
|--------|------------|:---:|------|
| Anti-Corr.MM. vs Mat.Corros. | self_mention_ratio | 0.905 | 大效应 |
| Anti-Corr.MM. vs CORROSION | frame_marker_ratio | 0.862 | 大效应 |
| Anti-Corr.MM. vs Corr.Sci. | frame_marker_ratio | 0.843 | 大效应 |
| Anti-Corr.MM. vs Corr.Eng.Sci. | frame_marker_ratio | 0.778 | 中大效应 |
| Anti-Corr.MM. vs Corr.Mat.Deg. | self_mention_ratio | 0.694 | 中效应 |
| Corr.Mat.Deg. vs Mat.Corros. | code_gloss_ratio | 0.356 | 小效应 |

### 5.3 期刊元话语风格指纹

每个期刊在10个特征上相对于总均值的标准化偏差（z-score）：

```
                    frame  self   trans  hedge  code   engage attitude booster evidential endoph
Anti-Corr.MM.      +0.85  +0.60  -0.26  -0.14  -0.11  -0.18   -0.04   -0.09    -0.06     -0.08
CORROSION           -0.12  -0.14  +0.09  +0.09  +0.07  +0.10   +0.04   -0.06    +0.09     +0.07
Corr.Eng.Sci.       -0.06  -0.06  -0.04  -0.00  -0.01  -0.05   +0.03   +0.03    +0.01     -0.01
Corr.Mat.Deg.       +0.07  +0.15  +0.29  +0.20  +0.36  +0.36   +0.31   +0.16    +0.09     -0.03
Corr.Sci.           -0.27  -0.04  +0.05  -0.15  -0.09  +0.11   +0.03   +0.02    -0.12     +0.07
Mat.Corros.         -0.22  -0.18  +0.02  +0.03  -0.01  -0.05   -0.08   +0.04    +0.00     -0.02

Key: +0.85 = 明显高于均值； -0.27 = 明显低于均值
```

**解读**：
- **Anti-Corr.MM.**：唯一真正有独特风格的期刊。大量使用结构标记（"this paper aims to"）和自我指称（"we", "our study"），作者声音响亮、文本结构显性化。
- **Corr.Mat.Deg.**：唯一使用更多语码注释和态度标记的期刊——解释性、评价性较强。
- **CORROSION/Corr.Eng.Sci./Corr.Sci./Mat.Corros.**：四个期刊的风格高度重叠，彼此之间不可区分。

### 5.4 对Copilot的风格对标实例

以投稿Anti-Corr.MM.为例，当用户摘要与目标期刊风格存在偏差时：

| 特征 | 用户摘要值 | Anti-Corr.MM.典型值 | d | 建议 |
|------|-----------|-------------------|:---:|------|
| frame_marker_ratio | 0.003 | 0.013 | +1.07 | "增加显性结构标记，如'This paper aims to...'" |
| self_mention_ratio | 0.003 | 0.008 | +0.73 | "使用更多自我指称（we, our study）" |
| transition_ratio | 0.018 | 0.011 | -0.31 | "适当减少显性逻辑过渡词" |

### 5.5 与Biber对比

| 指标 | Biber | Metadiscourse | 提升幅度 |
|------|:---:|:---:|:---:|
| 多分类 Accuracy | 0.332 | **0.409** | +23% |
| Accuracy - Baseline | +0.019 | **+0.096** | **5倍** |
| F1_macro | 0.173 | **0.259** | +50% |
| 最佳 OvR AUC | 0.751 | **0.874** | +16% |
| 最强成对 d | 0.835 | **0.905** | +8% |

---

## 6. 综合结论

### 6.1 头对头对比

| 实验 | Biber胜 | Metadiscourse胜 | 实际意义 |
|------|:---:|:---:|------|
| 回归 (NCC) | | ✓ (+22%) | 两者都不及格 (R^2 < 0.01) |
| 分类 (高/低引用) | ✓ (-2%) | | 两者都不及格 (AUC < 0.65) |
| 期刊 (多分类) | | ✓ (5倍提升) | **有真实信号** |

### 6.2 核心发现

1. **元话语在期刊区分上全面优于Biber**：Accuracy提升5倍（+1.9% → +9.6%），所有6个期刊的OvR AUC均有提升，Anti-Corr.MM.的AUC从0.751跃升至0.874。

2. **回归和分类两个方案同样失败**：无论用什么特征工程方案，摘要的语言风格都无法有效预测或区分引用影响力。这不是特征工程的问题——引用行为可能主要由非语言因素决定。

3. **期刊风格的差异集中在特定特征上**：self_mention_ratio和frame_marker_ratio贡献了最大的期刊间差异（d值达0.7-0.9）。这两个特征都与作者在文本中的"存在感"有关。

4. **只有1/6的期刊有真正独特的元话语指纹**：Anti-Corr.MM.的风格在多个特征上偏离均值超过0.5σ，其余5个期刊要么风格中性（CORROSION, Corr.Eng.Sci.），要么仅在1-2个特征上有中等偏离。

### 6.3 局限

- 词典匹配可能遗漏上下文相关的元话语使用（如"I"在STEM中罕见但在其他领域常见）
- 10个特征仍不足以捕捉学术摘要的全部修辞复杂性
- 除Anti-Corr.MM.外，其余5个期刊的区分度有限

---

## 7. 文件清单

```
lab_Metadiscourse/
├── EXPERIMENT_REPORT.md
├── metadiscourse_extractor.py    # 元话语特征提取器
├── run_all_analyses.py           # 三实验统一分析脚本
├── md_dictionaries/              # 10类元话语词典 (~400词条)
│   ├── transitions.txt
│   ├── frame_markers.txt
│   ├── endophorics.txt
│   ├── evidentials.txt
│   ├── code_glosses.txt
│   ├── hedges.txt
│   ├── boosters.txt
│   ├── attitude_markers.txt
│   ├── self_mentions.txt
│   └── engagement_markers.txt
└── output/
    ├── feature_matrix.csv        # 特征矩阵 5792×10
    ├── results_summary.csv       # 关键指标汇总
    ├── fig_m1_feature_importance.png
    ├── fig_m2_journal_ovr_auc.png
    ├── fig_m3_confusion_matrix.png
    ├── fig_m4_journal_profiles.png
    └── fig_m5_biber_vs_meta.png
```

---

**实验完成。元话语方案在期刊分类上显著优于Biber，但对引用预测同样无效。下一方案：句法复杂度(L2SCA)。**

*本报告由实验脚本自动输出的数据汇总生成。*
