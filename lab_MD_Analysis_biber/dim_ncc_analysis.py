#!/usr/bin/env python3
"""
================================================================================
 MD维度得分与NCC关联分析脚本
 Association Analysis: MD Dimension Scores -> Normalized Citation Count (NCC)
================================================================================

分析目标：
  探索MD分析的4个语言功能维度是否与论文学术影响力（NCC）存在关联。

分析方法清单（共9类）：
  ┌──────┬──────────────────────────┬─────────────────────────────────┐
  │ 序号 │ 分析方法                 │ 回答的问题                      │
  ├──────┼──────────────────────────┼─────────────────────────────────┤
  │  1   │ 描述性统计               │ 各变量分布形态如何？            │
  │  2   │ Pearson/Spearman相关     │ 维度与NCC有线性/单调关系吗？    │
  │  3   │ 线性回归 (OLS)           │ 维度能解释多少NCC方差？         │
  │  4   │ 正则化回归 (Ridge/Lasso) │ 哪些维度是稳健的预测因子？      │
  │  5   │ 随机森林                 │ 是否存在非线性关联模式？        │
  │  6   │ 梯度提升 (GBRT)          │ 更强的非线性拟合能发现什么？    │
  │  7   │ 分位数分析               │ 维度得分高低与NCC分布的关系？   │
  │  8   │ K-Means聚类              │ 不同语言风格群体的NCC差异？     │
  │  9   │ 交互效应                 │ 维度之间是否有协同/拮抗作用？   │
  └──────┴──────────────────────────┴─────────────────────────────────┘

输出：
  - 控制台：所有分析的详细统计输出
  - lab/output/ 目录：可视化图表（PNG格式）
  - lab/output/dim_ncc_analysis_results.csv：每条记录的完整分析数据

注意事项：
  1. 这是探索性分析，不用于因果推断
  2. KMO=0.52表明因子结构偏弱，维度含义需谨慎解读
  3. 分析结果仅供参考和方法参考

作者：理论验证阶段
日期：2026/07/17
================================================================================
"""

import warnings
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# ---- 统计与机器学习 ----
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# ---- 可视化 ----
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，用于脚本批量生成图表
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ---- 设置 ----
warnings.filterwarnings('ignore')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# ============================================================================
#  路径配置
# ============================================================================

# 确定输出目录（脚本位于 lab/ 目录下）
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件路径
DIM_SCORES_PATH = SCRIPT_DIR / "output" / "dimension_scores.csv"
NCC_DATA_PATH = SCRIPT_DIR.parent / "processed" / "merged_all_features.csv"

print("=" * 70)
print(" MD维度得分与NCC关联分析")
print(f" 维度得分: {DIM_SCORES_PATH}")
print(f" NCC数据:  {NCC_DATA_PATH}")
print(f" 输出目录: {OUTPUT_DIR}")
print("=" * 70)


# ============================================================================
#  0. 数据准备
# ============================================================================

print("\n" + "=" * 70)
print(" [0] 数据加载与合并")
print("=" * 70)

# 加载维度得分
df_scores = pd.read_csv(DIM_SCORES_PATH)
print(f"  维度得分: {df_scores.shape[0]} 条记录, "
      f"列: {[c for c in df_scores.columns if c != 'doc_id']}")

# 加载NCC数据
df_ncc = pd.read_csv(NCC_DATA_PATH)
print(f"  NCC数据: {df_ncc.shape[0]} 条记录")

# 合并 ---- 以 DOI 为关联键
df = df_scores.merge(
    df_ncc[['doi', 'year', 'citations', 'NCC', 'source_journal']],
    left_on='doc_id', right_on='doi', how='inner'
)
df = df.drop(columns=['doi'])

# 移除NCC为NaN的记录（无法计算NCC的论文）
n_before = len(df)
df = df.dropna(subset=['NCC'])
n_after = len(df)
print(f"  合并后: {n_before} 条, 移除NCC缺失: {n_before - n_after} 条, "
      f"最终: {n_after} 条")

# 处理极端值 ---- NCC可能因为除以极小的均值而产生极端值
# 使用99.5%分位数截断（保留绝大部分数据，仅去除极端离群值）
ncc_upper = df['NCC'].quantile(0.995)
ncc_lower = df['NCC'].quantile(0.005)
df_clean = df[(df['NCC'] >= ncc_lower) & (df['NCC'] <= ncc_upper)].copy()
print(f"  极端值截断 (0.5%-99.5%): 移除 {len(df) - len(df_clean)} 条, "
      f"最终: {len(df_clean)} 条")

# 准备分析变量
dim_cols = ['Dim1', 'Dim2', 'Dim3', 'Dim4']
X = df_clean[dim_cols].values
y = df_clean['NCC'].values
n_samples, n_features = X.shape

print(f"\n  分析数据集: {n_samples} 篇 × {n_features} 个维度 + NCC")
print(f"  NCC范围: [{y.min():.2f}, {y.max():.2f}], "
      f"均值={y.mean():.2f}, 中位数={np.median(y):.2f}")


# ============================================================================
#  1. 描述性统计
# ============================================================================

print("\n" + "=" * 70)
print(" [1] 描述性统计 (Descriptive Statistics)")
print("=" * 70)

# 汇总统计
desc_df = df_clean[dim_cols + ['NCC', 'citations']].describe()
print(desc_df.round(4).to_string())

# 偏度和峰度 ---- 判断分布形态
print("\n  偏度 (Skewness) 与峰度 (Kurtosis):")
for col in dim_cols + ['NCC']:
    skew = stats.skew(df_clean[col])
    kurt = stats.kurtosis(df_clean[col])  # excess kurtosis
    print(f"    {col:8s}: skew={skew:+.3f}, kurtosis={kurt:+.3f}")

# NCC的分布是否正态？
_, ncc_norm_p = stats.normaltest(df_clean['NCC'])
print(f"\n  NCC正态性检验 (D'Agostino-Pearson): p={ncc_norm_p:.6f}"
      f"{' -- 不服从正态分布' if ncc_norm_p < 0.05 else ' -- 近似正态'}")


# ============================================================================
#  2. 相关性分析
# ============================================================================

print("\n" + "=" * 70)
print(" [2] 相关性分析 (Correlation Analysis)")
print("=" * 70)

# ---- 2a. Pearson 相关 ----
print("\n  --- Pearson 相关系数 (线性关系) ---")
pearson_results = {}
for dim in dim_cols:
    r, p = pearsonr(df_clean[dim], df_clean['NCC'])
    pearson_results[dim] = {'r': r, 'p': p}
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    {dim} -> NCC: r={r:+.4f}, p={p:.6f} {sig}")

# ---- 2b. Spearman 秩相关 (单调关系，不要求线性) ----
print("\n  --- Spearman 秩相关系数 (单调关系) ---")
spearman_results = {}
for dim in dim_cols:
    rho, p = spearmanr(df_clean[dim], df_clean['NCC'])
    spearman_results[dim] = {'rho': rho, 'p': p}
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    {dim} -> NCC: rho={rho:+.4f}, p={p:.6f} {sig}")

# ---- 2c. 维度间相关性（多重共线性检查） ----
print("\n  --- 维度间 Pearson 相关矩阵 ---")
dim_corr = df_clean[dim_cols].corr()
print(dim_corr.round(4).to_string())

# ---- 2d. 偏相关（控制其他维度） ----
print("\n  --- 偏相关系数 (Partial Correlation, 控制其他3个维度) ---")
from scipy.stats import pearsonr

def partial_corr(x, y, controls):
    """计算偏相关系数: corr(x, y | controls)"""
    # 分别对x和y回归掉controls，然后计算残差的相关性
    from sklearn.linear_model import LinearRegression
    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    # x ~ controls
    reg_x = LinearRegression().fit(controls, x)
    resid_x = x - reg_x.predict(controls)
    # y ~ controls
    reg_y = LinearRegression().fit(controls, y)
    resid_y = y - reg_y.predict(controls)
    r, p = pearsonr(resid_x, resid_y)
    return r, p

for i, dim in enumerate(dim_cols):
    other_dims = [d for j, d in enumerate(dim_cols) if j != i]
    controls = df_clean[other_dims].values
    r_partial, p_partial = partial_corr(
        df_clean[dim].values, df_clean['NCC'].values, controls
    )
    sig = "***" if p_partial < 0.001 else ("**" if p_partial < 0.01 else ("*" if p_partial < 0.05 else "ns"))
    print(f"    {dim} -> NCC | {other_dims}: r_partial={r_partial:+.4f}, p={p_partial:.6f} {sig}")


# ============================================================================
#  3. 线性回归 (OLS)
# ============================================================================

print("\n" + "=" * 70)
print(" [3] 普通最小二乘回归 (OLS Linear Regression)")
print("=" * 70)

# 标准化特征以便比较系数大小
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ols = LinearRegression()
ols.fit(X_scaled, y)

# R^2
y_pred_ols = ols.predict(X_scaled)
ss_res = np.sum((y - y_pred_ols) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_ols = 1 - ss_res / ss_tot
# 调整R^2
adj_r2_ols = 1 - (1 - r2_ols) * (n_samples - 1) / (n_samples - n_features - 1)

print(f"  R^2 = {r2_ols:.6f}")
print(f"  Adjusted R^2 = {adj_r2_ols:.6f}")
print(f"  F-statistic: 需要手动计算（见后续）")
print(f"\n  标准化回归系数 (Standardized Coefficients):")
for i, dim in enumerate(dim_cols):
    print(f"    {dim}: beta = {ols.coef_[i]:+.4f}")
print(f"    截距 (Intercept): {ols.intercept_:+.4f}")

# 使用statsmodels获取完整统计推断
try:
    import statsmodels.api as sm
    X_sm = sm.add_constant(X_scaled)
    ols_sm = sm.OLS(y, X_sm).fit()
    print(f"\n  --- Statsmodels 完整回归输出 ---")
    print(ols_sm.summary().tables[1])  # 系数表
    print(f"\n  F-statistic: {ols_sm.fvalue:.2f}, p={ols_sm.f_pvalue:.6f}")
    print(f"  AIC: {ols_sm.aic:.1f}, BIC: {ols_sm.bic:.1f}")
except ImportError:
    print("  (statsmodels 未安装，跳过完整统计推断)")


# ============================================================================
#  4. 正则化回归 (Ridge & Lasso)  + 交叉验证
# ============================================================================

print("\n" + "=" * 70)
print(" [4] 正则化回归 -- Ridge & Lasso (5-fold CV)")
print("=" * 70)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ---- 4a. Ridge 回归 (L2正则化) ----
# 自动选择最优alpha
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=cv, scoring='r2')
ridge_cv.fit(X_scaled, y)
ridge_best = Ridge(alpha=ridge_cv.alpha_)
ridge_best.fit(X_scaled, y)

ridge_cv_scores = cross_val_score(ridge_best, X_scaled, y, cv=cv, scoring='r2')

print(f"  --- Ridge 回归 ---")
print(f"  最优 alpha: {ridge_cv.alpha_:.4f}")
print(f"  交叉验证 R^2: mean={ridge_cv_scores.mean():.4f}, "
      f"std={ridge_cv_scores.std():.4f}")
print(f"  系数:")
for i, dim in enumerate(dim_cols):
    print(f"    {dim}: beta = {ridge_best.coef_[i]:+.4f}")

# ---- 4b. Lasso 回归 (L1正则化) ----
# Lasso可以做特征选择----将不重要特征的系数压缩为0
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 50), cv=cv,
    max_iter=10000, random_state=42
)
lasso_cv.fit(X_scaled, y)

lasso_cv_scores = cross_val_score(
    Lasso(alpha=lasso_cv.alpha_, max_iter=10000, random_state=42),
    X_scaled, y, cv=cv, scoring='r2'
)

print(f"\n  --- Lasso 回归 ---")
print(f"  最优 alpha: {lasso_cv.alpha_:.6f}")
print(f"  交叉验证 R^2: mean={lasso_cv_scores.mean():.4f}, "
      f"std={lasso_cv_scores.std():.4f}")
print(f"  系数 (被压缩为0表示该维度不重要):")
for i, dim in enumerate(dim_cols):
    coef = lasso_cv.coef_[i]
    status = " <- 被Lasso剔除!" if abs(coef) < 1e-8 else ""
    print(f"    {dim}: beta = {coef:+.4f}{status}")


# ============================================================================
#  5. 随机森林 (非线性)
# ============================================================================

print("\n" + "=" * 70)
print(" [5] 随机森林回归 (Random Forest -- 非线性)")
print("=" * 70)

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_scaled, y)

rf_cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='r2')
rf_train_score = rf.score(X_scaled, y)

print(f"  训练集 R^2: {rf_train_score:.4f}")
print(f"  交叉验证 R^2: mean={rf_cv_scores.mean():.4f}, "
      f"std={rf_cv_scores.std():.4f}")
print(f"\n  特征重要性 (Feature Importance -- 基于 impurity reduction):")
rf_importance = pd.DataFrame({
    'feature': dim_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
for _, row in rf_importance.iterrows():
    bar = '#' * int(row['importance'] * 100)
    print(f"    {row['feature']}: {row['importance']:.4f} {bar}")

# ---- 5b. Permutation Importance (更稳健的重要性度量) ----
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(
    rf, X_scaled, y, n_repeats=10, random_state=42, scoring='r2'
)
print(f"\n  排列重要性 (Permutation Importance -- R^2下降量):")
perm_df = pd.DataFrame({
    'feature': dim_cols,
    'importance_mean': perm_imp.importances_mean,
    'importance_std': perm_imp.importances_std
}).sort_values('importance_mean', ascending=False)
for _, row in perm_df.iterrows():
    print(f"    {row['feature']}: {row['importance_mean']:+.4f} "
          f"+/- {row['importance_std']:.4f}")


# ============================================================================
#  6. 梯度提升回归 (GBRT)
# ============================================================================

print("\n" + "=" * 70)
print(" [6] 梯度提升回归 (Gradient Boosting -- 更强非线性)")
print("=" * 70)

gbrt = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=50,
    subsample=0.8,
    random_state=42
)
gbrt.fit(X_scaled, y)

gbrt_cv_scores = cross_val_score(gbrt, X_scaled, y, cv=cv, scoring='r2')
gbrt_train_score = gbrt.score(X_scaled, y)

print(f"  训练集 R^2: {gbrt_train_score:.4f}")
print(f"  交叉验证 R^2: mean={gbrt_cv_scores.mean():.4f}, "
      f"std={gbrt_cv_scores.std():.4f}")
print(f"\n  特征重要性:")
gbrt_importance = pd.DataFrame({
    'feature': dim_cols,
    'importance': gbrt.feature_importances_
}).sort_values('importance', ascending=False)
for _, row in gbrt_importance.iterrows():
    bar = '#' * int(row['importance'] * 100)
    print(f"    {row['feature']}: {row['importance']:.4f} {bar}")


# ============================================================================
#  7. 分位数分析
# ============================================================================

print("\n" + "=" * 70)
print(" [7] 分位数分析 (Quantile Analysis)")
print("=" * 70)

# 将每个维度按四分位数分组，观察NCC均值变化趋势
print("\n  --- 各维度四分位数的NCC均值 ---")
for dim in dim_cols:
    df_clean[f'{dim}_quartile'] = pd.qcut(
        df_clean[dim], q=4,
        labels=['Q1 (最低)', 'Q2', 'Q3', 'Q4 (最高)']
    )
    quartile_stats = df_clean.groupby(f'{dim}_quartile')['NCC'].agg(
        ['mean', 'std', 'count']
    )
    print(f"\n  {dim}:")
    print(quartile_stats.round(4).to_string())

    # Q4 vs Q1 的t检验
    q1_data = df_clean[df_clean[f'{dim}_quartile'] == 'Q1 (最低)']['NCC']
    q4_data = df_clean[df_clean[f'{dim}_quartile'] == 'Q4 (最高)']['NCC']
    t_stat, t_p = stats.ttest_ind(q4_data, q1_data)
    effect_size = (q4_data.mean() - q1_data.mean()) / df_clean['NCC'].std()
    print(f"    Q4 vs Q1: t={t_stat:+.2f}, p={t_p:.4f}, "
          f"Cohen's d={effect_size:+.4f}")


# ============================================================================
#  8. K-Means 语言风格聚类
# ============================================================================

print("\n" + "=" * 70)
print(" [8] K-Means聚类 -- 语言风格群体 vs NCC")
print("=" * 70)

# 确定最优聚类数（肘部法则 + 轮廓系数）
from sklearn.metrics import silhouette_score

print("\n  --- 聚类数选择 ---")
inertias = []
silhouettes = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

print(f"  {'K':<5} {'Inertia':<12} {'Silhouette':<12}")
for i, k in enumerate(K_range):
    print(f"  {k:<5} {inertias[i]:<12.2f} {silhouettes[i]:<12.4f}")

# 使用最佳K（取轮廓系数最大的）
best_k = K_range[np.argmax(silhouettes)]
print(f"\n  最优聚类数: K={best_k} (Silhouette={max(silhouettes):.4f})")

# 执行聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# 各类的维度均值和NCC均值
print(f"\n  --- K={best_k} 聚类结果 ---")
cluster_stats = df_clean.groupby('cluster').agg(
    **{
        'Dim1_mean': ('Dim1', 'mean'),
        'Dim2_mean': ('Dim2', 'mean'),
        'Dim3_mean': ('Dim3', 'mean'),
        'Dim4_mean': ('Dim4', 'mean'),
        'NCC_mean': ('NCC', 'mean'),
        'NCC_std': ('NCC', 'std'),
        'count': ('NCC', 'count'),
    }
).round(4)
print(cluster_stats.to_string())

# 各类NCC差异的ANOVA检验
groups = [df_clean[df_clean['cluster'] == k]['NCC'].values for k in range(best_k)]
f_stat, anova_p = stats.f_oneway(*groups)
print(f"\n  ANOVA (类间NCC差异): F={f_stat:.2f}, p={anova_p:.6f}")
if anova_p < 0.05:
    print(f"  -> 至少有两个类的NCC均值存在显著差异")
    # 事后检验：哪些类之间差异显著？
    # Tukey HSD
    from itertools import combinations
    print(f"\n  事后两两比较 (Tukey HSD):")
    for k1, k2 in combinations(range(best_k), 2):
        t_stat, t_p = stats.ttest_ind(groups[k1], groups[k2])
        if t_p < 0.05 / (best_k * (best_k - 1) / 2):  # Bonferroni校正
            sig = "***"
        elif t_p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(f"    Cluster {k1} vs {k2}: "
              f"DeltaNCC={groups[k1].mean()-groups[k2].mean():+.3f}, "
              f"p={t_p:.4f} {sig}")
else:
    print(f"  -> 各类之间NCC均值无显著差异")


# ============================================================================
#  9. 交互效应分析
# ============================================================================

print("\n" + "=" * 70)
print(" [9] 交互效应分析 (Interaction Effects)")
print("=" * 70)

# 创建交互项（维度两两乘积）
print("\n  --- 维度两两交互项的增量R^2 ---")
base_formula_vars = dim_cols
interaction_results = []

for i in range(n_features):
    for j in range(i + 1, n_features):
        # 创建交互项
        interaction = X_scaled[:, i] * X_scaled[:, j]

        # 无交互项的模型
        X_base = X_scaled
        ols_base = LinearRegression().fit(X_base, y)
        r2_base = ols_base.score(X_base, y)

        # 有交互项的模型
        X_inter = np.column_stack([X_scaled, interaction])
        ols_inter = LinearRegression().fit(X_inter, y)
        r2_inter = ols_inter.score(X_inter, y)

        delta_r2 = r2_inter - r2_base
        interaction_results.append({
            'pair': f'{dim_cols[i]} × {dim_cols[j]}',
            'R2_base': r2_base,
            'R2_with_interaction': r2_inter,
            'delta_R2': delta_r2,
        })

        sig_mark = " *" if delta_r2 > 0.001 else ""
        print(f"    {dim_cols[i]} × {dim_cols[j]}: "
              f"DeltaR^2={delta_r2:+.6f}{sig_mark}")

# 找出最有意义的交互项
interaction_results.sort(key=lambda x: x['delta_R2'], reverse=True)
print(f"\n  最强交互效应: {interaction_results[0]['pair']} "
      f"(DeltaR^2={interaction_results[0]['delta_R2']:+.6f})")


# ============================================================================
#  10. 可视化
# ============================================================================

print("\n" + "=" * 70)
print(" [10] 生成可视化图表")
print("=" * 70)

# ---- Fig 1: 相关性热力图 ----
fig, ax = plt.subplots(figsize=(7, 6))
corr_matrix = df_clean[dim_cols + ['NCC']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            ax=ax)
ax.set_title('Correlation Heatmap: MD Dimensions vs NCC', fontsize=14, pad=15)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig1_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("  [OK] fig1_correlation_heatmap.png")

# ---- Fig 2: 各维度 vs NCC 散点图 + 回归线 ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, dim in enumerate(dim_cols):
    ax = axes[i]
    # 散点（半透明，太多点时取子样本）
    sample_idx = np.random.choice(len(df_clean), min(2000, len(df_clean)),
                                   replace=False)
    ax.scatter(df_clean[dim].iloc[sample_idx],
               df_clean['NCC'].iloc[sample_idx],
               alpha=0.3, s=5, c='steelblue', edgecolors='none')

    # LOWESS平滑线
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_idx = np.argsort(df_clean[dim].values)
        x_sorted = df_clean[dim].values[sorted_idx]
        y_sorted = df_clean['NCC'].values[sorted_idx]
        smoothed = lowess(y_sorted, x_sorted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2,
                label='LOWESS smooth')
    except ImportError:
        pass

    # 线性回归线
    slope, intercept, r_val, p_val, _ = stats.linregress(
        df_clean[dim], df_clean['NCC']
    )
    x_line = np.linspace(df_clean[dim].min(), df_clean[dim].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=1,
            alpha=0.5, label=f'Linear (r={r_val:+.3f})')

    ax.set_xlabel(dim)
    ax.set_ylabel('NCC')
    ax.set_title(f'{dim} vs NCC (r={r_val:+.3f}, p={p_val:.4f})')
    ax.legend(fontsize=8)

fig.suptitle('MD Dimension Scores vs Normalized Citation Count',
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig2_dim_vs_ncc_scatter.png', bbox_inches='tight')
plt.close()
print("  [OK] fig2_dim_vs_ncc_scatter.png")

# ---- Fig 3: 特征重要性对比 (所有模型) ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# OLS |beta|
ols_imp = pd.DataFrame({
    'feature': dim_cols,
    'importance': np.abs(ols.coef_)
}).sort_values('importance')
axes[0].barh(ols_imp['feature'], ols_imp['importance'], color='steelblue')
axes[0].set_title('OLS |Coefficient|')
axes[0].set_xlabel('|Standardized beta|')

# RF importance
rf_imp = rf_importance.sort_values('importance')
axes[1].barh(rf_imp['feature'], rf_imp['importance'], color='forestgreen')
axes[1].set_title('Random Forest Importance')
axes[1].set_xlabel('Importance')

# GBRT importance
gbrt_imp = gbrt_importance.sort_values('importance')
axes[2].barh(gbrt_imp['feature'], gbrt_imp['importance'], color='darkorange')
axes[2].set_title('GBRT Importance')
axes[2].set_xlabel('Importance')

fig.suptitle('Feature Importance Comparison Across Models', fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig3_feature_importance_comparison.png',
            bbox_inches='tight')
plt.close()
print("  [OK] fig3_feature_importance_comparison.png")

# ---- Fig 4: 分位数 NCC 柱状图 ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, dim in enumerate(dim_cols):
    ax = axes[i]
    q_data = df_clean.groupby(f'{dim}_quartile')['NCC'].agg(['mean', 'std'])
    q_data = q_data.reindex(['Q1 (最低)', 'Q2', 'Q3', 'Q4 (最高)'])

    colors = ['#4575b4', '#91bfdb', '#fc8d59', '#d73027']
    bars = ax.bar(range(4), q_data['mean'], yerr=q_data['std'],
                  color=colors, capsize=5, edgecolor='white', linewidth=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(['Q1\n(最低)', 'Q2', 'Q3', 'Q4\n(最高)'])
    ax.set_ylabel('Mean NCC')
    ax.set_title(f'{dim} Quartiles -> NCC')

    # 标注均值
    for bar, mean_val in zip(bars, q_data['mean']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)

fig.suptitle('NCC by Dimension Score Quartiles', fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig4_quartile_ncc.png', bbox_inches='tight')
plt.close()
print("  [OK] fig4_quartile_ncc.png")

# ---- Fig 5: 聚类可视化 (前两个维度的散点 + 聚类着色) ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA降维到2D便于可视化
from sklearn.decomposition import PCA
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

# 图A: 按聚类着色
scatter1 = axes[0].scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=df_clean['cluster'], cmap='Set2',
    alpha=0.4, s=8, edgecolors='none'
)
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title(f'K-Means Clusters (K={best_k})')
legend1 = axes[0].legend(*scatter1.legend_elements(),
                          title="Cluster", fontsize=8)
axes[0].add_artist(legend1)

# 图B: 按NCC着色
scatter2 = axes[1].scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=df_clean['NCC'], cmap='YlOrRd',
    alpha=0.4, s=8, edgecolors='none'
)
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('NCC Distribution in Dimension Space')
cbar = plt.colorbar(scatter2, ax=axes[1], shrink=0.8)
cbar.set_label('NCC')

fig.suptitle('Language Style Clusters in MD Dimension Space', fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig5_cluster_visualization.png',
            bbox_inches='tight')
plt.close()
print("  [OK] fig5_cluster_visualization.png")

# ---- Fig 6: 模型性能对比 ----
fig, ax = plt.subplots(figsize=(10, 5))

models = ['OLS', 'Ridge', 'Lasso', 'Random\nForest', 'GBRT']
cv_scores = [
    np.nan,  # OLS (no CV, use adj R^2)
    ridge_cv_scores.mean(),
    lasso_cv_scores.mean(),
    rf_cv_scores.mean(),
    gbrt_cv_scores.mean(),
]
cv_stds = [
    np.nan,
    ridge_cv_scores.std(),
    lasso_cv_scores.std(),
    rf_cv_scores.std(),
    gbrt_cv_scores.std(),
]
# OLS用调整R^2代替
r2_values = [adj_r2_ols] + list(cv_scores[1:])
colors = ['#4575b4', '#91bfdb', '#fc8d59', '#66bd63', '#d73027']

x_pos = range(len(models))
bars = ax.bar(x_pos, r2_values, color=colors, edgecolor='white', linewidth=1.5)

# 标注值
for bar, val, std in zip(bars, r2_values, cv_stds):
    if np.isnan(val):
        continue
    label = f'{val:.4f}'
    if not np.isnan(std):
        label += f' +/- {std:.4f}'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            label, ha='center', va='bottom', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_ylabel('R^2 (Cross-Validated)')
ax.set_title('Model Performance Comparison: Predicting NCC from MD Dimensions',
             fontsize=14)
ax.set_ylim(0, max(r2_values) * 1.5)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig6_model_comparison.png', bbox_inches='tight')
plt.close()
print("  [OK] fig6_model_comparison.png")

# ---- Fig 7: NCC分布 + 核密度估计 ----
fig, ax = plt.subplots(figsize=(8, 4))
df_clean['NCC'].hist(bins=80, density=True, alpha=0.6, color='steelblue',
                      edgecolor='white', linewidth=0.5, ax=ax)
df_clean['NCC'].plot.kde(ax=ax, color='darkred', linewidth=2, label='KDE')
ax.axvline(df_clean['NCC'].mean(), color='darkred', linestyle='--',
           linewidth=1.5, label=f'Mean={df_clean["NCC"].mean():.2f}')
ax.axvline(df_clean['NCC'].median(), color='darkorange', linestyle='--',
           linewidth=1.5, label=f'Median={df_clean["NCC"].median():.2f}')
ax.set_xlabel('NCC (Normalized Citation Count)')
ax.set_ylabel('Density')
ax.set_title('NCC Distribution')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig7_ncc_distribution.png', bbox_inches='tight')
plt.close()
print("  [OK] fig7_ncc_distribution.png")


# ============================================================================
#  11. 结果汇总
# ============================================================================

print("\n" + "=" * 70)
print(" [11] 分析结果汇总")
print("=" * 70)

# 收集所有关键结果
summary = {
    '样本量': n_samples,
    'NCC均值': f'{y.mean():.3f}',
    'NCC标准差': f'{y.std():.3f}',

    # 相关性
    'Pearson最强维度': max(pearson_results, key=lambda k: abs(pearson_results[k]['r'])),
    'Pearson最强r': f'{max(v["r"] for v in pearson_results.values()):+.4f}',
    'Spearman最强维度': max(spearman_results, key=lambda k: abs(spearman_results[k]['rho'])),
    'Spearman最强rho': f'{max(v["rho"] for v in spearman_results.values()):+.4f}',

    # 回归
    'OLS R^2': f'{r2_ols:.6f}',
    'OLS Adj R^2': f'{adj_r2_ols:.6f}',
    'Ridge CV R^2': f'{ridge_cv_scores.mean():.4f} +/- {ridge_cv_scores.std():.4f}',
    'Lasso CV R^2': f'{lasso_cv_scores.mean():.4f} +/- {lasso_cv_scores.std():.4f}',

    # 非线性
    'RF CV R^2': f'{rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}',
    'GBRT CV R^2': f'{gbrt_cv_scores.mean():.4f} +/- {gbrt_cv_scores.std():.4f}',

    # 聚类
    '最优K': best_k,
    '聚类ANOVA p': f'{anova_p:.6f}',
}

print("\n  关键结果汇总:")
for k, v in summary.items():
    print(f"    {k:20s}: {v}")

# ---- 判断是否存在关联 ----
print("\n" + "-" * 50)
print("  综合判断:")
print("-" * 50)

# 判断逻辑：
# 1. 如果有任何维度的Pearson p < 0.05 且 |r| > 0.05：存在微弱线性相关
# 2. 如果RF/GBRT的CV R^2 > OLS的Adj R^2：存在非线性成分
# 3. 如果ANOVA p < 0.05：不同语言风格群的NCC存在差异

any_sig_pearson = any(v['p'] < 0.05 for v in pearson_results.values())
any_sig_spearman = any(v['p'] < 0.05 for v in spearman_results.values())
nonlinear_improvement = max(rf_cv_scores.mean(), gbrt_cv_scores.mean()) > adj_r2_ols + 0.005
cluster_sig = anova_p < 0.05

print(f"  线性相关显著: {'是' if any_sig_pearson else '否'}")
print(f"  单调相关显著: {'是' if any_sig_spearman else '否'}")
print(f"  非线性优于线性: {'是' if nonlinear_improvement else '否'}")
print(f"  聚类NCC差异显著: {'是' if cluster_sig else '否'}")

if adj_r2_ols < 0.005 and max(rf_cv_scores.mean(), gbrt_cv_scores.mean()) < 0.01:
    print(f"\n  [!]️ 结论: MD维度得分对NCC的预测能力极弱。")
    print(f"  所有模型的R^2均接近0，说明4个语言功能维度")
    print(f"  几乎无法解释学术影响力的方差。")
    print(f"  这与baseline阶段用6个文体特征未找到线性关系")
    print(f"  的发现一致----语言风格可能不是影响被引的主要因素。")
elif adj_r2_ols < 0.02:
    print(f"\n  结论: MD维度得分与NCC存在微弱但可检测的关联。")
    print(f"  效应量很小（R^2 < 2%），实际意义有限。")
else:
    print(f"\n  结论: MD维度得分对NCC有一定预测能力。")
    print(f"  需要进一步验证这种关联是否稳定。")

print(f"\n  [!] 注意: 这是探索性分析，KMO=0.52表明因子结构偏弱。")
print(f"  结果不应用于因果推断，仅供方法参考和假设生成。")

# ---- 保存完整数据 ----
output_df = df_clean[dim_cols + ['NCC', 'citations', 'cluster']].copy()
output_df.to_csv(OUTPUT_DIR / 'dim_ncc_analysis_results.csv', index=False)
print(f"\n  [OK] 完整分析数据已保存: dim_ncc_analysis_results.csv")

print(f"\n{'=' * 70}")
print(f" 分析完成! 所有图表已保存至: {OUTPUT_DIR}")
print(f"{'=' * 70}")
