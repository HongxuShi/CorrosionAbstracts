#!/usr/bin/env python3
"""
================================================================================
 高/低影响力论文分类分析
 High vs. Low Impact Paper Classification from MD Dimension Scores
================================================================================

核心问题（从回归改为分类）：
  语言特征能否区分"高影响力"和"低影响力"论文？

为什么分类比回归更合理：
  - 回归试图回答"NCC=3.2还是3.5?"——这对copilot没意义
  - 分类回答"这篇摘要的写法更像高被引还是低被引论文？"——对copilot有指导意义
  - 极端组对比能放大信号（砍掉中间模糊地带）

分组策略（5种，从宽松到极端）：
  ┌──────────┬──────────────────────┬─────────────────────────┐
  │ 分组策略 │ 高影响力定义         │ 低影响力定义            │
  ├──────────┼──────────────────────┼─────────────────────────┤
  │ S1       │ NCC Top 25%          │ NCC Bottom 25%          │
  │ S2       │ NCC Top 20%          │ NCC Bottom 20%          │
  │ S3       │ NCC Top 10%          │ NCC Bottom 50%          │
  │ S4       │ NCC Top 10%          │ NCC Bottom 10% (极端)   │
  │ S5       │ NCC > 0 (有引用)     │ NCC = 0 (零引用)        │
  └──────────┴──────────────────────┴─────────────────────────┘

分类器（4种，从简单到复杂）：
  - Logistic Regression (L2正则化，可解释的系数)
  - Random Forest (非线性，特征重要性)
  - XGBoost (梯度提升，通常是表格数据最强模型)
  - Linear SVM (简单线性决策边界)

评估指标：
  - AUC-ROC (主要指标)
  - F1, Precision, Recall
  - 5-fold Stratified Cross-Validation
  - Per-feature Cohen's d (效应量)
  - 混淆矩阵

输出：
  - 控制台：所有模型×所有分组策略的评估表
  - lab/output/fig_class_*.png：ROC曲线、特征重要性、Cohen's d等

作者：理论验证阶段
日期：2026/07/17
================================================================================
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

import numpy as np
import pandas as pd

# 统计与ML
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150,
    'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
})

# ============================================================================
# 路径配置
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_SCORES_PATH = SCRIPT_DIR / "output" / "dimension_scores.csv"
NCC_DATA_PATH = SCRIPT_DIR.parent / "processed" / "merged_all_features.csv"

print("=" * 70)
print(" 高/低影响力论文分类分析")
print(" MD Dimension Scores -> Binary Classification")
print("=" * 70)

# ============================================================================
# 0. 数据准备
# ============================================================================
print("\n[0] 数据加载...")
df_scores = pd.read_csv(DIM_SCORES_PATH)
df_ncc = pd.read_csv(NCC_DATA_PATH)

df = df_scores.merge(
    df_ncc[['doi', 'year', 'citations', 'NCC', 'source_journal']],
    left_on='doc_id', right_on='doi', how='inner'
).drop(columns=['doi'])

df = df.dropna(subset=['NCC'])
print(f"  合并后: {len(df)} 条")

dim_cols = ['Dim1', 'Dim2', 'Dim3', 'Dim4']

# ============================================================================
# 1. 多策略分组
# ============================================================================
print("\n[1] 构建分组标签...")

def build_labels(df, strategy_name, high_func, low_func):
    """为数据集构建二分类标签。返回带标签的df子集。"""
    high_mask = high_func(df)
    low_mask = low_func(df)
    subset = df[high_mask | low_mask].copy()
    subset['label'] = 0
    subset.loc[high_mask, 'label'] = 1
    return subset

# 5种分组策略
strategies = {}

# S1: Top 25% vs Bottom 25%
ncc_q75, ncc_q25 = df['NCC'].quantile(0.75), df['NCC'].quantile(0.25)
strategies['S1_Top25vsBot25'] = build_labels(
    df, 'S1',
    lambda d: d['NCC'] >= ncc_q75,
    lambda d: d['NCC'] <= ncc_q25
)

# S2: Top 20% vs Bottom 20%
ncc_q80, ncc_q20 = df['NCC'].quantile(0.80), df['NCC'].quantile(0.20)
strategies['S2_Top20vsBot20'] = build_labels(
    df, 'S2',
    lambda d: d['NCC'] >= ncc_q80,
    lambda d: d['NCC'] <= ncc_q20
)

# S3: Top 10% vs Bottom 50%
ncc_q90, ncc_q50 = df['NCC'].quantile(0.90), df['NCC'].quantile(0.50)
strategies['S3_Top10vsBot50'] = build_labels(
    df, 'S3',
    lambda d: d['NCC'] >= ncc_q90,
    lambda d: d['NCC'] <= ncc_q50
)

# S4: Top 10% vs Bottom 10% (极端组)
ncc_q90b, ncc_q10 = df['NCC'].quantile(0.90), df['NCC'].quantile(0.10)
strategies['S4_Top10vsBot10'] = build_labels(
    df, 'S4',
    lambda d: d['NCC'] >= ncc_q90b,
    lambda d: d['NCC'] <= ncc_q10
)

# S5: 零引用 vs 有引用
strategies['S5_CitedVsUncited'] = build_labels(
    df, 'S5',
    lambda d: d['citations'] > 0,
    lambda d: d['citations'] == 0
)

for name, sdf in strategies.items():
    n_high = sdf['label'].sum()
    n_low = len(sdf) - n_high
    print(f"  {name}: {len(sdf)} 篇 (高={n_high}, 低={n_low}, "
          f"比例={n_high/len(sdf):.1%}/{n_low/len(sdf):.1%})")

# ============================================================================
# 2. 单特征效应量分析 (Cohen's d for each dimension in each strategy)
# ============================================================================
print("\n[2] 单特征效应量分析 (Cohen's d)...")

def cohens_d(group1, group2):
    """计算Cohen's d效应量。0.2=小, 0.5=中, 0.8=大。"""
    n1, n2 = len(group1), len(group2)
    s_pooled = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    if s_pooled < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / s_pooled

effect_sizes = {}
for strat_name, sdf in strategies.items():
    high = sdf[sdf['label'] == 1]
    low = sdf[sdf['label'] == 0]
    effect_sizes[strat_name] = {}
    for dim in dim_cols:
        d = cohens_d(high[dim].values, low[dim].values)
        # 也做t检验
        t_stat, t_p = ttest_ind(high[dim], low[dim])
        effect_sizes[strat_name][dim] = {
            'cohens_d': d,
            't_stat': t_stat,
            'p_value': t_p,
            'high_mean': high[dim].mean(),
            'low_mean': low[dim].mean(),
        }

# 打印效应量汇总
print(f"\n  {'Strategy':<25s}", end='')
for dim in dim_cols:
    print(f"  {dim:>12s}", end='')
print(f"  {'best_dim':>12s}")
print("  " + "-" * 85)

for strat_name, effects in effect_sizes.items():
    print(f"  {strat_name:<25s}", end='')
    best_d = 0
    best_dim = ''
    for dim in dim_cols:
        d = effects[dim]['cohens_d']
        sig = '**' if effects[dim]['p_value'] < 0.01 else ('*' if effects[dim]['p_value'] < 0.05 else '')
        print(f"  {d:+.3f}{sig:2s}", end='')
        if abs(d) > abs(best_d):
            best_d = d
            best_dim = dim
    print(f"  {best_dim}({best_d:+.3f})")

# ============================================================================
# 3. 多模型 + 多策略交叉验证分类
# ============================================================================
print("\n[3] 分类模型交叉验证...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def build_models(random_state=42):
    """构建所有分类器。"""
    return {
        'Logistic Regression': LogisticRegression(
            penalty='l2', C=1.0, max_iter=5000, random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=30,
            random_state=random_state, n_jobs=-1
        ),
        'XGBoost': None,  # 将在try块中初始化
        'Linear SVM': CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, dual='auto', random_state=random_state),
            cv=3
        ),
    }

# 尝试导入XGBoost
try:
    from xgboost import XGBClassifier
    _xgb_available = True
except ImportError:
    _xgb_available = False
    print("  (XGBoost未安装，跳过)")

# 存储所有结果
all_results = {}

for strat_name, sdf in strategies.items():
    X = StandardScaler().fit_transform(sdf[dim_cols].values)
    y = sdf['label'].values

    all_results[strat_name] = {
        'n_samples': len(sdf),
        'n_high': int(y.sum()),
        'n_low': int(len(y) - y.sum()),
        'baseline': max(y.mean(), 1 - y.mean()),  # 多数类比例
        'models': {},
    }

    models = build_models()
    if _xgb_available:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric='logloss', random_state=42, verbosity=0
        )

    for model_name, model in models.items():
        if model is None:
            continue

        scoring = {
            'auc': 'roc_auc',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
        }
        try:
            scores = cross_validate(
                model, X, y, cv=cv, scoring=scoring, n_jobs=-1
            )
            all_results[strat_name]['models'][model_name] = {
                'auc_mean': scores['test_auc'].mean(),
                'auc_std': scores['test_auc'].std(),
                'f1_mean': scores['test_f1'].mean(),
                'f1_std': scores['test_f1'].std(),
                'precision_mean': scores['test_precision'].mean(),
                'precision_std': scores['test_precision'].std(),
                'recall_mean': scores['test_recall'].mean(),
                'recall_std': scores['test_recall'].std(),
            }
        except Exception as e:
            print(f"  [!] {strat_name} / {model_name}: {e}")

# ============================================================================
# 4. 结果汇总表
# ============================================================================
print("\n[4] 分类结果汇总")
print("=" * 100)
print(f"  {'Strategy':<25s} {'Model':<20s} {'AUC':>8s} {'F1':>8s} "
      f"{'Precision':>10s} {'Recall':>8s} {'Baseline':>8s}")
print("  " + "-" * 95)

best_per_strategy = {}

for strat_name, strat_results in all_results.items():
    baseline = strat_results['baseline']
    printed_strat = False
    best_auc = 0
    best_model_name = ''

    for model_name, metrics in strat_results['models'].items():
        auc_str = f"{metrics['auc_mean']:.3f}+/-{metrics['auc_std']:.3f}"
        f1_str = f"{metrics['f1_mean']:.3f}"
        prec_str = f"{metrics['precision_mean']:.3f}"
        rec_str = f"{metrics['recall_mean']:.3f}"

        strat_label = strat_name if not printed_strat else ''
        print(f"  {strat_label:<25s} {model_name:<20s} {auc_str:>8s} {f1_str:>8s} "
              f"{prec_str:>10s} {rec_str:>8s} {baseline:.3f}{' (多数类)' if not printed_strat else ''}")
        printed_strat = True

        if metrics['auc_mean'] > best_auc:
            best_auc = metrics['auc_mean']
            best_model_name = model_name

    best_per_strategy[strat_name] = (best_model_name, best_auc)

# 最佳组合
overall_best = max(best_per_strategy.items(), key=lambda x: x[1][1])
print(f"\n  最佳组合: {overall_best[0]} + {overall_best[1][0]} "
      f"(AUC={overall_best[1][1]:.3f})")

# ============================================================================
# 5. 特征重要性（Logistic Regression系数 + RF特征重要性）
# ============================================================================
print("\n[5] 特征重要性分析...")

# 对最佳策略拟合最终模型并提取特征重要性
feature_importance = {}

for strat_name in ['S1_Top25vsBot25', 'S2_Top20vsBot20', 'S4_Top10vsBot10']:
    if strat_name not in strategies:
        continue
    sdf = strategies[strat_name]
    X = StandardScaler().fit_transform(sdf[dim_cols].values)
    y = sdf['label'].values

    # Logistic Regression系数
    lr = LogisticRegression(penalty='l2', C=1.0, max_iter=5000, random_state=42)
    lr.fit(X, y)

    # Random Forest重要性
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=30,
        random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    # Permutation importance (for RF)
    from sklearn.inspection import permutation_importance
    perm_imp = permutation_importance(
        rf, X, y, n_repeats=20, random_state=42, scoring='roc_auc'
    )

    feature_importance[strat_name] = {
        'lr_coef': {dim: lr.coef_[0][i] for i, dim in enumerate(dim_cols)},
        'rf_importance': {dim: rf.feature_importances_[i] for i, dim in enumerate(dim_cols)},
        'perm_importance': {dim: perm_imp.importances_mean[i] for i, dim in enumerate(dim_cols)},
        'perm_std': {dim: perm_imp.importances_std[i] for i, dim in enumerate(dim_cols)},
    }

    print(f"\n  {strat_name}:")
    print(f"    {'Dim':<8s} {'LR Coef':>10s} {'RF Imp':>10s} {'Perm Imp':>12s}")
    for dim in dim_cols:
        print(f"    {dim:<8s} {feature_importance[strat_name]['lr_coef'][dim]:+10.4f} "
              f"{feature_importance[strat_name]['rf_importance'][dim]:10.4f} "
              f"{feature_importance[strat_name]['perm_importance'][dim]:10.4f} "
              f"+/-{feature_importance[strat_name]['perm_std'][dim]:.4f}")

# ============================================================================
# 6. 可视化
# ============================================================================
print("\n[6] 生成可视化图表...")

# ---- Fig 1: Cohen's d热力图 (所有策略 × 所有维度) ----
fig, ax = plt.subplots(figsize=(10, 6))
d_matrix = np.zeros((len(effect_sizes), len(dim_cols)))
for i, (strat_name, effects) in enumerate(effect_sizes.items()):
    for j, dim in enumerate(dim_cols):
        d_matrix[i, j] = effects[dim]['cohens_d']

# 用缩略策略名
short_names = [s.replace('_', '\n') for s in effect_sizes.keys()]
sns.heatmap(d_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=dim_cols, yticklabels=short_names,
            linewidths=1, cbar_kws={'label': "Cohen's d", 'shrink': 0.8},
            vmin=-0.3, vmax=0.3, ax=ax)
ax.set_title("Cohen's d: High vs Low Impact (per Dimension & Strategy)", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_class1_cohens_d_heatmap.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_class1_cohens_d_heatmap.png")

# ---- Fig 2: AUC对比柱状图 (所有模型 × 所有策略) ----
fig, ax = plt.subplots(figsize=(14, 7))
model_colors = {
    'Logistic Regression': '#4575b4',
    'Random Forest': '#66bd63',
    'XGBoost': '#d73027',
    'Linear SVM': '#fc8d59',
}

x_positions = []
x_labels = []
bar_idx = 0
n_models = 0

for strat_name, strat_results in all_results.items():
    models_in_strat = list(strat_results['models'].keys())
    n_models = max(n_models, len(models_in_strat))
    for j, (model_name, metrics) in enumerate(strat_results['models'].items()):
        x_positions.append(bar_idx)
        color = model_colors.get(model_name, '#999999')
        bar = ax.bar(bar_idx, metrics['auc_mean'],
                     yerr=metrics['auc_std'],
                     color=color, edgecolor='white', linewidth=1,
                     capsize=3, width=0.7,
                     label=model_name if bar_idx < len(models_in_strat) else '')
        # 标注AUC值
        ax.text(bar_idx, metrics['auc_mean'] + metrics['auc_std'] + 0.005,
                f"{metrics['auc_mean']:.3f}", ha='center', fontsize=7, rotation=90)
        bar_idx += 1
    # 策略分隔线
    if bar_idx > 0:
        x_labels.append((bar_idx - len(models_in_strat) / 2 - 0.5, strat_name.replace('_', '\n')))

ax.axhline(y=0.5, color='gray', linewidth=0.8, linestyle='--', label='Random (0.5)')
# 标注baseline
for i, (strat_name, strat_results) in enumerate(all_results.items()):
    bl = strat_results['baseline']
    ax.axhline(y=bl, color='#bdbdbd', linewidth=0.5, linestyle=':', alpha=0.7)

ax.set_xticks([p[0] for p in x_labels])
ax.set_xticklabels([p[1] for p in x_labels], fontsize=8)
ax.set_ylabel('AUC-ROC (5-fold CV)')
ax.set_title('Classification Performance: AUC-ROC Across Strategies & Models', fontsize=14)
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.set_ylim(0.35, 0.9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_class2_auc_comparison.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_class2_auc_comparison.png")

# ---- Fig 3: 最佳模型的ROC曲线 (per strategy) ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, strat_name in enumerate(list(strategies.keys())[:6]):
    if idx >= len(axes):
        break
    ax = axes[idx]
    sdf = strategies[strat_name]
    X = StandardScaler().fit_transform(sdf[dim_cols].values)
    y = sdf['label'].values

    # 对每个模型画ROC（使用outer CV的预测）
    for model_name, ModelClass, kwargs in [
        ('LR', LogisticRegression, {'penalty': 'l2', 'C': 1.0, 'max_iter': 5000}),
        ('RF', RandomForestClassifier, {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 30}),
    ]:
        model = ModelClass(random_state=42, **kwargs)
        # 用一个fold的预测作为ROC示例
        from sklearn.model_selection import cross_val_predict
        try:
            y_proba = cross_val_predict(
                model, X, y, cv=cv, method='predict_proba', n_jobs=-1
            )[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            auc = roc_auc_score(y, y_proba)
            ax.plot(fpr, tpr, linewidth=1.5, alpha=0.8,
                    label=f'{model_name} (AUC={auc:.3f})')
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5, label='Random')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(strat_name.replace('_', ' '), fontsize=11)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# 隐藏多余的subplot
for idx in range(len(strategies), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('ROC Curves: All Strategies (LR & RF)', fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_class3_roc_curves.png', bbox_inches='tight')
plt.close()
print("  [OK] fig_class3_roc_curves.png")

# ---- Fig 4: 最佳策略的特征重要性全景 ----
# 选择一个代表性策略 (S2: Top 20% vs Bottom 20%)
best_strat = 'S2_Top20vsBot20'
if best_strat in feature_importance:
    fi = feature_importance[best_strat]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # LR系数
    lr_vals = [fi['lr_coef'][d] for d in dim_cols]
    colors_lr = ['#d73027' if v < 0 else '#4575b4' for v in lr_vals]
    axes[0].barh(dim_cols, lr_vals, color=colors_lr, edgecolor='white')
    axes[0].axvline(0, color='black', linewidth=0.5)
    axes[0].set_title('Logistic Regression Coefficients')
    axes[0].set_xlabel('Coefficient (standardized)')

    # RF重要性
    rf_vals = [fi['rf_importance'][d] for d in dim_cols]
    axes[1].barh(dim_cols, rf_vals, color='forestgreen', edgecolor='white')
    axes[1].set_title('Random Forest Importance')
    axes[1].set_xlabel('Importance (impurity)')

    # Permutation重要性
    perm_vals = [fi['perm_importance'][d] for d in dim_cols]
    perm_errs = [fi['perm_std'][d] for d in dim_cols]
    colors_perm = ['#d73027' if v < 0 else '#4575b4' for v in perm_vals]
    axes[2].barh(dim_cols, perm_vals, xerr=perm_errs, color=colors_perm,
                 edgecolor='white', capsize=3)
    axes[2].axvline(0, color='black', linewidth=0.5)
    axes[2].set_title('Permutation Importance (+/- std)')
    axes[2].set_xlabel('AUC Drop when Permuted')

    fig.suptitle(f'Feature Importance: {best_strat}', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_class4_feature_importance.png',
                bbox_inches='tight')
    plt.close()
    print("  [OK] fig_class4_feature_importance.png")

# ---- Fig 5: 高低组维度分布对比 (小提琴图) ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

sdf = strategies['S2_Top20vsBot20']  # 使用S2策略
for i, dim in enumerate(dim_cols):
    ax = axes[i]
    data_high = sdf[sdf['label'] == 1][dim].values
    data_low = sdf[sdf['label'] == 0][dim].values

    parts = ax.violinplot(
        [data_low, data_high], positions=[0, 1],
        showmeans=True, showmedians=True
    )
    # 着色
    for pc, color in zip(parts['bodies'], ['#4575b4', '#d73027']):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # 标注效应量
    d_val = effect_sizes['S2_Top20vsBot20'][dim]['cohens_d']
    p_val = effect_sizes['S2_Top20vsBot20'][dim]['p_value']
    ax.set_title(f'{dim} (d={d_val:+.3f}, p={p_val:.4f})')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low Impact\n(Bottom 20%)', 'High Impact\n(Top 20%)'])
    ax.set_ylabel('Dimension Score')

fig.suptitle('Dimension Score Distributions: High vs Low Impact (S2)', fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_class5_violin_distributions.png',
            bbox_inches='tight')
plt.close()
print("  [OK] fig_class5_violin_distributions.png")

# ============================================================================
# 7. 综合结论
# ============================================================================
print("\n[7] 综合结论")
print("=" * 70)

# 汇总各策略的最佳AUC
print("\n  各策略最佳AUC:")
for strat_name, (model_name, auc) in best_per_strategy.items():
    bl = all_results[strat_name]['baseline']
    improvement = auc - 0.5
    above_baseline = auc - bl
    print(f"    {strat_name:<30s} {model_name:<20s} AUC={auc:.3f} "
          f"(baseline={bl:.3f}, >random={improvement:+.3f}, >baseline={above_baseline:+.3f})")

# 找到最强的单特征效应
best_effect = {'strat': '', 'dim': '', 'd': 0}
for strat_name, effects in effect_sizes.items():
    for dim, vals in effects.items():
        if abs(vals['cohens_d']) > abs(best_effect['d']):
            best_effect = {'strat': strat_name, 'dim': dim, 'd': vals['cohens_d'],
                           'p': vals['p_value']}

print(f"\n  最强单特征效应: {best_effect['strat']} / {best_effect['dim']} "
      f"(d={best_effect['d']:+.3f}, p={best_effect['p']:.4f})")

# 最终判断
best_auc_overall = max(v[1] for v in best_per_strategy.values())
if best_auc_overall < 0.55:
    print(f"\n  [CONCLUSION] 分类效果微弱 (最佳AUC={best_auc_overall:.3f})。")
    print(f"  即使改为分类问题，MD维度得分区分高/低影响力论文的能力仍然极有限。")
    print(f"  AUC仅略高于0.5（随机猜测），说明在当前特征集下，")
    print(f"  论文的引用命运无法通过摘要的语言风格来有效预测。")
    print(f"\n  对Copilot的启示：")
    print(f"  1. 15个Biber特征 + 4个PCA维度不足以提供写作建议")
    print(f"  2. 需要引入其他语言维度（话语分析、信息结构、修辞功能等）")
    print(f"  3. 或者承认：引用影响力主要由非文本因素决定")
elif best_auc_overall < 0.65:
    print(f"\n  [CONCLUSION] 存在微弱的分类信号 (最佳AUC={best_auc_overall:.3f})。")
    print(f"  语言特征可以提供有限的指导，但单独使用不够。")
else:
    print(f"\n  [CONCLUSION] 分类效果中等 (最佳AUC={best_auc_overall:.3f})。")
    print(f"  语言特征对区分论文影响力有明显帮助。")

print(f"\n  [OK] 分析完成。图表保存至: {OUTPUT_DIR}")
print("=" * 70)
