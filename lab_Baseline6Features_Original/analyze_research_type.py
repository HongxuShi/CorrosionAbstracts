#!/usr/bin/env python3
"""
研究类型分类分析
Research Type Classification — Auto-labeling + Feature-based Classification

自动标签策略 (基于标题+摘要关键词):
  - review: 综述/回顾类 (review, survey, overview, state of the art, critical review)
  - experimental: 实验报告类 (investigation, study, effect, behavior, measurement, test)
  - modeling: 建模/计算类 (model, simulation, computational, theoretical, prediction, DFT, FEM)

分析:
  1. 三类论文的分布
  2. 特征能否区分研究类型
  3. 各类型的特征差异 (ANOVA + Cohen's d)
  4. NCC在三类之间的差异
"""
import sys, re, warnings; from pathlib import Path
import numpy as np; import pandas as pd
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from itertools import combinations
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'font.size':10})

if len(sys.argv) < 3:
    print("Usage: python analyze_research_type.py <feature_csv> <output_dir>")
    sys.exit(1)

FEATURE_CSV = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2]); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NCC_PATH = Path(__file__).resolve().parent.parent / "processed" / "merged_all_features.csv"

print(f"Research Type Analysis: {FEATURE_CSV.name} -> {OUTPUT_DIR}")

df_f = pd.read_csv(FEATURE_CSV)
df_ncc = pd.read_csv(NCC_PATH)
df = df_f.merge(df_ncc[['doi','title','abstract','year','NCC','source_journal']], left_on='doc_id',right_on='doi',how='inner')

feat_cols = [c for c in df_f.columns if c not in ['doc_id'] and np.issubdtype(df_f[c].dtype, np.number)]

# === 1. Auto-label research types ===
print("\n[1] Auto-labeling research types...")

def classify_research_type(title, abstract):
    text = str(title).lower() + ' ' + str(abstract).lower()

    # Review indicators
    review_kw = ['review', 'survey', 'overview', 'state of the art', 'state-of-the-art',
                 'critical review', 'literature review', 'comprehensive review',
                 'systematic review', 'meta-analysis', 'retrospective']
    # Modeling indicators
    model_kw = ['model', 'simulation', 'computational', 'theoretical', 'prediction',
                'dft', 'finite element', 'molecular dynamics', 'monte carlo',
                'neural network', 'machine learning', 'artificial intelligence',
                'first-principles', 'ab initio', 'density functional',
                'numerical', 'mathematical model', 'kinetic model', 'thermodynamic',
                'phase field', 'calculation', 'calculated']
    # Experimental indicators
    exp_kw = ['investigation', 'experimental', 'measurement', 'effect of', 'behavior of',
              'influence of', 'impact of', 'role of', 'study of', 'analysis of',
              'characterization', 'microstructure', 'morphology', 'scanning electron',
              'x-ray diffraction', 'xrd', 'sem', 'tem', 'eds', 'xps', 'raman',
              'electrochemical impedance', 'polarization', 'weight loss',
              'potentiodynamic', 'open circuit', 'salt spray', 'immersion test',
              'corrosion rate', 'corrosion resistance', 'corrosion behavior',
              'inhibition efficiency', 'coating', 'inhibitor', 'synthesized',
              'fabricated', 'prepared by', 'deposited', 'specimen', 'sample',
              'tested', 'evaluated', 'assessed', 'measured', 'determined', 'examined']

    review_score = sum(1 for kw in review_kw if kw in text)
    model_score = sum(1 for kw in model_kw if kw in text)
    exp_score = sum(1 for kw in exp_kw if kw in text)

    scores = {'review': review_score, 'modeling': model_score, 'experimental': exp_score}
    max_type = max(scores, key=scores.get)

    # If no clear signal, default to experimental (most common)
    if scores[max_type] == 0:
        return 'experimental'

    # If review has strong signal, override
    if review_score >= 2:
        return 'review'

    return max_type

df['research_type'] = df.apply(lambda r: classify_research_type(r['title'], r['abstract']), axis=1)
type_counts = df['research_type'].value_counts()
print(f"  Distribution: {dict(type_counts)}")

# === 2. Type classification ===
print("\n[2] Type Classification...")
le = LabelEncoder(); y_type = le.fit_transform(df['research_type'])
X_type = StandardScaler().fit_transform(df[feat_cols].values)
n_types = len(type_counts)

if n_types >= 2 and min(type_counts) >= 5:
    cv_s = StratifiedKFold(n_splits=min(5, min(type_counts)), shuffle=True, random_state=42)
    rf_c = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1)
    sc = cross_validate(rf_c, X_type, y_type, cv=cv_s, scoring={'accuracy':'accuracy','f1_macro':'f1_macro'}, n_jobs=-1)
    bl_type = max(np.bincount(y_type))/len(y_type)
    print(f"  Acc={sc['test_accuracy'].mean():.3f} (bl={bl_type:.3f}) F1={sc['test_f1_macro'].mean():.3f}")

    # Confusion matrix
    rf_c.fit(X_type, y_type)
    yp = cross_val_predict(rf_c, X_type, y_type, cv=cv_s, n_jobs=-1)
    cm = confusion_matrix(y_type, yp)
    cm_n = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
    for i, tn in enumerate(le.classes_):
        print(f"    {tn:15s} recall={cm_n[i,i]:.2f}")

    # Feature importance
    rf_imp = pd.DataFrame({'f':feat_cols,'i':rf_c.feature_importances_}).sort_values('i',ascending=False)
    print(f"  Top features: {', '.join(f'{r.f}({r.i:.3f})' for _,r in rf_imp.head(5).iterrows())}")
else:
    sc = {'test_accuracy': np.array([np.nan]), 'test_f1_macro': np.array([np.nan])}
    bl_type = np.nan
    cm_n = np.array([])

# === 3. NCC by research type ===
print("\n[3] NCC by Research Type...")
for rt in sorted(df['research_type'].unique()):
    subset = df[df['research_type'] == rt]
    print(f"  {rt:15s}: NCC mean={subset['NCC'].mean():.3f} std={subset['NCC'].std():.3f} n={len(subset)}")

# ANOVA: NCC across types
groups = [df[df['research_type']==rt]['NCC'].values for rt in sorted(df['research_type'].unique())]
if len(groups) >= 2:
    f_stat, anova_p = f_oneway(*groups)
    print(f"  ANOVA: F={f_stat:.2f} p={anova_p:.4f}")

# === 4. Per-type feature profiles ===
print("\n[4] Per-type feature deviations (z-score)...")
for rt in sorted(df['research_type'].unique()):
    sdf = df[df['research_type']==rt]
    print(f"  {rt:15s}:", end='')
    for f in feat_cols[:6]:
        z = (sdf[f].mean()-df[f].mean())/df[f].std()
        if abs(z) > 0.1:
            print(f" {f}={z:+.2f}", end='')
    print()

# === 5. Chart ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Type distribution
types_sorted = sorted(type_counts.index)
axes[0].bar(types_sorted, [type_counts[t] for t in types_sorted],
            color=['#4575b4','#d73027','#66bd63'], edgecolor='white')
axes[0].set_title(f'Research Type Distribution (N={len(df)})')
axes[0].set_ylabel('Count')
for i,t in enumerate(types_sorted): axes[0].text(i, type_counts[t]+10, str(type_counts[t]), ha='center')

# NCC by type
ncc_by_type = [df[df['research_type']==t]['NCC'].mean() for t in types_sorted]
axes[1].bar(types_sorted, ncc_by_type, color=['#4575b4','#d73027','#66bd63'], edgecolor='white')
axes[1].set_title('Mean NCC by Research Type')
for i,v in enumerate(ncc_by_type): axes[1].text(i, v+0.02, f'{v:.3f}', ha='center')
fig.suptitle(f'Research Type Analysis ({FEATURE_CSV.parent.name})', fontsize=14)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_research_type.png',bbox_inches='tight'); plt.close()

# Save
results = {
    'scheme': FEATURE_CSV.parent.name,
    'n_types': n_types,
    'type_distribution': str(dict(type_counts)),
    'type_acc': sc['test_accuracy'].mean() if len(sc['test_accuracy'])>0 else np.nan,
    'type_bl': bl_type,
    'type_f1': sc['test_f1_macro'].mean() if len(sc['test_f1_macro'])>0 else np.nan,
    'ncc_anova_f': f_stat if len(groups)>=2 else np.nan,
    'ncc_anova_p': anova_p if len(groups)>=2 else np.nan,
}
pd.DataFrame([results]).to_csv(OUTPUT_DIR/'results_research_type.csv', index=False)
print(f"\n[OK] Done.")
