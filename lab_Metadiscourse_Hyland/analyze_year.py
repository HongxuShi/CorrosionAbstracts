#!/usr/bin/env python3
"""
年代分析: 特征 → 发表年份 (回归 + 近/远期分类)
Year Analysis: Features -> Publication Year (Regression + Early/Late Classification)

用法: python analyze_year.py <feature_csv> <output_dir>
"""
import sys, warnings; from pathlib import Path
import numpy as np; import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'font.size':10})

if len(sys.argv) < 3:
    print("Usage: python analyze_year.py <feature_csv> <output_dir>")
    sys.exit(1)

FEATURE_CSV = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2]); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NCC_PATH = Path(__file__).resolve().parent.parent / "processed" / "merged_all_features.csv"

print(f"Year Analysis: {FEATURE_CSV.name} -> {OUTPUT_DIR}")

# Load
df_f = pd.read_csv(FEATURE_CSV)
df_ncc = pd.read_csv(NCC_PATH)
df = df_f.merge(df_ncc[['doi','year','NCC','source_journal']], left_on='doc_id',right_on='doi',how='inner')
df = df.dropna(subset=['year']).copy()

feat_cols = [c for c in df_f.columns if c not in ['doc_id'] and np.issubdtype(df_f[c].dtype, np.number)]
if not feat_cols: feat_cols = [c for c in df_f.columns if c != 'doc_id']
print(f"  Features: {len(feat_cols)}, Samples: {len(df)}")

X = StandardScaler().fit_transform(df[feat_cols].values)
y_year = df['year'].values
n = len(df)
cv_k = KFold(n_splits=5, shuffle=True, random_state=42)

# === 1. Regression: features -> year ===
print("\n[1] Year Regression...")
pr = {f:{'r':r,'p':p} for f in feat_cols for r,p in [pearsonr(df[f],y_year)]}
bf = max(pr,key=lambda k:abs(pr[k]['r']))

ols = LinearRegression().fit(X, y_year)
yp = ols.predict(X); r2 = 1-np.sum((y_year-yp)**2)/np.sum((y_year-np.mean(y_year))**2)
adj = 1-(1-r2)*(n-1)/(n-len(feat_cols)-1)

ridge = RidgeCV(alphas=np.logspace(-3,3,50),cv=3).fit(X,y_year)
rcv = cross_val_score(Ridge(alpha=ridge.alpha_),X,y_year,cv=cv_k,scoring='r2')

lasso = LassoCV(alphas=np.logspace(-4,1,50),cv=3,max_iter=10000,random_state=42).fit(X,y_year)
lcv = cross_val_score(Lasso(alpha=lasso.alpha_,max_iter=10000,random_state=42),X,y_year,cv=cv_k,scoring='r2')

rf_r = RandomForestRegressor(n_estimators=500,max_depth=10,min_samples_leaf=50,random_state=42,n_jobs=-1)
rf_r.fit(X,y_year); rfcv = cross_val_score(rf_r,X,y_year,cv=cv_k,scoring='r2')

# MAE (Mean Absolute Error — years)
from sklearn.metrics import mean_absolute_error
mae_cv = cross_val_score(rf_r, X, y_year, cv=cv_k, scoring='neg_mean_absolute_error')

print(f"  Best |r|: {bf} r={pr[bf]['r']:+.4f}")
print(f"  R^2={r2:.4f} Adj={adj:.4f} | Ridge={rcv.mean():.4f} | Lasso={lcv.mean():.4f} | RF={rfcv.mean():.4f}")
print(f"  MAE (RF CV): {-mae_cv.mean():.1f} +/- {mae_cv.std():.1f} years")

# === 2. Classification: Early (<=2018) vs Late (>=2021) ===
print("\n[2] Early(<=2018) vs Late(>=2021) Classification...")
median_year = df['year'].median()
df['era'] = 'mid'
df.loc[df['year'] <= 2018, 'era'] = 'early'
df.loc[df['year'] >= 2021, 'era'] = 'late'
era_df = df[df['era'].isin(['early','late'])].copy()
y_era = (era_df['era'] == 'late').astype(int)
X_era = StandardScaler().fit_transform(era_df[feat_cols].values)

cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_c = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1)
era_scores = cross_validate(rf_c, X_era, y_era, cv=cv_s, scoring={'auc':'roc_auc','f1':'f1'}, n_jobs=-1)
bl_era = max(y_era.mean(), 1-y_era.mean())

print(f"  Early={sum(y_era==0)}, Late={sum(y_era==1)}")
print(f"  AUC={era_scores['test_auc'].mean():.3f}+/-{era_scores['test_auc'].std():.3f} (baseline={bl_era:.3f})")
print(f"  F1={era_scores['test_f1'].mean():.3f}")

# === 3. Per-feature year trend ===
print("\n[3] Top features by year correlation:")
for f, v in sorted(pr.items(), key=lambda x: abs(x[1]['r']), reverse=True)[:5]:
    direction = 'increasing' if v['r'] > 0 else 'decreasing'
    print(f"  {f:25s} r={v['r']:+.4f} p={v['p']:.4f} -> {direction} over time")

# === 4. Chart ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Feature-year correlation
items = sorted(pr.items(), key=lambda x: abs(x[1]['r']))
axes[0].barh([i[0] for i in items], [i[1]['r'] for i in items],
             color=['#d73027' if v['r']<0 else '#4575b4' for _,v in items], edgecolor='white')
axes[0].axvline(0, color='black', linewidth=0.5)
axes[0].set_title('Feature-Year Pearson r')
# Year distribution by era
era_counts = df['year'].value_counts().sort_index()
axes[1].bar(era_counts.index, era_counts.values, color='steelblue', edgecolor='white')
axes[1].axvline(2018.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].axvline(2020.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].set_title(f'Year Distribution (N={len(df)})')
axes[1].set_xlabel('Year')
fig.suptitle(f'Year Analysis ({FEATURE_CSV.parent.name})', fontsize=14)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_year.png',bbox_inches='tight'); plt.close()

# === Summary ===
results = {
    'scheme': FEATURE_CSV.parent.name,
    'n_samples': n, 'n_features': len(feat_cols),
    'year_r2_ols': r2, 'year_adj_r2': adj,
    'year_ridge_cv_r2': rcv.mean(), 'year_lasso_cv_r2': lcv.mean(),
    'year_rf_cv_r2': rfcv.mean(), 'year_mae_years': -mae_cv.mean(),
    'year_best_feature': bf, 'year_best_r': pr[bf]['r'],
    'era_auc': era_scores['test_auc'].mean(), 'era_baseline': bl_era,
    'era_f1': era_scores['test_f1'].mean(),
}
pd.DataFrame([results]).to_csv(OUTPUT_DIR/'results_year.csv', index=False)
print(f"\n[OK] Results saved to {OUTPUT_DIR}")
