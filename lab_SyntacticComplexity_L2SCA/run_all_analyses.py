#!/usr/bin/env python3
"""
句法复杂度三实验分析 + 与前两个方案的对比
Syntactic Complexity: Regression + Classification + Journal + 3-way comparison
"""
import warnings, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from itertools import combinations

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'font.size':10})

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR))
from syntactic_extractor import SyntacticComplexityExtractor

NCC_PATH = SCRIPT_DIR.parent / "processed" / "merged_all_features.csv"
FEATURE_NAMES = SyntacticComplexityExtractor.FEATURE_NAMES
# Remove zero-variance feature
FEATURE_NAMES = [f for f in FEATURE_NAMES if f != 'T_S']

J_ABV = {
    'Materials and Corrosion':'Mat.Corros.','CORROSION':'CORROSION',
    'Anti-Corrosion Methods and Materials':'Anti-Corr.MM.',
    'Corrosion Engineering Science and Technology The International Journal of Corrosion Processes and Corrosion Control':'Corr.Eng.Sci.',
    'Corrosion Science':'Corr.Sci.','Corrosion and Materials Degradation':'Corr.Mat.Deg.',
}

print("="*70)
print(f" Syntactic Complexity Analysis: {len(FEATURE_NAMES)} features")
print("="*70)

# ===================================================================
# 0. Extraction
# ===================================================================
print("\n[0] Extracting features...")
df_ncc = pd.read_csv(NCC_PATH)
extractor = SyntacticComplexityExtractor()
df_f = extractor.extract_all(df_ncc)
print(f"  Shape: {df_f.shape}")

df = df_f.merge(df_ncc[['doi','year','citations','NCC','source_journal']], left_on='doc_id',right_on='doi',how='inner').drop(columns=['doi'])
df = df.dropna(subset=['NCC'])
ncc_u, ncc_l = df['NCC'].quantile(0.995), df['NCC'].quantile(0.005)
df = df[(df['NCC']>=ncc_l)&(df['NCC']<=ncc_u)].copy()
df['j'] = df['source_journal'].map(J_ABV)
print(f"  Final: {len(df)} abstracts")

df_f.to_csv(OUTPUT_DIR/'feature_matrix.csv', index=False)

X_all = StandardScaler().fit_transform(df[FEATURE_NAMES].values)
y_ncc = df['NCC'].values
n, p = X_all.shape
cv_k = KFold(n_splits=5, shuffle=True, random_state=42)
cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===================================================================
# 1. Descriptive
# ===================================================================
print("\n[1] Descriptive...")
for f in sorted(FEATURE_NAMES, key=lambda x: df[x].mean(), reverse=True)[:5]:
    print(f"  {f:10s} mean={df[f].mean():.3f} std={df[f].std():.3f}")

# ===================================================================
# 2. Regression
# ===================================================================
print("\n[2] Regression: Features -> NCC...")
pearson_res = {f: {'r':r,'p':p} for f in FEATURE_NAMES for r,p in [pearsonr(df[f], y_ncc)]}
best_f = max(pearson_res, key=lambda k: abs(pearson_res[k]['r']))

ols = LinearRegression().fit(X_all, y_ncc)
yp = ols.predict(X_all)
r2_o = 1 - np.sum((y_ncc-yp)**2)/np.sum((y_ncc-np.mean(y_ncc))**2)
adjr2 = 1-(1-r2_o)*(n-1)/(n-p-1)

ridge = RidgeCV(alphas=np.logspace(-3,3,50),cv=3).fit(X_all,y_ncc)
r_cv = cross_val_score(Ridge(alpha=ridge.alpha_),X_all,y_ncc,cv=cv_k,scoring='r2')

lasso = LassoCV(alphas=np.logspace(-4,1,50),cv=3,max_iter=10000,random_state=42).fit(X_all,y_ncc)
l_cv = cross_val_score(Lasso(alpha=lasso.alpha_,max_iter=10000,random_state=42),X_all,y_ncc,cv=cv_k,scoring='r2')

rf_r = RandomForestRegressor(n_estimators=500,max_depth=10,min_samples_leaf=50,random_state=42,n_jobs=-1)
rf_r.fit(X_all,y_ncc)
rf_cv = cross_val_score(rf_r,X_all,y_ncc,cv=cv_k,scoring='r2')
rf_imp = pd.DataFrame({'feature':FEATURE_NAMES,'importance':rf_r.feature_importances_}).sort_values('importance',ascending=False)

print(f"  Best |r|: {best_f} r={pearson_res[best_f]['r']:+.4f}")
print(f"  OLS R^2={r2_o:.4f} Adj={adjr2:.4f} | Ridge CV={r_cv.mean():.4f} | Lasso CV={l_cv.mean():.4f} | RF CV={rf_cv.mean():.4f}")

# ===================================================================
# 3. Classification
# ===================================================================
print("\n[3] Classification...")
cls_res = {}
for strat_name, hi_q, lo_q in [('S1_Top25vsBot25',0.75,0.25),('S2_Top20vsBot20',0.80,0.20),('S4_Top10vsBot10',0.90,0.10)]:
    th_hi, th_lo = df['NCC'].quantile(hi_q), df['NCC'].quantile(lo_q)
    sub = df[(df['NCC']>=th_hi)|(df['NCC']<=th_lo)].copy()
    ys = (sub['NCC']>=th_hi).astype(int)
    if ys.sum()<10 or len(sub)<100: continue
    Xs = StandardScaler().fit_transform(sub[FEATURE_NAMES].values)
    n_cv = min(5,ys.sum(),len(ys)-ys.sum())
    cvu = StratifiedKFold(n_splits=max(3,n_cv),shuffle=True,random_state=42) if n_cv>=3 else KFold(n_splits=3,shuffle=True,random_state=42)
    rf_c = RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=30,random_state=42,n_jobs=-1)
    sc = cross_validate(rf_c,Xs,ys,cv=cvu,scoring={'auc':'roc_auc','f1':'f1'},n_jobs=-1)
    cls_res[strat_name] = {'auc':sc['test_auc'].mean(),'auc_std':sc['test_auc'].std(),'f1':sc['test_f1'].mean(),'n':len(sub),'baseline':max(ys.mean(),1-ys.mean())}
    print(f"  {strat_name}: AUC={cls_res[strat_name]['auc']:.3f}+/-{cls_res[strat_name]['auc_std']:.3f}")

# ===================================================================
# 4. Journal Classification
# ===================================================================
print("\n[4] Journal Classification...")
journals = sorted(df['source_journal'].unique())
le = LabelEncoder(); yj = le.fit_transform(df['source_journal'])
n_j = len(journals)

rf_mc = RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_leaf=30,random_state=42,n_jobs=-1)
mc = cross_validate(rf_mc,X_all,yj,cv=cv_s,scoring={'accuracy':'accuracy','f1_macro':'f1_macro'},n_jobs=-1)
bl = max(np.bincount(yj))/len(yj)

rf_mc.fit(X_all,yj)
ovr_r = {}
for i,jn in enumerate(journals):
    yb = (yj==i).astype(int)
    rf_o = RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=30,random_state=42,n_jobs=-1)
    sc = cross_validate(rf_o,X_all,yb,cv=cv_s,scoring={'auc':'roc_auc','f1':'f1'},n_jobs=-1)
    ovr_r[J_ABV[jn]] = {'auc':sc['test_auc'].mean(),'auc_std':sc['test_auc'].std(),'n':(yj==i).sum()}

yp_cv = cross_val_predict(rf_mc,X_all,yj,cv=cv_s,n_jobs=-1)
cm = confusion_matrix(yj,yp_cv); cm_n = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]

pair_d = []
for j1,j2 in combinations(range(n_j),2):
    d1,d2 = df[df['source_journal']==journals[j1]],df[df['source_journal']==journals[j2]]
    for f in ['MLC','CN_C','CN_T','C_T']:
        n1,n2 = len(d1),len(d2)
        sp=np.sqrt(((n1-1)*np.var(d1[f],ddof=1)+(n2-1)*np.var(d2[f],ddof=1))/(n1+n2-2))
        d=abs(np.mean(d1[f])-np.mean(d2[f]))/max(sp,1e-10)
        pair_d.append({'j1':J_ABV[journals[j1]],'j2':J_ABV[journals[j2]],'feat':f,'d':d})
pair_d.sort(key=lambda x:x['d'],reverse=True)

best_ovr = max(ovr_r.items(),key=lambda x:x[1]['auc'])
print(f"  Acc={mc['test_accuracy'].mean():.3f} (bl={bl:.3f}) F1={mc['test_f1_macro'].mean():.3f}")
print(f"  Best OvR: {best_ovr[0]} AUC={best_ovr[1]['auc']:.3f}")
print(f"  Best pairwise: {pair_d[0]['j1']} vs {pair_d[0]['j2']} ({pair_d[0]['feat']}) d={pair_d[0]['d']:.3f}")

# ===================================================================
# 5. 3-way Comparison
# ===================================================================
print("\n[5] 3-Way Comparison (Biber vs Metadiscourse vs Syntactic)...")
prev_results = {
    'Biber':       {'r2':0.0063,'auc':0.604,'acc':0.332,'bl':0.313,'ovr':0.751,'f1':0.173},
    'Metadiscourse':{'r2':0.0077,'auc':0.592,'acc':0.409,'bl':0.313,'ovr':0.874,'f1':0.259},
}
syn_auc = max(v['auc'] for v in cls_res.values()) if cls_res else 0
syn_ovr = best_ovr[1]['auc']

comp = pd.DataFrame({
    'Metric': ['Regression R^2','Classif. Best AUC','Journal Acc','Acc-Baseline','Journal Best OvR AUC','Journal F1_macro'],
    'Biber':       [0.0063,0.604,0.332,0.019,0.751,0.173],
    'Metadiscourse':[0.0077,0.592,0.409,0.096,0.874,0.259],
    'Syntactic':   [r2_o, syn_auc, mc['test_accuracy'].mean(), mc['test_accuracy'].mean()-bl, syn_ovr, mc['test_f1_macro'].mean()],
})
print(comp.round(4).to_string(index=False))

# Determine overall ranking
print(f"\n  Journal performance ranking:")
acc_bl = {'Biber':0.019,'Metadiscourse':0.096,'Syntactic':mc['test_accuracy'].mean()-bl}
for m,v in sorted(acc_bl.items(),key=lambda x:x[1],reverse=True):
    print(f"    {m}: Acc-Baseline = {v:+.3f}")

# ===================================================================
# 6. Visualizations
# ===================================================================
print("\n[6] Charts...")

# Fig 1: Feature importance
fig,axes=plt.subplots(1,2,figsize=(14,5))
rf_s = rf_imp.sort_values('importance')
axes[0].barh(rf_s['feature'],rf_s['importance'],color='forestgreen',edgecolor='white')
axes[0].set_title('RF Feature Importance (NCC Regression)')
corr_v = [(f,pearson_res[f]['r']) for f in FEATURE_NAMES]; corr_v.sort(key=lambda x:abs(x[1]))
axes[1].barh([c[0] for c in corr_v],[c[1] for c in corr_v],color=['#d73027' if v<0 else '#4575b4' for _,v in corr_v],edgecolor='white')
axes[1].axvline(0,color='black',linewidth=0.5)
axes[1].set_title('Pearson r with NCC')
fig.suptitle('Syntactic Complexity Features',fontsize=14)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_s1_importance.png',bbox_inches='tight'); plt.close()
print("  [OK] fig_s1_importance.png")

# Fig 2: Journal OvR AUC
fig,ax=plt.subplots(figsize=(10,5))
ovr_i = sorted(ovr_r.items(),key=lambda x:x[1]['auc'],reverse=True)
nms,aus,sts = zip(*[(k,v['auc'],v['auc_std']) for k,v in ovr_i])
cs = ['#d73027' if a>0.75 else '#fc8d59' if a>0.65 else '#4575b4' for a in aus]
ax.barh(range(len(nms)),aus,xerr=sts,color=cs,edgecolor='white',linewidth=1.5,capsize=3)
ax.set_yticks(range(len(nms))); ax.set_yticklabels(nms,fontsize=11)
ax.axvline(0.5,color='gray',linewidth=0.8,linestyle='--')
ax.set_xlabel('AUC-ROC'); ax.set_title('One-vs-Rest Journal Classification: Syntactic Complexity',fontsize=14)
for i,(n,a) in enumerate(zip(nms,aus)): ax.text(a+0.005,i,f'{a:.3f}',va='center',fontsize=9)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_s2_journal_auc.png',bbox_inches='tight'); plt.close()
print("  [OK] fig_s2_journal_auc.png")

# Fig 3: 3-way comparison bar chart
fig,axes=plt.subplots(1,3,figsize=(16,5))
methods=['Biber','Metadiscourse','Syntactic']
colors=['#4575b4','#fc8d59','#66bd63']
# R^2
r2s = [prev_results['Biber']['r2'],prev_results['Metadiscourse']['r2'],r2_o]
axes[0].bar(methods,r2s,color=colors,edgecolor='white',linewidth=2)
for i,v in enumerate(r2s): axes[0].text(i,v+0.0005,f'{v:.4f}',ha='center',fontsize=11,fontweight='bold')
axes[0].set_title('Regression R^2')
# AUC
aucs = [prev_results['Biber']['auc'],prev_results['Metadiscourse']['auc'],syn_auc]
axes[1].bar(methods,aucs,color=colors,edgecolor='white',linewidth=2)
axes[1].axhline(0.5,color='gray',linewidth=0.8,linestyle='--')
for i,v in enumerate(aucs): axes[1].text(i,v+0.01,f'{v:.3f}',ha='center',fontsize=11,fontweight='bold')
axes[1].set_title('Classification AUC')
# Journal Acc-Baseline
accs = [0.019,0.096,mc['test_accuracy'].mean()-bl]
axes[2].bar(methods,accs,color=colors,edgecolor='white',linewidth=2)
axes[2].axhline(0,color='black',linewidth=0.5)
for i,v in enumerate(accs): axes[2].text(i,v+0.003 if v>=0 else v-0.012,f'{v:+.3f}',ha='center',fontsize=11,fontweight='bold')
axes[2].set_title('Journal Acc Above Baseline')
fig.suptitle('3-Way Comparison: Biber vs Metadiscourse vs Syntactic Complexity',fontsize=15,y=1.02)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_s3_threeway.png',bbox_inches='tight'); plt.close()
print("  [OK] fig_s3_threeway.png")

# Save summary
summary = {'n_samples':n,'n_features':p,'r2_ols':r2_o,'adj_r2':adjr2,
           'ridge_cv_r2':r_cv.mean(),'lasso_cv_r2':l_cv.mean(),'rf_cv_r2':rf_cv.mean(),
           'best_pearson_r':pearson_res[best_f]['r'],'best_feature':best_f,
           'best_class_auc':syn_auc,'journal_acc':mc['test_accuracy'].mean(),
           'journal_bl':bl,'journal_f1':mc['test_f1_macro'].mean(),'journal_best_ovr':syn_ovr}
pd.DataFrame([summary]).to_csv(OUTPUT_DIR/'results_summary.csv',index=False)

print(f"\n[OK] Done. Output: {OUTPUT_DIR}")
print("="*70)
