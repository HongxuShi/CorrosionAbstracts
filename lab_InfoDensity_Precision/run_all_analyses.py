#!/usr/bin/env python3
"""Info Density: Regression + Classification + Journal + 4-way comparison"""
import warnings, sys, os; from pathlib import Path
import numpy as np; import pandas as pd
from scipy import stats; from scipy.stats import pearsonr
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
from info_density_extractor import InfoDensityExtractor

NCC_PATH = SCRIPT_DIR.parent / "processed" / "merged_all_features.csv"
FEATURE_NAMES = InfoDensityExtractor.FEATURE_NAMES

J_ABV = {
    'Materials and Corrosion':'Mat.Corros.','CORROSION':'CORROSION',
    'Anti-Corrosion Methods and Materials':'Anti-Corr.MM.',
    'Corrosion Engineering Science and Technology The International Journal of Corrosion Processes and Corrosion Control':'Corr.Eng.Sci.',
    'Corrosion Science':'Corr.Sci.','Corrosion and Materials Degradation':'Corr.Mat.Deg.',
}

print("="*70)
print(f" Info Density Analysis: {len(FEATURE_NAMES)} features")
print("="*70)

# 0. Extraction
print("\n[0] Extracting...")
df_ncc = pd.read_csv(NCC_PATH)
ext = InfoDensityExtractor()
df_f = ext.extract_all(df_ncc)
df = df_f.merge(df_ncc[['doi','year','citations','NCC','source_journal']], left_on='doc_id',right_on='doi',how='inner').drop(columns=['doi'])
df = df.dropna(subset=['NCC'])
u,l = df['NCC'].quantile(0.995), df['NCC'].quantile(0.005)
df = df[(df['NCC']>=l)&(df['NCC']<=u)].copy()
df['j'] = df['source_journal'].map(J_ABV)
print(f"  Final: {len(df)} abstracts")
df_f.to_csv(OUTPUT_DIR/'feature_matrix.csv', index=False)

# Remove zero-variance features
variances = df[FEATURE_NAMES].var()
valid_feats = [f for f in FEATURE_NAMES if variances[f] > 1e-8]
removed = [f for f in FEATURE_NAMES if f not in valid_feats]
if removed: print(f"  Removed zero-var: {removed}")

X_all = StandardScaler().fit_transform(df[valid_feats].values)
y_ncc = df['NCC'].values
n,p = X_all.shape
cv_k = KFold(n_splits=5,shuffle=True,random_state=42)
cv_s = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# 1. Regression
print("\n[1] Regression...")
pr = {f:{'r':r,'p':pp} for f in valid_feats for r,pp in [pearsonr(df[f],y_ncc)]}
bf = max(pr, key=lambda k: abs(pr[k]['r']))
ols = LinearRegression().fit(X_all,y_ncc)
yp=ols.predict(X_all)
r2o=1-np.sum((y_ncc-yp)**2)/np.sum((y_ncc-np.mean(y_ncc))**2)
adj=1-(1-r2o)*(n-1)/(n-p-1)
ridge=RidgeCV(alphas=np.logspace(-3,3,50),cv=3).fit(X_all,y_ncc)
rcv=cross_val_score(Ridge(alpha=ridge.alpha_),X_all,y_ncc,cv=cv_k,scoring='r2')
lasso=LassoCV(alphas=np.logspace(-4,1,50),cv=3,max_iter=10000,random_state=42).fit(X_all,y_ncc)
lcv=cross_val_score(Lasso(alpha=lasso.alpha_,max_iter=10000,random_state=42),X_all,y_ncc,cv=cv_k,scoring='r2')
rf_r=RandomForestRegressor(n_estimators=500,max_depth=10,min_samples_leaf=50,random_state=42,n_jobs=-1)
rf_r.fit(X_all,y_ncc); rfcv=cross_val_score(rf_r,X_all,y_ncc,cv=cv_k,scoring='r2')
print(f"  Best |r|: {bf} r={pr[bf]['r']:+.4f}")
print(f"  R^2={r2o:.4f} Adj={adj:.4f} | Ridge={rcv.mean():.4f} | Lasso={lcv.mean():.4f} | RF={rfcv.mean():.4f}")

# 2. Classification
print("\n[2] Classification...")
cr = {}
for sn,hq,lq in [('S1',0.75,0.25),('S2',0.80,0.20),('S4',0.90,0.10)]:
    th, tl = df['NCC'].quantile(hq), df['NCC'].quantile(lq)
    sub = df[(df['NCC']>=th)|(df['NCC']<=tl)].copy()
    ys = (sub['NCC']>=th).astype(int)
    if ys.sum()<10 or len(sub)<100: continue
    Xs = StandardScaler().fit_transform(sub[valid_feats].values)
    rf_c = RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=30,random_state=42,n_jobs=-1)
    nc = min(5,ys.sum(),len(ys)-ys.sum())
    cvu = StratifiedKFold(n_splits=max(3,nc),shuffle=True,random_state=42) if nc>=3 else KFold(n_splits=3,shuffle=True,random_state=42)
    sc = cross_validate(rf_c,Xs,ys,cv=cvu,scoring={'auc':'roc_auc','f1':'f1'},n_jobs=-1)
    cr[sn] = {'auc':sc['test_auc'].mean(),'auc_std':sc['test_auc'].std(),'f1':sc['test_f1'].mean()}
    print(f"  {sn}: AUC={cr[sn]['auc']:.3f}+/-{cr[sn]['auc_std']:.3f}")

# 3. Journal
print("\n[3] Journal...")
jns = sorted(df['source_journal'].unique())
le = LabelEncoder(); yj = le.fit_transform(df['source_journal'])
rf_mc = RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_leaf=30,random_state=42,n_jobs=-1)
mc = cross_validate(rf_mc,X_all,yj,cv=cv_s,scoring={'accuracy':'accuracy','f1_macro':'f1_macro'},n_jobs=-1)
bl_j = max(np.bincount(yj))/len(yj)
rf_mc.fit(X_all,yj)
ovr = {}
for i,jn in enumerate(jns):
    yb=(yj==i).astype(int)
    rf_o=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=30,random_state=42,n_jobs=-1)
    sc=cross_validate(rf_o,X_all,yb,cv=cv_s,scoring={'auc':'roc_auc','f1':'f1'},n_jobs=-1)
    ovr[J_ABV[jn]]={'auc':sc['test_auc'].mean(),'auc_std':sc['test_auc'].std()}
yp_cv=cross_val_predict(rf_mc,X_all,yj,cv=cv_s,n_jobs=-1)
cm=confusion_matrix(yj,yp_cv); cmn=cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
# Pairwise d
pd_list=[]
for j1,j2 in combinations(range(len(jns)),2):
    d1,d2=df[df['source_journal']==jns[j1]],df[df['source_journal']==jns[j2]]
    for f in valid_feats[:5]:
        n1,n2=len(d1),len(d2)
        sp=np.sqrt(((n1-1)*np.var(d1[f],ddof=1)+(n2-1)*np.var(d2[f],ddof=1))/(n1+n2-2))
        d=abs(np.mean(d1[f])-np.mean(d2[f]))/max(sp,1e-10)
        pd_list.append({'j1':J_ABV[jns[j1]],'j2':J_ABV[jns[j2]],'feat':f,'d':d})
pd_list.sort(key=lambda x:x['d'],reverse=True)
bo = max(ovr.items(),key=lambda x:x[1]['auc'])
print(f"  Acc={mc['test_accuracy'].mean():.3f} (bl={bl_j:.3f}) F1={mc['test_f1_macro'].mean():.3f}")
print(f"  Best OvR: {bo[0]} AUC={bo[1]['auc']:.3f}")
print(f"  Best pairwise: {pd_list[0]['j1']} vs {pd_list[0]['j2']} ({pd_list[0]['feat']}) d={pd_list[0]['d']:.3f}")

# 4. 4-way comparison
print("\n[4] 4-Way Comparison...")
id_auc = max(v['auc'] for v in cr.values()) if cr else 0
prev = {
    'Biber':          [0.0063,0.604,0.332,0.019,0.751,0.173],
    'Metadiscourse':  [0.0077,0.592,0.409,0.096,0.874,0.259],
    'Syntactic':      [0.0017,0.551,0.341,0.028,0.717,0.183],
}
now = [r2o, id_auc, mc['test_accuracy'].mean(), mc['test_accuracy'].mean()-bl_j, bo[1]['auc'], mc['test_f1_macro'].mean()]

metrics = ['Reg R^2','Cls AUC','Jrn Acc','Acc-BL','Jrn OvR','Jrn F1']
print(f"  {'Metric':<12s} {'Biber':>8s} {'Meta':>8s} {'Syn':>8s} {'InfoD':>8s}  {'Best':>8s}")
for i,(m,b,mt,s) in enumerate(zip(metrics, prev['Biber'], prev['Metadiscourse'], prev['Syntactic'])):
    vals = {'Biber':b,'Meta':mt,'Syn':s,'InfoD':now[i]}
    best = max(vals,key=vals.get) if 'R^2' not in m and 'AUC' not in m and 'OvR' not in m and 'F1' not in m else max(vals,key=vals.get)
    # For Acc-BL and R^2, higher is better
    best_method = max(vals, key=vals.get)
    print(f"  {m:<12s} {b:8.4f} {mt:8.4f} {s:8.4f} {now[i]:8.4f}  -> {best_method}")

print(f"\n  Info Density ranking:")
acc_bls = {'Biber':0.019,'Meta':0.096,'Syn':0.028,'InfoD':now[3]}
for m,v in sorted(acc_bls.items(),key=lambda x:x[1],reverse=True):
    print(f"    {m}: {v:+.3f}")

# 5. Charts
print("\n[5] Charts...")

# Fig 1: Pairwise heatmap
fig,ax=plt.subplots(figsize=(8,7))
dmat=np.zeros((4,4)); lbls=['Biber','Meta','Syn','InfoD']
for i,j in combinations(range(4),2):
    dmat[i,j]=abs(prev[['Biber','Metadiscourse','Syntactic'][j-1] if j>0 else 'Biber'][3] if j>0 else prev['Biber'][3])
# Simplified: just show journal Acc-BL comparison
acc_data = [0.019,0.096,0.028,now[3]]
bars=ax.bar(['Biber\n(15+4PCA)','Metadiscourse\n(10 feat)','Syntactic\n(13 feat)','Info Density\n(10 feat)'],
           acc_data,color=['#4575b4','#d73027','#66bd63','#fc8d59'],edgecolor='white',linewidth=2)
for bar,v in zip(bars,acc_data): ax.text(bar.get_x()+bar.get_width()/2, v+0.003 if v>=0 else v-0.015, f'{v:+.3f}', ha='center',fontweight='bold')
ax.axhline(0,color='black',linewidth=0.5)
ax.set_title('Journal Classification: Accuracy Above Baseline\n(4 Schemes Comparison)',fontsize=14)
ax.set_ylabel('Accuracy - Baseline')
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_i1_comparison.png',bbox_inches='tight'); plt.close()
print("  [OK] fig_i1_comparison.png")

# Fig 2: Feature importance
fig,axes=plt.subplots(1,2,figsize=(14,5))
rf_imp=pd.DataFrame({'f':valid_feats,'i':rf_r.feature_importances_}).sort_values('i')
axes[0].barh(rf_imp['f'],rf_imp['i'],color='forestgreen',edgecolor='white')
axes[0].set_title('RF Importance (NCC Regression)')
cv2=[(f,pr[f]['r']) for f in valid_feats]; cv2.sort(key=lambda x:abs(x[1]))
axes[1].barh([c[0] for c in cv2],[c[1] for c in cv2],color=['#d73027' if v<0 else '#4575b4' for _,v in cv2],edgecolor='white')
axes[1].axvline(0,color='black',linewidth=0.5)
axes[1].set_title('Pearson r with NCC')
fig.suptitle('Info Density Features',fontsize=14)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_i2_importance.png',bbox_inches='tight'); plt.close()
print("  [OK] fig_i2_importance.png")

summary = {'n':n,'p':p,'r2':r2o,'adj_r2':adj,'rcv':rcv.mean(),'lcv':lcv.mean(),'rfcv':rfcv.mean(),'best_r':pr[bf]['r'],'best_f':bf,'best_auc':id_auc,'jrn_acc':mc['test_accuracy'].mean(),'jrn_bl':bl_j,'jrn_f1':mc['test_f1_macro'].mean(),'jrn_ovr':bo[1]['auc']}
pd.DataFrame([summary]).to_csv(OUTPUT_DIR/'results_summary.csv',index=False)
print(f"\n[OK] Done. {OUTPUT_DIR}")
