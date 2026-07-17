#!/usr/bin/env python3
"""Baseline 6 Features: Regression + Classification + Journal + 5-way comparison"""
import warnings, sys; from pathlib import Path
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
NCC_PATH = SCRIPT_DIR.parent / "processed" / "merged_all_features.csv"
FEATURES = ['ASL','MWL','LD','LC','JD','HD']

J_ABV = {
    'Materials and Corrosion':'Mat.Corros.','CORROSION':'CORROSION',
    'Anti-Corrosion Methods and Materials':'Anti-Corr.MM.',
    'Corrosion Engineering Science and Technology The International Journal of Corrosion Processes and Corrosion Control':'Corr.Eng.Sci.',
    'Corrosion Science':'Corr.Sci.','Corrosion and Materials Degradation':'Corr.Mat.Deg.',
}

print("="*70)
print(f" Baseline 6 Features: {FEATURES}")
print("="*70)

# 0. Load
print("\n[0] Loading...")
df_ncc = pd.read_csv(NCC_PATH)
df = df_ncc.dropna(subset=['NCC']).copy()
u,l = df['NCC'].quantile(0.995), df['NCC'].quantile(0.005)
df = df[(df['NCC']>=l)&(df['NCC']<=u)].copy()
df = df.dropna(subset=FEATURES)
df['j'] = df['source_journal'].map(J_ABV)
print(f"  {len(df)} abstracts, 6 features")

# Remove extreme outliers in features (99.9th percentile clipping)
for f in FEATURES:
    hi = df[f].quantile(0.999)
    lo = df[f].quantile(0.001)
    df = df[(df[f]>=lo)&(df[f]<=hi)]

X_all = StandardScaler().fit_transform(df[FEATURES].values)
y_ncc = df['NCC'].values
n,p = X_all.shape
cv_k = KFold(n_splits=5,shuffle=True,random_state=42)
cv_s = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

print(f"  After clipping: {len(df)} abstracts")

# 1. Regression
print("\n[1] Regression -> NCC...")
pr = {f:{'r':r,'p':pp} for f in FEATURES for r,pp in [pearsonr(df[f],y_ncc)]}
bf = max(pr,key=lambda k:abs(pr[k]['r']))
ols=LinearRegression().fit(X_all,y_ncc)
yp=ols.predict(X_all); r2o=1-np.sum((y_ncc-yp)**2)/np.sum((y_ncc-np.mean(y_ncc))**2)
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
print("\n[2] Classification -> High/Low Impact...")
cr={}
for sn,hq,lq in [('S1',0.75,0.25),('S2',0.80,0.20),('S4',0.90,0.10)]:
    th,tl=df['NCC'].quantile(hq),df['NCC'].quantile(lq)
    sub=df[(df['NCC']>=th)|(df['NCC']<=tl)].copy()
    ys=(sub['NCC']>=th).astype(int)
    if ys.sum()<10 or len(sub)<100: continue
    Xs=StandardScaler().fit_transform(sub[FEATURES].values)
    rf_c=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=30,random_state=42,n_jobs=-1)
    nc=min(5,ys.sum(),len(ys)-ys.sum())
    cvu=StratifiedKFold(n_splits=max(3,nc),shuffle=True,random_state=42) if nc>=3 else KFold(n_splits=3,shuffle=True,random_state=42)
    sc=cross_validate(rf_c,Xs,ys,cv=cvu,scoring={'auc':'roc_auc','f1':'f1'},n_jobs=-1)
    cr[sn]={'auc':sc['test_auc'].mean(),'auc_std':sc['test_auc'].std()}
    print(f"  {sn}: AUC={cr[sn]['auc']:.3f}+/-{cr[sn]['auc_std']:.3f}")

# 3. Journal
print("\n[3] Journal Classification...")
jns=sorted(df['source_journal'].unique())
le=LabelEncoder(); yj=le.fit_transform(df['source_journal'])
rf_mc=RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_leaf=30,random_state=42,n_jobs=-1)
mc=cross_validate(rf_mc,X_all,yj,cv=cv_s,scoring={'accuracy':'accuracy','f1_macro':'f1_macro'},n_jobs=-1)
bl_j=max(np.bincount(yj))/len(yj)
rf_mc.fit(X_all,yj)
ovr={}
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
    for f in FEATURES:
        n1,n2=len(d1),len(d2)
        sp=np.sqrt(((n1-1)*np.var(d1[f],ddof=1)+(n2-1)*np.var(d2[f],ddof=1))/(n1+n2-2))
        d=abs(np.mean(d1[f])-np.mean(d2[f]))/max(sp,1e-10)
        pd_list.append({'j1':J_ABV[jns[j1]],'j2':J_ABV[jns[j2]],'feat':f,'d':d})
pd_list.sort(key=lambda x:x['d'],reverse=True)
bo=max(ovr.items(),key=lambda x:x[1]['auc'])
print(f"  Acc={mc['test_accuracy'].mean():.3f} (bl={bl_j:.3f}) F1={mc['test_f1_macro'].mean():.3f}")
print(f"  Best OvR: {bo[0]} AUC={bo[1]['auc']:.3f}")
print(f"  Best pairwise: {pd_list[0]['j1']} vs {pd_list[0]['j2']} ({pd_list[0]['feat']}) d={pd_list[0]['d']:.3f}")

# 4. 5-way comparison
print("\n[4] 5-Way Comparison...")
b_auc=max(v['auc'] for v in cr.values()) if cr else 0
prev={
    'Biber':         [0.0063,0.604,0.332,0.019,0.751,0.173],
    'Metadiscourse': [0.0077,0.592,0.409,0.096,0.874,0.259],
    'Syntactic':     [0.0017,0.551,0.341,0.028,0.717,0.183],
    'InfoDensity':   [0.0071,0.634,0.403,0.090,0.891,0.241],
}
now=[r2o,b_auc,mc['test_accuracy'].mean(),mc['test_accuracy'].mean()-bl_j,bo[1]['auc'],mc['test_f1_macro'].mean()]

print(f"  {'Metric':<12s} {'Biber':>7s} {'Meta':>7s} {'Syn':>7s} {'InfoD':>7s} {'Base6':>7s}  {'Best'}")
metrics=['Reg R^2','Cls AUC','Jrn Acc','Acc-BL','Jrn OvR','Jrn F1']
for i,m in enumerate(metrics):
    vals={'Biber':prev['Biber'][i],'Meta':prev['Metadiscourse'][i],'Syn':prev['Syntactic'][i],'InfoD':prev['InfoDensity'][i],'Base6':now[i]}
    best=max(vals,key=vals.get)
    print(f"  {m:<12s} {prev['Biber'][i]:7.4f} {prev['Metadiscourse'][i]:7.4f} {prev['Syntactic'][i]:7.4f} {prev['InfoDensity'][i]:7.4f} {now[i]:7.4f}  -> {best}")

print(f"\n  Journal Acc-BL ranking:")
acc_bl={'Biber':0.019,'Meta':0.096,'Syn':0.028,'InfoD':0.090,'Base6':now[3]}
for m,v in sorted(acc_bl.items(),key=lambda x:x[1],reverse=True):
    print(f"    {m:15s} {v:+.3f}")

# 5. Charts
print("\n[5] Charts...")

# Fig 1: 5-way Acc-BL comparison
fig,ax=plt.subplots(figsize=(10,5))
methods=['Biber\n(15+4PCA)','Metadiscourse\n(10)','Syntactic\n(13)','InfoDensity\n(10)','Baseline 6\n(orig)']
accs=[0.019,0.096,0.028,0.090,now[3]]
colors=['#4575b4','#d73027','#66bd63','#fc8d59','#9467bd']
bars=ax.bar(methods,accs,color=colors,edgecolor='white',linewidth=2)
for bar,v in zip(bars,accs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.003 if v>=0 else v-0.015, f'{v:+.3f}', ha='center',fontweight='bold',fontsize=12)
ax.axhline(0,color='black',linewidth=0.5)
ax.set_title('Journal Classification: Accuracy Above Baseline\n(5 Schemes Comparison)',fontsize=14)
ax.set_ylabel('Accuracy - Baseline')
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_b1_5way.png',bbox_inches='tight'); plt.close()
print("  [OK] fig_b1_5way.png")

# Fig 2: Feature importance
fig,axes=plt.subplots(1,2,figsize=(12,5))
rf_imp=pd.DataFrame({'f':FEATURES,'i':rf_r.feature_importances_}).sort_values('i')
axes[0].barh(rf_imp['f'],rf_imp['i'],color='forestgreen',edgecolor='white')
axes[0].set_title('RF Importance (NCC Regression)')
cv2=[(f,pr[f]['r']) for f in FEATURES]; cv2.sort(key=lambda x:abs(x[1]))
axes[1].barh([c[0] for c in cv2],[c[1] for c in cv2],color=['#d73027' if v<0 else '#4575b4' for _,v in cv2],edgecolor='white')
axes[1].axvline(0,color='black',linewidth=0.5)
axes[1].set_title('Pearson r with NCC')
fig.suptitle('Baseline 6 Features',fontsize=14)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_b2_importance.png',bbox_inches='tight'); plt.close()
print("  [OK] fig_b2_importance.png")

# Save
summary={'n':n,'p':p,'r2':r2o,'adj':adj,'rcv':rcv.mean(),'lcv':lcv.mean(),'rfcv':rfcv.mean(),'best_r':pr[bf]['r'],'best_f':bf,'best_auc':b_auc,'jrn_acc':mc['test_accuracy'].mean(),'jrn_bl':bl_j,'jrn_f1':mc['test_f1_macro'].mean(),'jrn_ovr':bo[1]['auc']}
pd.DataFrame([summary]).to_csv(OUTPUT_DIR/'results_summary.csv',index=False)
print(f"\n[OK] Done.")
