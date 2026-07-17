#!/usr/bin/env python3
"""
摘要完整性基准分析
Abstract Completeness Benchmark — Descriptive Statistics + Feature-based Completeness Score

分析内容:
  1. 语料库摘要结构基准 (句数/词数/数字句/缩写句的分布)
  2. 基于特征预测"是否包含所有标准要素"的能力
  3. 输出可用于copilot的百分位对标表
"""
import sys, re, warnings; from pathlib import Path
import numpy as np; import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'font.size':10})

if len(sys.argv) < 3:
    print("Usage: python analyze_completeness.py <feature_csv> <output_dir>")
    sys.exit(1)

FEATURE_CSV = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2]); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NCC_PATH = Path(__file__).resolve().parent.parent / "processed" / "merged_all_features.csv"

print(f"Completeness Analysis: {FEATURE_CSV.name} -> {OUTPUT_DIR}")

# Load
df_f = pd.read_csv(FEATURE_CSV)
df_ncc = pd.read_csv(NCC_PATH)
df = df_f.merge(df_ncc[['doi','year','NCC','source_journal','abstract']], left_on='doc_id',right_on='doi',how='inner')
print(f"  Samples: {len(df)}")

feat_cols = [c for c in df_f.columns if c not in ['doc_id'] and np.issubdtype(df_f[c].dtype, np.number)]

# === 1. Abstract structure benchmarks ===
print("\n[1] Abstract Structure Benchmarks...")

# Compute structural indicators from raw abstracts
def compute_structure(text):
    if not isinstance(text, str) or not text.strip():
        return {'n_sentences': 0, 'n_words': 0, 'has_numbers': 0, 'has_abbrev': 0,
                'has_method_words': 0, 'has_conclusion_words': 0, 'has_result_words': 0}
    clean = re.sub(r'<[^>]+>', '', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = clean.split()

    # Method words: "was/were measured/calculated/determined/examined/investigated/tested/analyzed/characterized/evaluated/assessed/performed/conducted/carried out/employed/used/utilized/applied/immersed/exposed/subjected/submitted/prepared/synthesized/fabricated/obtained/acquired/recorded/monitored"
    method_pattern = r'\b(was|were)\s+(measured|calculated|determined|examined|investigated|tested|analyzed|characterized|evaluated|assessed|performed|conducted|carried|employed|used|utilized|applied|immersed|exposed|subjected|prepared|synthesized|fabricated|obtained|recorded|monitored)\b'
    # Conclusion words
    conclusion_pattern = r'\b(conclude|concluded|conclusion|indicate|indicated|indicates|demonstrate|demonstrated|demonstrates|reveal|revealed|reveals|confirm|confirmed|confirms|show|showed|shown|found|find|finding|suggest|suggested|suggests|results)\b'
    # Result words
    result_pattern = r'\b(result|results|finding|findings|observation|observations|revealed|reveals|showed|shown|found|observed|exhibited|exhibits|displayed|display|yielded|yield|obtained|achieved|recorded)\b'

    return {
        'n_sentences': len(sentences),
        'n_words': len(words),
        'has_numbers': 1 if re.search(r'\d+', clean) else 0,
        'has_abbrev': 1 if re.search(r'[A-Z]{2,}', clean) else 0,
        'has_method_words': 1 if re.search(method_pattern, clean.lower()) else 0,
        'has_conclusion_words': 1 if re.search(conclusion_pattern, clean.lower()) else 0,
        'has_result_words': 1 if re.search(result_pattern, clean.lower()) else 0,
    }

structures = df['abstract'].apply(compute_structure)
struct_df = pd.DataFrame(structures.tolist())
df = pd.concat([df, struct_df], axis=1)

# Benchmarks
benchmarks = {}
for col in ['n_sentences','n_words']:
    benchmarks[col] = {'mean':df[col].mean(),'std':df[col].std(),'p10':df[col].quantile(0.10),'p25':df[col].quantile(0.25),'p50':df[col].quantile(0.50),'p75':df[col].quantile(0.75),'p90':df[col].quantile(0.90)}
for col in ['has_numbers','has_abbrev','has_method_words','has_conclusion_words','has_result_words']:
    benchmarks[col] = {'prevalence': df[col].mean()}

print(f"  Sentences: mean={benchmarks['n_sentences']['mean']:.1f} p50={benchmarks['n_sentences']['p50']:.0f}")
print(f"  Words: mean={benchmarks['n_words']['mean']:.0f} p50={benchmarks['n_words']['p50']:.0f}")
for col in ['has_numbers','has_method_words','has_conclusion_words','has_result_words']:
    print(f"  {col}: {benchmarks[col]['prevalence']*100:.0f}% of abstracts")

# === 2. Completeness score ===
print("\n[2] Completeness Score...")
# Score = sum of 5 binary elements (numbers, method, conclusion, results, abbreviation)
df['completeness_score'] = df['has_numbers'] + df['has_method_words'] + df['has_conclusion_words'] + df['has_result_words'] + df['has_abbrev']
df['is_complete'] = (df['completeness_score'] >= 4).astype(int)
print(f"  Score distribution: mean={df['completeness_score'].mean():.2f}/5")
print(f"  >=4 elements (complete): {df['is_complete'].mean()*100:.1f}%")

# Can features predict completeness?
if len(feat_cols) > 0 and df['is_complete'].sum() >= 10:
    X_c = StandardScaler().fit_transform(df[feat_cols].values)
    y_c = df['is_complete'].values
    cv_s = StratifiedKFold(n_splits=min(5, y_c.sum(), len(y_c)-y_c.sum()), shuffle=True, random_state=42)
    if cv_s.n_splits >= 2:
        rf_c = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1)
        sc = cross_validate(rf_c, X_c, y_c, cv=cv_s, scoring={'auc':'roc_auc','f1':'f1'}, n_jobs=-1)
        bl_c = max(y_c.mean(), 1-y_c.mean())
        print(f"  Completeness AUC={sc['test_auc'].mean():.3f} (bl={bl_c:.3f}) F1={sc['test_f1'].mean():.3f}")
    else:
        sc = {'test_auc': np.array([np.nan]), 'test_f1': np.array([np.nan])}
        bl_c = max(y_c.mean(), 1-y_c.mean())
else:
    sc = {'test_auc': np.array([np.nan]), 'test_f1': np.array([np.nan])}
    bl_c = np.nan

# === 3. NCC vs completeness correlation ===
print("\n[3] NCC vs Completeness...")
r_cp, p_cp = pearsonr(df['completeness_score'], df['NCC'].fillna(0))
print(f"  completeness_score vs NCC: r={r_cp:+.4f} p={p_cp:.4f}")

for col in ['has_numbers','has_method_words','has_conclusion_words','has_result_words','has_abbrev']:
    r, p = pearsonr(df[col], df['NCC'].fillna(0))
    if p < 0.05:
        print(f"  {col} vs NCC: r={r:+.4f} p={p:.4f} *")

# === 4. Chart ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Completeness score distribution
df['completeness_score'].hist(bins=6, ax=axes[0], color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Completeness Score (0-5)')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Abstract Completeness Distribution (mean={df["completeness_score"].mean():.2f})')
# Element prevalence
elements = ['has_numbers','has_method_words','has_conclusion_words','has_result_words','has_abbrev']
prevs = [df[e].mean()*100 for e in elements]
labels = ['Numbers','Method\nWords','Conclusion\nWords','Result\nWords','Abbreviations']
axes[1].barh(labels, prevs, color=['#4575b4','#66bd63','#d73027','#fc8d59','#9467bd'], edgecolor='white')
axes[1].set_xlabel('Prevalence (%)')
axes[1].set_title('Structural Element Prevalence')
for i,v in enumerate(prevs): axes[1].text(v+1, i, f'{v:.0f}%', va='center')
fig.suptitle(f'Abstract Completeness ({FEATURE_CSV.parent.name})', fontsize=14)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'fig_completeness.png',bbox_inches='tight'); plt.close()

# Save
results = {
    'scheme': FEATURE_CSV.parent.name,
    'mean_sentences': benchmarks['n_sentences']['mean'],
    'p50_sentences': benchmarks['n_sentences']['p50'],
    'mean_words': benchmarks['n_words']['mean'],
    'p50_words': benchmarks['n_words']['p50'],
    'has_numbers_pct': benchmarks['has_numbers']['prevalence'],
    'has_method_pct': benchmarks['has_method_words']['prevalence'],
    'has_conclusion_pct': benchmarks['has_conclusion_words']['prevalence'],
    'has_result_pct': benchmarks['has_result_words']['prevalence'],
    'has_abbrev_pct': benchmarks['has_abbrev']['prevalence'],
    'mean_completeness': df['completeness_score'].mean(),
    'complete_pct': df['is_complete'].mean(),
    'completeness_vs_ncc_r': r_cp,
    'completeness_auc': sc['test_auc'].mean() if len(sc['test_auc'])>0 else np.nan,
}
pd.DataFrame([results]).to_csv(OUTPUT_DIR/'results_completeness.csv', index=False)
print(f"\n[OK] Done.")
