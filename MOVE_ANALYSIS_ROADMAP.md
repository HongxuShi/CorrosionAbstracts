# 修辞Move分析：从零到工作系统的完整技术路线

## Rhetorical Move Analysis — Complete A-to-Z Implementation Guide

---

## 0. 你需要先知道的

### 0.1 我们要做什么

训练一个分类器，输入一句学术摘要，输出它属于哪个修辞Move（Background/Gap/Purpose/Method/Result/Conclusion）。

然后用这个分类器标注全部5871篇摘要，提取每篇的Move结构特征（哪个Move占比多少、顺序对不对、缺了什么），最终集成到copilot里做摘要结构诊断。

### 0.2 为什么是Move分析

前面五个lab已经证明：POS特征、元话语、句法复杂度、信息密度——都无法稳定预测引用影响力。但Move分析的不同之处在于它**不预测影响力**，它做的是**结构识别**——判断"这段文字是在讲方法还是在讲结论"。这个任务的信号本身就比"这篇论文会被引几次"强得多。

### 0.3 为什么可以本地跑

Move分类是句子级7分类任务。不涉及复杂推理、不涉及多步逻辑。对LLM来说就是"读一句学术英语，输出一个标签"——7B-14B的模型完全胜任。不需要GPT-4o级别的推理能力。

---

## 1. 环境准备

### 1.1 硬件前提

- GPU: NVIDIA RTX 5070 Ti (12GB VRAM) ✓
- RAM: 建议32GB以上
- 磁盘: 至少50GB可用（模型文件约8GB + 语料和中间文件约5GB）

### 1.2 创建conda环境

```bash
# 创建独立环境
conda create -n move python=3.11 -y
conda activate move

# 核心依赖
pip install vllm                    # LLM推理框架
pip install transformers            # HuggingFace工具链
pip install torch                   # PyTorch (CUDA 12.x)
pip install pandas numpy scipy      # 数据处理
pip install scikit-learn            # RF baseline + 评估
pip install tqdm jsonlines          # 进度条 + JSONL读写

# 可选（阶段二SciBERT fine-tune时再装）
# pip install datasets accelerate peft
```

### 1.3 验证GPU可用

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
```

预期输出：
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Ti
VRAM: 12.0 GB
```

---

## 2. 模型选型

### 2.1 为什么选Qwen3 14B Instruct

经过联网对比评测（2025-2026年数据），选择Qwen3 14B Instruct的理由：

| 考量因素             |      Qwen3 14B      | Llama 3.1 8B | Mistral 7B | Gemma 2 9B |
| ---------------- | :-----------------: | :----------: | :--------: | :--------: |
| JSON格式遵循         |        ⭐⭐⭐⭐⭐        |     ⭐⭐⭐      |    ⭐⭐⭐     |    ⭐⭐⭐⭐    |
| 学术英语理解           |        ⭐⭐⭐⭐         |     ⭐⭐⭐⭐     |    ⭐⭐⭐     |    ⭐⭐⭐     |
| 多语言迁移(中→英prompt) |        ⭐⭐⭐⭐⭐        |      ⭐⭐      |     ⭐⭐     |     ⭐⭐     |
| 4-bit量化后质量损失     |         极小          |      小       |     小      |     小      |
| 12GB显存适配         | ✅ (14B AWQ ≈ 7.5GB) |      ✅       |     ✅      |     ✅      |

**最关键的因素是JSON格式遵循**。7000次推理调用中，如果格式错误率是5%就需要手动修复350条。Qwen3在这个指标上显著优于其他7-9B模型。

14B参数在4-bit量化(AWQ)后约7.5GB显存，配12GB显卡刚好。虽然比7B模型多占一点空间，但换来的是明显更稳定的JSON输出——对于标注pipeline来说，稳定性比速度重要。

### 2.2 下载模型

```bash
# 方法一：vLLM直接加载（推荐，首次运行自动下载）
# 不需要手动下载，vLLM会自动从HuggingFace拉取

# 方法二：手动下载（如果网络不稳定）
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-14B-Instruct-AWQ --local-dir ./models/Qwen3-14B-AWQ
```

### 2.3 显存核算

```
Qwen3 14B AWQ 4-bit:
  模型权重:        ~7.5 GB
  KV Cache:        ~1.5 GB  (max_model_len=2048, batch=16)
  CUDA overhead:   ~0.8 GB
  ─────────────────────────
  总计:            ~9.8 GB
  
  12GB VRAM - 9.8GB = 2.2GB 余量 ✓
```

**如果显存不够**（系统其他进程占用）：
- 减小 `max_model_len` 到 1024（摘要每句不超过100词，1024 tokens足够）
- 减小 `gpu_memory_utilization` 到 0.80
- 备选：换 Qwen3 8B Instruct AWQ（~4.5GB）

---

## 3. Move分类体系定义

### 3.1 七类Move标签

基于Swales(1990, 2004) CARS模型，针对腐蚀科学摘要调整为7类：

| 标签             | 全称   | 交际功能              | 典型信号词/句式                                                                                    |
| -------------- | ---- | ----------------- | ------------------------------------------------------------------------------------------- |
| **Background** | 研究背景 | 建立研究领域、已知事实、主题重要性 | 现在时、领域术语、"is a major problem"、"has been widely used"                                        |
| **Gap**        | 知识空白 | 指出前人研究的不足或未解决的问题  | "however"、"remains unclear"、"few studies have"、"little is known"                            |
| **Purpose**    | 研究目的 | 宣布本文的目标/范围        | "this paper aims to"、"we investigate"、"the objective is"、"herein we report"                 |
| **Method**     | 研究方法 | 描述实验步骤、材料、表征手段    | 过去时+被动语态、"was measured"、"were prepared"、"using XRD/SEM/EIS"                                 |
| **Result**     | 研究结果 | 报告实验发现、数据、观察      | "results show"、"was found to be"、"exhibited"、"increased/decreased"、"the corrosion rate was" |
| **Conclusion** | 研究结论 | 解释结果意义、提出推论/建议    | "indicates that"、"suggests that"、"this demonstrates"、"therefore"、"it is concluded"          |
| **Other**      | 其他   | 不属于以上类别的句子（极少）    | 致谢、纯引用、公式编号等                                                                                |

### 3.2 标注示例（腐蚀科学领域真实摘要）

**示例1 — 标准IMRaD结构**：
```
[Background] Microbially influenced corrosion (MIC) is acknowledged 
  to be the direct cause of catastrophic corrosion failures, with 
  associated damage costs ranging to many billions of US$ annually.
[Gap] In spite of extensive research and numerous publications, 
  fundamental questions relating to MIC remain unanswered.
[Purpose] The following review provides an overview of current MIC 
  research and stresses the lack of information related to MIC 
  recognition, prediction and mitigation.
[Method] The review establishes a link between management decisions 
  and root causes.
[Conclusion] A holistic, proactive approach to MIC is suggested in 
  which an entire system is considered, monitored and improved.
```

**示例2 — 实验报告型**：
```
[Purpose] The purpose of this investigation was to study the inhibitive 
  action of chitosan extracted from Archachatina marginata snail shells 
  on the corrosion of plain carbon steel in acid media.
[Method] Weight loss and thermometric methods were used during this 
  investigation. Characterization of the obtained chitosan was 
  accomplished with Fourier transform infrared spectroscopy analysis.
[Method] The effects of parameters influencing the inhibition process 
  (concentration and temperature) were evaluated, and the sorption 
  isotherms and thermodynamic parameters were derived.
[Result] The results obtained showed that chitosan has good inhibition 
  potential with an efficiency of 93.2 per cent.
[Result] The inhibition efficiency decreased with an increase in 
  temperature but increased with increasing concentration of chitosan.
[Result] Test results best fitted the Langmuir Isotherm with a 
  correlation coefficient (R^2) of 0.999.
[Conclusion] The thermodynamic parameters studied reveal that the 
  adsorption of chitosan on the surface of mild steel is spontaneous.
```

### 3.3 边界情况处理指南

以下情况容易混淆，标注时需要特别注意：

| 边界情况 | 判断标准 | 标签 |
|---------|---------|------|
| "The results indicate that..." | 前半句像Result，但动词是"indicate"（解释性） | **Conclusion** |
| "X was measured using Y" | 包含方法(Method)中的具体操作 | **Method** |
| "Previous studies have shown..." | 引用前人工作建立背景 | **Background** |
| "However, little is known about..." | "However"开头但指出空白 | **Gap** |
| "We found that X increased by 30%" | 包含具体数字的结果报告 | **Result** |
| Method段末尾的"was calculated/determined" | 即使包含数字，如果描述的是方法流程 | **Method** |
| Result段末尾的"indicating/suggesting that" | "suggesting that"引导的是解释 | **Conclusion** |

---

## 4. 阶段一：LLM Zero-shot标注

### 4.1 目标

使用Qwen3 14B对500-1000篇摘要进行逐句Move标注，产生训练数据集。预期7类F1约80-85%（基于Qwen系列的学术文本理解能力估计）。

### 4.2 标注脚本

```python
#!/usr/bin/env python3
"""move_labeler.py — Qwen3 14B Zero-shot Move标注脚本"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ============================================================
# 配置
# ============================================================

MODEL_NAME = "Qwen/Qwen3-14B-Instruct-AWQ"
INPUT_CSV = "processed/merged_all_features.csv"
OUTPUT_JSONL = "data/move_annotations.jsonl"
LABEL_COUNT = 500  # 先标注500篇测试质量

SYSTEM_PROMPT = """You are an expert in academic discourse analysis, specifically Swales' CARS (Create-a-Research-Space) model.

Your task: classify each sentence of a corrosion science abstract into exactly ONE rhetorical move.

Move definitions:
- Background: Establishes research context, known facts, importance of the topic. Often uses present tense and domain terminology.
- Gap: Identifies a knowledge gap, problem, or limitation in previous work. Often uses "however", "remains unclear", "few studies", "little is known".
- Purpose: States the aim, objective, or scope of the current study. Often uses "this paper aims to", "we investigate", "the objective is", "herein".
- Method: Describes experimental procedures, materials, measurement techniques. Often uses past tense + passive voice + specific equipment names (XRD, SEM, EIS).
- Result: Reports findings, data, observations, or outcomes. Often contains numbers, "showed", "exhibited", "was found to be", "increased/decreased".
- Conclusion: Interprets results, discusses implications, or makes recommendations. Often uses "indicates that", "suggests that", "this demonstrates", "therefore".
- Other: Does not fit any of the above (very rare — acknowledgments, standalone citations, etc.)

CRITICAL RULES:
1. Output ONLY valid JSON. No explanation text. No markdown.
2. For each sentence, include: index (1-based integer), text (the sentence verbatim), move (one of the 7 labels above), confidence (float 0-1, your confidence in this classification).
3. A sentence that reports data with numbers is Result. A sentence that interprets what those numbers mean is Conclusion.
4. If unsure between two labels, pick the one that better matches the sentence's PRIMARY communicative function.
5. Use "Other" only when none of the other 6 labels apply."""


def build_prompt(abstract_text: str) -> str:
    """为单篇摘要构建标注prompt。"""
    # 先按句号分句
    sentences = re.split(r'(?<=[.!?])\s+', abstract_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # 给每个句子编号
    numbered = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))

    return f"""{SYSTEM_PROMPT}

Now analyze this abstract. Return JSON only.

Abstract:
{numbered}

Output format:
{{"sentences": [{{"index": 1, "text": "...", "move": "Background", "confidence": 0.95}}, ...]}}"""


def safe_parse_json(response: str) -> Optional[dict]:
    """从LLM输出中安全提取JSON，兼容常见格式问题。"""
    # 尝试1: 直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试2: 提取 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试3: 提取第一个 { ... } 块（贪婪匹配）
    match = re.search(r'\{.*"sentences".*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 尝试4: 提取所有 { "index": ... } 对象并手动组装
    sentence_matches = re.findall(
        r'\{\s*"index":\s*(\d+),\s*"text":\s*"([^"]*)",\s*"move":\s*"([^"]*)",\s*"confidence":\s*([\d.]+)\s*\}',
        response
    )
    if sentence_matches:
        sentences = []
        for idx, text, move, conf in sentence_matches:
            sentences.append({
                "index": int(idx),
                "text": text,
                "move": move,
                "confidence": float(conf)
            })
        return {"sentences": sentences}

    return None


def main():
    # ---- 初始化vLLM ----
    print("Loading Qwen3 14B AWQ...")
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        dtype="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    sampling_params = SamplingParams(
        temperature=0.0,       # 分类任务 → 0温度，确定性输出
        max_tokens=512,
        stop=["<|im_end|>"],
    )

    # ---- 加载数据 ----
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=['abstract'])
    df = df.head(LABEL_COUNT)  # 先标500篇

    print(f"Labeling {len(df)} abstracts...")

    # ---- 批量推理 ----
    prompts = [build_prompt(row['abstract']) for _, row in df.iterrows()]
    dois = df['doi'].tolist()

    outputs = llm.generate(prompts, sampling_params)

    # ---- 解析结果 ----
    success_count = 0
    fail_count = 0
    failed_dois = []

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for doi, prompt, output in tqdm(
            zip(dois, prompts, outputs),
            total=len(prompts),
            desc="Parsing"
        ):
            response_text = output.outputs[0].text
            parsed = safe_parse_json(response_text)

            if parsed and 'sentences' in parsed:
                parsed['abstract_id'] = doi
                f.write(json.dumps(parsed, ensure_ascii=False) + '\n')
                success_count += 1
            else:
                fail_count += 1
                failed_dois.append(doi)
                # 保存失败响应以便调试
                with open(OUTPUT_JSONL + '.failures', 'a', encoding='utf-8') as ef:
                    ef.write(f"DOI: {doi}\nResponse: {response_text}\n{'='*60}\n")

    print(f"\nDone!")
    print(f"  Successful: {success_count}/{LABEL_COUNT} ({100*success_count/LABEL_COUNT:.1f}%)")
    print(f"  Failed: {fail_count}")
    if failed_dois:
        print(f"  Failed DOIs saved to: {OUTPUT_JSONL}.failures")
    print(f"  Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    import torch
    main()
```

### 4.3 运行

```bash
conda activate move
python move_labeler.py
```

预期运行时间：500篇约6-10分钟（vLLM batch推理）。

### 4.4 质量验证

```python
#!/usr/bin/env python3
"""move_validate.py — 标注质量验证"""

import json
import random
from collections import Counter
from pathlib import Path

# 加载标注结果
annotations = []
with open("data/move_annotations.jsonl") as f:
    for line in f:
        annotations.append(json.loads(line))

# === 检查1: Move分布 ===
move_counts = Counter()
for ann in annotations:
    for sent in ann['sentences']:
        move_counts[sent['move']] += 1

total = sum(move_counts.values())
print("Move Distribution:")
for move, count in move_counts.most_common():
    print(f"  {move:15s}: {count:5d} ({100*count/total:5.1f}%)")

# === 检查2: 平均置信度 ===
confidences = []
for ann in annotations:
    for sent in ann['sentences']:
        confidences.append(sent['confidence'])
print(f"\nMean confidence: {sum(confidences)/len(confidences):.3f}")

# === 检查3: 低置信度样本 ===
low_conf = [
    (ann['abstract_id'], sent['index'], sent['move'], sent['confidence'], sent['text'][:100])
    for ann in annotations
    for sent in ann['sentences']
    if sent['confidence'] < 0.70
]
print(f"\nLow confidence (<0.70) samples: {len(low_conf)}/{total}")
for doi, idx, move, conf, text in low_conf[:10]:
    print(f"  [{doi}] sent#{idx} move={move} conf={conf:.2f}: {text}...")

# === 检查4: 异常Move序列 ===
# 检查Method出现在Result之后的情况（可能标注错误）
for ann in annotations[:20]:  # 随机抽20篇人工检查
    moves = [s['move'] for s in ann['sentences']]
    if 'Method' in moves and 'Result' in moves:
        method_idx = moves.index('Method')
        result_idx = moves.index('Result')
        if method_idx > result_idx:
            print(f"\n[!] Unusual order: Method after Result")
            print(f"  DOI: {ann['abstract_id']}")
            for s in ann['sentences']:
                print(f"    [{s['index']}] {s['move']:12s}: {s['text'][:80]}...")
```

如果验证发现问题较多（>10%的低置信度样本或>15%的格式解析失败），可以：
1. 优化prompt中的边界情况说明
2. 调整temperature为0.1做第二轮标注
3. 对低置信度样本进行人工复核后重新标注

---

## 5. 阶段二：训练生产级分类器

### 5.1 目标

用标注好的500-1000篇数据训练一个轻量分类器，部署在copilot里。好处：
- 推理不需要GPU（RF只需CPU）
- 100%格式确定性（不会出现JSON解析失败）
- 可解释的特征重要性

### 5.2 数据准备

```python
#!/usr/bin/env python3
"""move_prepare_training_data.py — 将标注数据转换为训练格式"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path

# 加载标注
annotations = []
with open("data/move_annotations.jsonl") as f:
    for line in f:
        ann = json.loads(line)
        annotations.append(ann)

# 加载原始摘要（用于获取完整上下文）
df = pd.read_csv("processed/merged_all_features.csv")
doi_to_abs = dict(zip(df['doi'], df['abstract']))

# 构建训练集：每行 = 一个句子
rows = []
for ann in annotations:
    doi = ann['abstract_id']
    full_text = doi_to_abs.get(doi, '')
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    total_sents = len(sentences)

    for sent_info in ann['sentences']:
        idx = sent_info['index'] - 1  # 转为0-based
        text = sent_info['text']
        move = sent_info['move']
        conf = sent_info['confidence']

        # 只用高置信度样本做训练
        if conf < 0.75:
            continue

        # 位置特征
        position = idx / max(total_sents - 1, 1)  # 0-1归一化位置

        # 简单词汇特征（不需要spaCy，足够做baseline）
        words = text.lower().split()
        word_count = len(words)

        rows.append({
            'doi': doi,
            'sent_idx': idx,
            'text': text,
            'move': move,
            'confidence': conf,
            'position_ratio': position,
            'word_count': word_count,
            'is_first_sentence': 1 if idx == 0 else 0,
            'is_last_sentence': 1 if idx == total_sents - 1 else 0,
            'has_we': 1 if any(w in words for w in ['we', 'our']) else 0,
            'has_however': 1 if 'however' in words else 0,
            'has_this_paper': 1 if 'this paper' in text.lower() or 'this study' in text.lower() else 0,
            'has_number': 1 if bool(re.search(r'\d+', text)) else 0,
            'has_result_word': 1 if bool(re.search(r'\b(show|showed|shown|found|find|reveal|exhibit|indicate|demonstrate)\b', text.lower())) else 0,
            'has_method_word': 1 if bool(re.search(r'\b(was|were)\s+(measured|calculated|determined|examined|investigated|tested|analyzed|characterized|evaluated|performed|conducted|employed|used|prepared|immersed)\b', text.lower())) else 0,
            'starts_with_however': 1 if text.lower().strip().startswith('however') else 0,
            'has_suggest_recommend': 1 if bool(re.search(r'\b(suggest|suggested|recommend|propose|conclude|indicate)\b', text.lower())) else 0,
        })

train_df = pd.DataFrame(rows)
print(f"Training samples: {len(train_df)}")
print(f"Move distribution:\n{train_df['move'].value_counts()}")

# 保存
train_df.to_csv("data/move_training_data.csv", index=False)
```

### 5.3 训练RF Baseline

```python
#!/usr/bin/env python3
"""move_train_rf.py — 训练Random Forest Move分类器"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 加载
df = pd.read_csv("data/move_training_data.csv")

# 特征列
feature_cols = [
    'position_ratio', 'word_count', 'is_first_sentence', 'is_last_sentence',
    'has_we', 'has_however', 'has_this_paper', 'has_number',
    'has_result_word', 'has_method_word', 'starts_with_however',
    'has_suggest_recommend',
]

X = StandardScaler().fit_transform(df[feature_cols].values)
le = LabelEncoder()
y = le.fit_transform(df['move'])

print(f"Training: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")
print(f"Classes: {list(le.classes_)}")

# 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
)

scoring = {'acc': 'accuracy', 'f1_macro': 'f1_macro'}
scores = cross_validate(rf, X, y, cv=cv, scoring=scoring, n_jobs=-1)

print(f"\nCV Results:")
print(f"  Accuracy: {scores['test_acc'].mean():.3f} +/- {scores['test_acc'].std():.3f}")
print(f"  F1_macro: {scores['test_f1_macro'].mean():.3f} +/- {scores['test_f1_macro'].std():.3f}")

# 在全部数据上训练最终模型
rf.fit(X, y)

# 特征重要性
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
for _, row in importances.iterrows():
    bar = '#' * int(row['importance'] * 50)
    print(f"  {row['feature']:25s}: {row['importance']:.4f} {bar}")

# Per-class report
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=le.classes_))

# 混淆矩阵
cm = confusion_matrix(y, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=le.classes_, yticklabels=le.classes_,
            cmap='YlOrRd', vmin=0, vmax=1, linewidths=1, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Move Classification Confusion Matrix\nRF Acc={scores["test_acc"].mean():.3f} F1={scores["test_f1_macro"].mean():.3f}')
fig.tight_layout(); fig.savefig("move_confusion_matrix.png", dpi=150)
print("  Saved: move_confusion_matrix.png")

# 保存模型
joblib.dump(rf, "models/move_classifier_rf.joblib")
joblib.dump(le, "models/move_label_encoder.joblib")
joblib.dump(scaler := StandardScaler().fit(df[feature_cols].values), "models/move_scaler.joblib")
print("  Saved: models/move_classifier_rf.joblib")

# 准确率决策
if scores['test_f1_macro'].mean() > 0.80:
    print("\n[OK] RF F1 > 0.80 — sufficient for production. Skip SciBERT.")
    print("  Use models/move_classifier_rf.joblib for copilot integration.")
elif scores['test_f1_macro'].mean() > 0.70:
    print("\n[WARN] RF F1 0.70-0.80 — borderline. Consider SciBERT for +5-10% improvement.")
else:
    print("\n[FAIL] RF F1 < 0.70 — insufficient. Proceed to SciBERT fine-tuning.")
```

### 5.4 SciBERT Fine-tuning（备选，仅当RF F1 < 0.80时需要）

```python
#!/usr/bin/env python3
"""move_train_scibert.py — SciBERT fine-tuning for Move classification"""

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# === Load data ===
df = pd.read_csv("data/move_training_data.csv")
le = LabelEncoder()
df['label'] = le.fit_transform(df['move'])
num_labels = len(le.classes_)

# === Prepare dataset ===
# 使用原文句子而非特征（SciBERT自己提取特征）
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.15, seed=42)

# === Tokenize ===
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128, padding=False)

dataset = dataset.map(tokenize_fn, batched=True)

# === Train ===
model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased",
    num_labels=num_labels,
)

training_args = TrainingArguments(
    output_dir="./models/scibert_move",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=True,  # 混合精度训练，节省显存
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("models/scibert_move_final")
tokenizer.save_pretrained("models/scibert_move_final")
print(f"Model saved to models/scibert_move_final")
```

---

## 6. 阶段三：全量Move标注 + 特征构建

### 6.1 用训练好的分类器标注全部5871篇

```python
#!/usr/bin/env python3
"""move_label_all.py — 全量Move标注"""

import re, joblib, json
import pandas as pd
import numpy as np
from tqdm import tqdm

# 加载模型
rf = joblib.load("models/move_classifier_rf.joblib")
le = joblib.load("models/move_label_encoder.joblib")
scaler = joblib.load("models/move_scaler.joblib")

# 特征列（与训练时一致）
feature_cols = [
    'position_ratio', 'word_count', 'is_first_sentence', 'is_last_sentence',
    'has_we', 'has_however', 'has_this_paper', 'has_number',
    'has_result_word', 'has_method_word', 'starts_with_however',
    'has_suggest_recommend',
]

# 加载全部数据
df = pd.read_csv("processed/merged_all_features.csv")
all_annotations = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
    text = row['abstract']
    if not isinstance(text, str) or not text.strip():
        continue

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    total_sents = len(sentences)

    sent_data = []
    for idx, sent_text in enumerate(sentences):
        words = sent_text.lower().split()
        wc = len(words)
        pos = idx / max(total_sents - 1, 1) if total_sents > 1 else 0

        feats = [
            pos, wc,
            1 if idx == 0 else 0,
            1 if idx == total_sents - 1 else 0,
            1 if any(w in words for w in ['we', 'our']) else 0,
            1 if 'however' in words else 0,
            1 if 'this paper' in sent_text.lower() or 'this study' in sent_text.lower() else 0,
            1 if bool(re.search(r'\d+', sent_text)) else 0,
            1 if bool(re.search(r'\b(show|showed|shown|found|find|reveal|exhibit|indicate)\b', sent_text.lower())) else 0,
            1 if bool(re.search(r'\b(was|were)\s+(measured|calculated|determined|examined|investigated|tested|analyzed|characterized|evaluated|performed|conducted|employed|used|prepared)\b', sent_text.lower())) else 0,
            1 if sent_text.lower().strip().startswith('however') else 0,
            1 if bool(re.search(r'\b(suggest|suggested|recommend|propose|conclude|indicate)\b', sent_text.lower())) else 0,
        ]

        X = scaler.transform([feats])
        move_id = rf.predict(X)[0]
        move_label = le.inverse_transform([move_id])[0]
        proba = rf.predict_proba(X)[0]

        sent_data.append({
            'index': idx + 1,
            'text': sent_text,
            'move': move_label,
            'confidence': float(proba[move_id]),
        })

    all_annotations.append({
        'abstract_id': row['doi'],
        'sentences': sent_data,
    })

# 保存
with open("data/move_annotations_all.jsonl", 'w', encoding='utf-8') as f:
    for ann in all_annotations:
        f.write(json.dumps(ann, ensure_ascii=False) + '\n')

print(f"Done: {len(all_annotations)} abstracts labeled")
```

### 6.2 构建Move结构特征

```python
#!/usr/bin/env python3
"""move_build_features.py — 提取Move结构特征矩阵"""

import json, re
import pandas as pd
import numpy as np
from collections import Counter

# 加载全量标注
annotations = []
with open("data/move_annotations_all.jsonl") as f:
    for line in f:
        annotations.append(json.loads(line))

# 为每篇摘要提取Move结构特征
rows = []
for ann in annotations:
    moves = [s['move'] for s in ann['sentences']]
    texts = [s['text'] for s in ann['sentences']]
    n_total = len(moves)

    # 基础统计
    move_counts = Counter(moves)
    move_sequence = ' -> '.join(moves)

    # 构建特征
    features = {
        'abstract_id': ann['abstract_id'],
        'n_sentences': n_total,
        'n_moves': len(set(moves)),  # 使用了多少种不同的move

        # 各Move是否存在
        'has_background': 1 if 'Background' in moves else 0,
        'has_gap': 1 if 'Gap' in moves else 0,
        'has_purpose': 1 if 'Purpose' in moves else 0,
        'has_method': 1 if 'Method' in moves else 0,
        'has_result': 1 if 'Result' in moves else 0,
        'has_conclusion': 1 if 'Conclusion' in moves else 0,

        # 各Move占比
        'background_ratio': move_counts.get('Background', 0) / n_total,
        'gap_ratio': move_counts.get('Gap', 0) / n_total,
        'purpose_ratio': move_counts.get('Purpose', 0) / n_total,
        'method_ratio': move_counts.get('Method', 0) / n_total,
        'result_ratio': move_counts.get('Result', 0) / n_total,
        'conclusion_ratio': move_counts.get('Conclusion', 0) / n_total,

        # 结构完整性
        'completeness': sum([
            1 if 'Background' in moves else 0,
            1 if 'Purpose' in moves else 0,
            1 if 'Method' in moves else 0,
            1 if 'Result' in moves else 0,
            1 if 'Conclusion' in moves else 0,
        ]),

        # 顺序特征
        'starts_with_background': 1 if moves[0] == 'Background' else 0,
        'starts_with_purpose': 1 if moves[0] == 'Purpose' else 0,
        'ends_with_conclusion': 1 if moves[-1] == 'Conclusion' else 0,
        'ends_with_result': 1 if moves[-1] == 'Result' else 0,
        'has_gap_before_purpose': 1 if 'Gap' in moves and 'Purpose' in moves and moves.index('Gap') < moves.index('Purpose') else 0,
    }

    rows.append(features)

move_df = pd.DataFrame(rows)
move_df.to_csv("data/move_feature_matrix.csv", index=False)
print(f"Move features: {move_df.shape[0]} abstracts × {move_df.shape[1]} features")
print(f"Feature columns: {list(move_df.columns)}")
```

---

## 7. 阶段四：Move特征的实验分析

Move特征矩阵构建完成后，复用现有五lab的分析框架（回归→NCC、分类高/低引用、期刊分类、年代预测、完整性、研究类型），创建 `lab_MoveAnalysis/`：

```
lab_MoveAnalysis/
├── move_feature_matrix.csv       ← 阶段三输出
├── run_all_analyses.py           ← 复用Syntactic lab的分析脚本
├── analyze_year.py               ← 复用
├── analyze_completeness.py       ← 复用
├── analyze_research_type.py      ← 复用
└── output/
```

将Move特征加入六方案对比矩阵中。

---

## 8. 阶段五：Copilot集成

### 8.1 Move诊断输出设计

```markdown
# 摘要Move结构分析

## 你的Move分布
Background  ████████░░ 35%  (语料库典型: 20-30%) ⚠️ 偏长 — 建议压缩至1-2句
Purpose     ██░░░░░░░░ 10%  (语料库典型: 10-15%) ✓
Method      ██░░░░░░░░  8%  (语料库典型: 25-35%) ⚠️ 方法描述过短
Result      ██████░░░░ 28%  (语料库典型: 20-30%) ✓
Conclusion  ████░░░░░░ 19%  (语料库典型: 10-15%) ⚠️ 结论段偏长

## 结构完整性: 4/5
✓ Background ✓ Purpose ✓ Result ✓ Conclusion
✗ Method (仅占8%，且缺乏实验细节的关键词)

## Move顺序
你的: Background → Purpose → Result → Method → Conclusion
典型: Background → Purpose → Method → Result → Conclusion
⚠️ Method在Result之后 — 不规范的倒置

## 增效建议
- 你的Method段仅1句(8%)。语料库中实验类摘要的方法段平均2-3句。
  建议在实验条件部分增加1-2句，描述材料、设备或测量参数。
- [来源: Meta分析] 目标期刊Anti-Corr.MM.偏好显性结构标记。
  建议在Purpose句前增加frame marker (如"This study aims to...")
- [来源: 年代趋势] 你的被动语态密度(0.08)处于2018年水平。
  2024年典型值为0.03——考虑使用更主动的表述。
```

### 8.2 技术实现要点

```python
# copilot中的Move分类推理（轻量CPU推理）
import joblib, re
import numpy as np

class MoveClassifier:
    def __init__(self):
        self.rf = joblib.load("models/move_classifier_rf.joblib")
        self.le = joblib.load("models/move_label_encoder.joblib")
        self.scaler = joblib.load("models/move_scaler.joblib")
        # 加载语料库基准
        self.benchmarks = pd.read_csv("data/move_benchmarks.csv")

    def analyze(self, abstract_text: str) -> dict:
        sentences = re.split(r'(?<=[.!?])\s+', abstract_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        n = len(sentences)

        moves = []
        for idx, sent in enumerate(sentences):
            features = self._extract_features(sent, idx, n)
            X = self.scaler.transform([features])
            move_id = self.rf.predict(X)[0]
            move_label = self.le.inverse_transform([move_id])[0]
            moves.append(move_label)

        return {
            'moves': moves,
            'distribution': self._compute_distribution(moves),
            'benchmarks': self._get_benchmarks(),
            'suggestions': self._generate_suggestions(moves),
        }
    # ... (完整实现在copilot项目中完成)
```

---

## 9. 全流程时间线

| 天数 | 任务 | 产出 |
|:---:|------|------|
| 1 | 环境配置 + 下载模型 + 编写标注脚本 | `move_labeler.py` 可运行 |
| 1 | 标注500篇 + 质量验证 | `move_annotations.jsonl` (500篇) |
| 2 | 训练RF + 评估准确率 → 决定是否需要SciBERT | `move_classifier_rf.joblib` |
| 2 | (如需) SciBERT fine-tune | `scibert_move_final/` |
| 3 | 全量5871篇标注 + Move特征提取 | `move_feature_matrix.csv` |
| 3-4 | 六项分析实验 (复用现有lab框架) | `lab_MoveAnalysis/output/` |
| 4-5 | 六方案对比矩阵更新 + copilot集成设计 | 完整六方案对比表 |
| 5 | 文档更新 + 汇报 | 最终报告 |

**总计：4-5个工作日**

---

## 10. 关键决策点

这个路线图中你需要在两个节点做判断：

**决策点1：RF是否够用？**
- F1 > 0.80 → 跳过SciBERT，直接用RF部署
- F1 0.70-0.80 → 可选SciBERT（+5-10%提升，但增加GPU依赖）
- F1 < 0.70 → 必须SciBERT

**决策点2：Qwen3标注质量是否达标？**
- JSON解析成功率 > 95% → 继续
- < 95% → 优化prompt或考虑回退到Gemini Flash（便宜且JSON好）
