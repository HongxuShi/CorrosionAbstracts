# 数据采集与特征提取操作流程

## 前置条件

```bash
# 安装依赖（在 Anaconda 环境中）
pip install pyalex pandas numpy

# 或使用项目环境文件
conda env create -f environment.yml
```

**API Key 配置**（可选，但建议设置以获得更快访问速度）：

```bash
export OPENALEX_API_KEY="your_key_here"
```

## 完整流水线

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1           Step 2              Step 3         Step 4     │
│  逐期刊采集  →  特征提取  →  汇合  →  开始分析                    │
│  fetch_journal  extract_features  merge_datasets   notebooks/    │
│      .py             .py              .py                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1：逐期刊数据采集

```bash
# 1.1 先用干运行查看数据量
python scripts/fetch_journal.py --journal corrosion_science --dry-run

# 1.2 确认数量合理后，正式采集
python scripts/fetch_journal.py --journal corrosion_science \
    -o raw/CorrosionScience.json

# 1.3 重复以上两步，采集全部 6 个期刊
python scripts/fetch_journal.py --journal corrosion \
    -o raw/Corrosion.json

python scripts/fetch_journal.py --journal corrosion_engineering \
    -o raw/CorrosionEngineering.json

python scripts/fetch_journal.py --journal materials_corrosion \
    -o raw/MaterialsCorrosion.json

python scripts/fetch_journal.py --journal anti_corrosion_methods \
    -o raw/AntiCorrosionMethods.json

python scripts/fetch_journal.py --journal corrosion_materials_degradation \
    -o raw/CorrosionMaterialsDegradation.json
```

**注意事项：**
- 如果采集过程中断，加 `--resume` 参数从断点继续
- 采集完成后检查输出报告中的年份分布和摘要覆盖率

---

## Step 2：特征提取

对每个期刊的原始 JSON 分别运行特征提取：

```bash
python scripts/extract_features.py \
    -i raw/CorrosionScience.json \
    -o processed/CorrosionScience_features.csv

python scripts/extract_features.py \
    -i raw/Corrosion.json \
    -o processed/Corrosion_features.csv

# ... 重复 4 次
```

**输出列说明：**

| 列名 | 含义 |
|------|------|
| doi | 论文 DOI（唯一标识） |
| title | 论文标题 |
| year | 发表年份 |
| citations | 绝对被引次数 |
| is_oa | 是否开放获取 |
| type | 文献类型 |
| source_journal | 来源期刊名 |
| abstract | 摘要全文 |
| ASL | 平均句长 |
| MWL | 平均词长 |
| LD | 词汇密度 |
| LC | 词汇高级度 (TTR) |
| JD | 术语密度 |
| HD | 层次结构密度（模糊限制语密度） |
| NCC | 规范化引用计数 |

---

## Step 3：汇合数据集

```bash
python scripts/merge_datasets.py
```

默认从 `processed/` 读取所有 `*_features.csv`，输出 `processed/merged_all_features.csv`。

**汇合日志示例：**
```
[扫描] 找到 6 个特征文件
[读取] 正在逐个加载...
  ✓ CorrosionScience                   685 条
  ✓ Corrosion                         1257 条
  ...
[去重] 发现 0 条重复 DOI
[合并] 总计 5682 条
[检查] 特征完整记录: 5682/5682 (100.0%)
```

---

## Step 4：分析

汇合后的数据集 `processed/merged_all_features.csv` 可直接用于分析。

在 `notebooks/` 中创建分析 Notebook，始终将 `source_journal` 作为控制变量：

```python
import pandas as pd
df = pd.read_csv('../processed/merged_all_features.csv')

# 分组统计
df.groupby('source_journal')[['ASL', 'MWL', 'LD', 'LC', 'JD', 'HD', 'NCC']].mean()
```

---

## 数据文件说明

```
raw/                          # 原始数据（OpenAlex API 输出，不可手动修改）
├── CorrosionScience.json
├── Corrosion.json
├── CorrosionEngineering.json
├── MaterialsCorrosion.json
├── AntiCorrosionMethods.json
└── CorrosionMaterialsDegradation.json

processed/                    # 加工后数据（可从 raw/ 重新生成）
├── CorrosionScience_features.csv
├── Corrosion_features.csv
├── CorrosionEngineering_features.csv
├── MaterialsCorrosion_features.csv
├── AntiCorrosionMethods_features.csv
├── CorrosionMaterialsDegradation_features.csv
└── merged_all_features.csv   # 最终汇合数据集

data/                         # 参考词表（特征提取的依赖文件）
├── essential-word-list.txt
├── hedge_data.txt
└── df_j.txt
```

---

## 常见问题

**Q: 采集时中断了怎么办？**
A: 重新运行相同命令并添加 `--resume`，脚本会从检查点自动恢复。

**Q: 某个期刊数据量很少？**
A: 检查该期刊的 ISSN 是否正确。新创刊的期刊（如 Corrosion and Materials Degradation）文章数天然较少。

**Q: 特征提取时 JD 全是 0？**
A: 检查 `data/df_j.txt` 是否完整。如果文件未被正确放置，脚本会报错。

**Q: 如何重新生成整个 processed/ 目录？**
A: 删除 `processed/*.csv`，从 Step 2 重新执行即可。raw/ 中的数据是一切的上游。
