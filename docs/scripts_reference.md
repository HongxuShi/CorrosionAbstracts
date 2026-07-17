# 脚本参考手册

> 生成日期：2026-07-16  
> 适用场景：在图书馆等稳定网络环境下，从零开始完成 6 个期刊的数据采集、特征提取与汇合

---

## 目录

- [0. 环境准备](#0-环境准备)
- [1. fetch_journal.py — 单期刊数据采集](#1-fetch_journalpy--单期刊数据采集)
- [2. extract_features.py — 六大特征提取](#2-extract_featurespy--六大特征提取)
- [3. merge_datasets.py — 多期刊汇合](#3-merge_datasetspy--多期刊汇合)
- [附录A：一键复制命令（按顺序执行）](#附录a一键复制命令按顺序执行)
- [附录B：常见问题排查](#附录b常见问题排查)
- [附录C：输出文件清单与验证](#附录c输出文件清单与验证)

---

## 0. 环境准备

### 0.1 激活环境

```bash
# 在 Anaconda Prompt 或终端中
conda activate base

# 验证依赖
python -c "import pandas, numpy, pyalex; print('环境就绪')"
```

### 0.2 设置 API Key（可选但推荐）

```bash
# Windows (CMD)
set OPENALEX_API_KEY=Lw2LtYlvpY9073xqu7hVSx

# Windows (PowerShell)  
$env:OPENALEX_API_KEY="Lw2LtYlvpY9073xqu7hVSx"

# Linux / WSL
export OPENALEX_API_KEY="Lw2LtYlvpY9073xqu7hVSx"
```

不设置也可以运行，但 API 访问速度会较慢。

### 0.3 确保工作目录正确

```bash
cd CorrosionAbstracts

# 确认目录结构
dir scripts   # Windows
ls scripts/   # Linux/WSL
```

应该看到三个脚本：`fetch_journal.py`、`extract_features.py`、`merge_datasets.py`

---

## 1. fetch_journal.py — 单期刊数据采集

### 1.1 功能说明

从 OpenAlex API 按期刊 ISSN 拉取论文元数据与摘要，输出标准 JSON 文件。

**核心特性：**
- 断点续传：意外中断后加 `--resume` 从断点继续，不丢数据
- 干运行：`--dry-run` 先看数量再决定是否下载
- 自动限速：内置请求间隔，避免 API 封禁
- 事后报告：完成后自动输出 DOI 覆盖率、摘要覆盖率、年份分布

### 1.2 参数列表

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--journal` | 选项 | 预设期刊标识符（见下方列表） | 无 |
| `--issn` | 字符串 | 自定义 ISSN（不使用预设时） | 无 |
| `--list` | 开关 | 列出所有预设期刊并退出 | — |
| `-o`, `--output` | 路径 | 输出 JSON 文件路径（采集模式必需） | 无 |
| `--from` | 日期 | 起始日期 `YYYY-MM-DD` | `2015-01-01` |
| `--to` | 日期 | 结束日期 `YYYY-MM-DD` | `2026-12-31` |
| `--dry-run` | 开关 | 仅查询数量，不下载数据 | — |
| `--resume` | 开关 | 从检查点文件恢复中断的采集 | — |
| `--api-key` | 字符串 | OpenAlex API Key（优先级高于环境变量） | 无 |

### 1.3 预设期刊对照表

| 标识符 | 期刊全称 | ISSN | 预估文章数 |
|--------|----------|------|-----------|
| `corrosion_science` | Corrosion Science | 0010-938X | ~685 |
| `corrosion` | Corrosion | 0010-9312 | ~1,257 |
| `corrosion_engineering` | Corrosion Engineering, Science and Technology | 1478-422X | ~881 |
| `materials_corrosion` | Materials and Corrosion | 0947-5117 | ~1,701 |
| `anti_corrosion_methods` | Anti-Corrosion Methods and Materials | 0003-5599 | ~892 |
| `corrosion_materials_degradation` | Corrosion and Materials Degradation | 2624-5558 | ~266 |

### 1.4 完整操作示例

以 Corrosion Science 为例，展示完整的三步流程：

```bash
# 步骤 1：干运行，查看预计数据量
python scripts/fetch_journal.py --journal corrosion_science --dry-run

# 输出示例：
#   期刊:    Corrosion Science
#   ISSN:    0010-938X
#   预计可获取: 685 条记录
#   约需分页:   4 页 (每页200条)

# 步骤 2：正式采集
python scripts/fetch_journal.py --journal corrosion_science -o raw/CorrosionScience.json

# 输出示例：
#   [采集] 开始分页拉取数据...
#     第 1/4 页 | 本页新增 200 条 | 累计 200/685
#     第 2/4 页 | 本页新增 200 条 | 累计 400/685
#     ...
#   [完成] 数据采集完毕！
#     采集记录: 685 条
#     摘要覆盖率: 685/685 (100.0%)
#     年份分布:
#       2015: 31     ████
#       2016: 22     ███
#       ...

# 步骤 3：如果中断了，断点续传
python scripts/fetch_journal.py --journal corrosion_science -o raw/CorrosionScience.json --resume
```

### 1.5 异常处理

| 情况 | 处理方式 |
|------|----------|
| `Ctrl+C` 手动中断 | 检查点自动保存，下次加 `--resume` 继续 |
| 网络超时 | pyalex 自动重试 5 次（指数退避），无需人工干预 |
| API 返回空 | 检查 ISSN 是否正确，或用 `--dry-run` 确认 |
| 输出文件为 0 字节 | 大概率是 API Key 失效或网络问题，检查后重试 |

---

## 2. extract_features.py — 六大特征提取

### 2.1 功能说明

从 OpenAlex 原始 JSON（或含 `abstract` 列的 CSV）出发，一站式完成：
1. 摘要文本还原（倒排索引 → 自然文本）
2. 六大文体特征提取（ASL、MWL、LD、LC、JD、HD）
3. NCC 规范化引用计数计算（期刊×年份双维度归一化）
4. 输出包含所有原始字段和特征的完整 CSV

**核心特性：**
- 自适应 NCC：多期刊自动启用双维度归一化，单期刊退化为年份归一化
- 自动 `source_journal` 提取：从 JSON 中解析来源期刊名
- 两种输入模式：OpenAlex JSON（默认）或已有 CSV（`--from-csv`）
- 结果报告：输出各特征的均值、标准差、缺失率

### 2.2 参数列表

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `-i`, `--input` | 路径 | 输入文件（JSON 或 CSV） | **必需** |
| `-o`, `--output` | 路径 | 输出 CSV 路径 | `dataset_with_features.csv` |
| `--data-dir` | 路径 | 参考词表目录 | 自动查找 `../data` 或 `./data` |
| `--from-csv` | 开关 | 输入为 CSV 而非 JSON | — |
| `--no-report` | 开关 | 跳过统计报告输出 | — |
| `--abstract-col` | 字符串 | 摘要列名称（CSV 模式） | `abstract` |

### 2.3 参考数据依赖

脚本需要三个参考词表文件，位于 `data/` 目录：

| 文件 | 用途 | 行数 |
|------|------|------|
| `essential-word-list.txt` | 停用词表 → LD 计算 | 176 |
| `hedge_data.txt` | 模糊限制语 → HD 计算 | 162 |
| `df_j.txt` | 腐蚀术语 → JD 计算 | 289 |

**如果缺失**：脚本启动时会报错并提示文件路径。

### 2.4 输出列说明

| 列名 | 类型 | 含义 | 来源 |
|------|------|------|------|
| `doi` | 字符串 | 论文 DOI（唯一标识） | OpenAlex |
| `title` | 字符串 | 论文标题 | OpenAlex |
| `year` | 整数 | 发表年份 | OpenAlex |
| `citations` | 整数 | 绝对被引次数 | OpenAlex |
| `is_oa` | 布尔 | 是否开放获取 | OpenAlex |
| `type` | 字符串 | 文献类型 | OpenAlex |
| `source_journal` | 字符串 | 来源期刊名 | OpenAlex（自动解析） |
| `abstract` | 字符串 | 还原后的摘要全文 | OpenAlex（倒排索引还原） |
| **`ASL`** | 浮点 | 平均句长 = 词数/句号数 | **脚本计算** |
| **`MWL`** | 浮点 | 平均词长 = 去标点字符数/词数 | **脚本计算** |
| **`LD`** | 浮点 | 词汇密度 = (词数-停用词)/词数 | **脚本计算** |
| **`LC`** | 浮点 | 词汇高级度 = 不重复词数/词数 | **脚本计算** |
| **`JD`** | 浮点 | 术语密度 = 术语命中数/词数 | **脚本计算** |
| **`HD`** | 浮点 | 层次结构密度 = 模糊词数/词数 | **脚本计算** |
| **`NCC`** | 浮点 | 规范化引用 = 引用数/同期刊同年份均值 | **脚本计算** |

### 2.5 六大特征计算细节

| 特征 | 公式 | 关键细节 |
|------|------|----------|
| ASL | `word_count / sentence_count` | 句子数 = `.` 的数量（简单近似） |
| MWL | `char_count_nopunc / word_count` | 标点集 = `,.'"!-%~`，逐字符移除 |
| LD | `(total_words - stop_count) / total_words` | 用 `" word "` 边界匹配停用词 |
| LC | `unique_word_types / total_word_tokens` | Type-Token Ratio，空格包围去重 |
| JD | `jargon_match_count / word_count` | 简单子串匹配 `term in text` |
| HD | `hedge_match_count / word_count` | 简单子串匹配 `term in text` |

所有公式严格复现原始 `FeatureEngineer.ipynb` 中的计算逻辑。

### 2.6 NCC 归一化策略

| 数据情况 | 归一化方式 | 示例 |
|----------|-----------|------|
| 多期刊、有 `source_journal` | `citations / mean(journal, year)` | NCC = 28 / 27.8 = 1.01 |
| 单期刊或无期刊列 | `citations / mean(year)`（降级） | NCC = 28 / 24.6 = 1.14 |
| 某(期刊,年份)仅 1 篇 | NCC = 1.0（论文即自身均值） | 标记为低置信度 |

### 2.7 完整操作示例

```bash
# 从 OpenAlex JSON 提取特征
python scripts/extract_features.py \
    -i raw/CorrosionScience.json \
    -o processed/CorrosionScience_features.csv

# 输出示例：
#   [加载] JSON 中共有 685 条记录
#   [加载] 摘要覆盖率: 685/685 (100.0%)
#   [加载] 检测到 1 个期刊: Corrosion Science
#   ...
#   [1/6] ASL (平均句长)      → 685/685 条有效 (均值=20.82)
#   [2/6] MWL (平均词长)      → 685/685 条有效 (均值=6.74)
#   [3/6] LD  (词汇密度)      → 685/685 条有效 (均值=0.698)
#   [4/6] LC  (词汇高级度)    → 685/685 条有效 (均值=0.690)
#   [5/6] JD  (术语密度)      → 685/685 条有效 (均值=0.044)
#   [6/6] HD  (层次结构密度)  → 685/685 条有效 (均值=0.016)
#   归一化模式: 单维度（仅年份）
#   期刊数: 1（单期刊，退化为年份归一化）
#   ✓ NCC 计算完成
#   特征提取全部完成！总耗时: 0.9 秒

# 从已有 CSV 提取特征（如果输入已是含 abstract 的 CSV）
python scripts/extract_features.py \
    --from-csv \
    -i raw/dataset_cs_abs.csv \
    -o processed/test_output.csv
```

### 2.8 验证输出

```bash
python -c "
import pandas as pd
df = pd.read_csv('processed/CorrosionScience_features.csv')
print(f'总行数: {len(df)}')
print(f'总列数: {len(df.columns)}')
print(f'列名: {list(df.columns)}')
print(f'有摘要: {df[\"abstract\"].notna().sum()}')
print(f'NCC均值: {df[\"NCC\"].mean():.2f}')
print(f'各特征缺失数:')
print(df[['ASL','MWL','LD','LC','JD','HD','NCC']].isna().sum())
"
```

---

## 3. merge_datasets.py — 多期刊汇合

### 3.1 功能说明

扫描 `processed/` 目录下所有 `*_features.csv`，纵向拼接为单一数据集。

**核心特性：**
- 自动扫描：无需逐一指定文件，按命名模式匹配
- DOI 去重：同一篇论文出现在多期刊采集结果中则保留首次出现
- 列一致性检查：不同期刊的输出列不一致时报警
- 汇合报告：各期刊记录数、去重数、特征完整率

### 3.2 参数列表

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--input-dir` | 路径 | 包含 `*_features.csv` 的目录 | `../processed`（相对脚本） |
| `-o`, `--output` | 路径 | 输出 CSV 路径 | `../processed/merged_all_features.csv` |

### 3.3 完整操作示例

```bash
# 确认 processed/ 下有 6 个 *_features.csv
dir processed\*_features.csv     # Windows
ls processed/*_features.csv      # Linux/WSL

# 一键汇合
python scripts/merge_datasets.py

# 输出示例：
#   [扫描] 找到 6 个特征文件:
#     - CorrosionScience_features.csv              (685 KB)
#     - Corrosion_features.csv                     (1257 KB)
#     - CorrosionEngineering_features.csv          (881 KB)
#     - MaterialsCorrosion_features.csv            (1701 KB)
#     - AntiCorrosionMethods_features.csv          (892 KB)
#     - CorrosionMaterialsDegradation_features.csv (266 KB)
#   [读取] 正在逐个加载...
#     ✓ CorrosionScience                            685 条, 16 列
#     ✓ Corrosion                                  1257 条, 16 列
#     ...
#   [合并] 正在纵向拼接...
#     合并后总计: 5682 条
#     各期刊记录数:
#       Materials and Corrosion                       1701 条
#       Corrosion                                     1257 条
#       ...
#   [去重] ✓ 无重复 DOI
#   [检查] 特征完整记录: 5682/5682 (100.0%)
#   输出文件: processed/merged_all_features.csv
```

### 3.4 特殊处理

| 情况 | 行为 |
|------|------|
| 无 `source_journal` 列 | 正常合并，报告中将不显示期刊分布 |
| 某期刊缺少某列 | 报错提示列不一致，但继续合并（缺失列填 NaN） |
| 存在重复 DOI | 自动去重，保留首次出现的记录 |
| 列顺序不同 | 自动对齐，按第一份文件的列顺序输出 |

---

## 附录A：一键复制命令（按顺序执行）

以下命令可逐段复制到终端执行。每段结束后检查输出再继续下一段。

### A.1 全部干运行（先看数据量，约 30 秒）

```bash
python scripts/fetch_journal.py --journal corrosion_science --dry-run
python scripts/fetch_journal.py --journal corrosion --dry-run
python scripts/fetch_journal.py --journal corrosion_engineering --dry-run
python scripts/fetch_journal.py --journal materials_corrosion --dry-run
python scripts/fetch_journal.py --journal anti_corrosion_methods --dry-run
python scripts/fetch_journal.py --journal corrosion_materials_degradation --dry-run
```

### A.2 全部采集（约 2-5 分钟，取决于网速）

```bash
python scripts/fetch_journal.py --journal corrosion_science -o raw/CorrosionScience.json
python scripts/fetch_journal.py --journal corrosion -o raw/Corrosion.json
python scripts/fetch_journal.py --journal corrosion_engineering -o raw/CorrosionEngineering.json
python scripts/fetch_journal.py --journal materials_corrosion -o raw/MaterialsCorrosion.json
python scripts/fetch_journal.py --journal anti_corrosion_methods -o raw/AntiCorrosionMethods.json
python scripts/fetch_journal.py --journal corrosion_materials_degradation -o raw/CorrosionMaterialsDegradation.json
```

### A.3 全部特征提取（约 5-10 秒）

```bash
python scripts/extract_features.py -i raw/CorrosionScience.json -o processed/CorrosionScience_features.csv
python scripts/extract_features.py -i raw/Corrosion.json -o processed/Corrosion_features.csv
python scripts/extract_features.py -i raw/CorrosionEngineering.json -o processed/CorrosionEngineering_features.csv
python scripts/extract_features.py -i raw/MaterialsCorrosion.json -o processed/MaterialsCorrosion_features.csv
python scripts/extract_features.py -i raw/AntiCorrosionMethods.json -o processed/AntiCorrosionMethods_features.csv
python scripts/extract_features.py -i raw/CorrosionMaterialsDegradation.json -o processed/CorrosionMaterialsDegradation_features.csv
```

### A.4 汇合

```bash
python scripts/merge_datasets.py
```

### A.5 最终验证

```bash
python -c "
import pandas as pd
df = pd.read_csv('processed/merged_all_features.csv')
print(f'总记录数: {len(df)}')
print(f'期刊数:   {df[\"source_journal\"].nunique()}')
print(f'年份范围: {int(df[\"year\"].min())}–{int(df[\"year\"].max())}')
print(f'特征完整: {df[[\"ASL\",\"MWL\",\"LD\",\"LC\",\"JD\",\"HD\",\"NCC\"]].dropna().shape[0]} / {len(df)}')
print()
print('各期刊记录数:')
print(df['source_journal'].value_counts().to_string())
print()
print('NCC 均值（应为 1.0）:', round(df['NCC'].mean(), 2))
"
```

期望输出：
```
总记录数: ~5682
期刊数:   6
年份范围: 2015–2026
特征完整: ~5682 / ~5682
NCC 均值（应为 1.0）: 1.0
```

---

## 附录B：常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| `ModuleNotFoundError: No module named 'pyalex'` | 依赖未安装 | `pip install pyalex pandas numpy` |
| `ModuleNotFoundError: No module named 'pandas'` | 同上 | `pip install pandas` |
| 采集条数为 0 | ISSN 错误或日期范围无效 | 检查 `--list` 输出的 ISSN，或用 `--dry-run` 验证 |
| 摘要覆盖率为 0% | 未加 `has_abstract=True` 筛选 | 脚本已内置此筛选，若仍为 0 则可能是 API 返回异常 |
| NCC 均值不是 1.0 | 存在缺失值或异常年份 | 检查 `year` 列是否有 NaN，查看低置信度警告 |
| JD 全为 0 | `df_j.txt` 未正确加载 | 检查 `data/df_j.txt` 是否存在，行数应为 289 |
| `FileNotFoundError: data/essential-word-list.txt` | 参考文件路径错误 | 用 `--data-dir ./data` 显式指定路径 |
| 采集中断 | 网络波动或手动 Ctrl+C | 加 `--resume` 重新运行，从断点继续 |

---

## 附录C：输出文件清单与验证

| 文件路径 | 类型 | 大小（约） | 验证方式 |
|----------|------|-----------|----------|
| `raw/CorrosionScience.json` | JSON | ~3 MB | `python -c "import json; print(len(json.load(open('raw/CorrosionScience.json'))))"` |
| `raw/Corrosion.json` | JSON | ~5 MB | 同上 |
| `raw/CorrosionEngineering.json` | JSON | ~4 MB | 同上 |
| `raw/MaterialsCorrosion.json` | JSON | ~7 MB | 同上 |
| `raw/AntiCorrosionMethods.json` | JSON | ~4 MB | 同上 |
| `raw/CorrosionMaterialsDegradation.json` | JSON | ~1 MB | 同上 |
| `processed/CorrosionScience_features.csv` | CSV | ~0.5 MB | `python -c "import pandas as pd; df=pd.read_csv('...'); print(len(df), df['NCC'].mean())"` |
| `processed/*_features.csv`（其余 5 个） | CSV | 各 ~0.3-0.8 MB | 同上 |
| `processed/merged_all_features.csv` | CSV | ~5 MB | 见 A.5 验证脚本 |

---

> **明天到图书馆后**：设置 API Key → 复制 A.1 干运行确认 → 复制 A.2 采集 → 复制 A.3 提取 → 复制 A.4 汇合 → 复制 A.5 验证。全程预计 5-10 分钟。
