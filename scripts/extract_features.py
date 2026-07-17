#!/usr/bin/env python3
"""
================================================================================
 腐蚀领域学术摘要文体特征一站式提取脚本
 One-stop Feature Extraction for Corrosion Science Abstracts
================================================================================

 功能说明：
   从 OpenAlex API 获取的原始 JSON 数据集出发，一次性完成以下所有处理：
   1. 摘要文本还原（从 OpenAlex 倒排索引格式还原为可读文本）
   2. 六大文体特征的自动提取
   3. 规范化引用计数（NCC）的计算
   4. 输出包含所有特征的完整 CSV 数据集

 六大文体特征：
   ┌────────┬──────────────────────┬─────────────────────────────┐
   │ 缩写   │ 特征名称             │ 计算方式                    │
   ├────────┼──────────────────────┼─────────────────────────────┤
   │ ASL    │ 平均句长             │ 总词数 / 句号数             │
   │ MWL    │ 平均词长             │ 去标点字符数 / 总词数       │
   │ LD     │ 词汇密度             │ (总词数-停用词) / 总词数    │
   │ LC     │ 词汇高级度 (TTR)     │ 不重复词数 / 总词数         │
   │ JD     │ 术语密度             │ 腐蚀术语数 / 总词数         │
   │ HD     │ 层次结构密度(模糊语) │ 模糊限制语数 / 总词数       │
   │ NCC    │ 规范化引用计数       │ 引用数/(同期刊同年份均值)   │
   └────────┴──────────────────────┴─────────────────────────────┘

 依赖文件（需放在 data/ 目录下）：
   - essential-word-list.txt : 停用词表（用于 LD 计算）
   - hedge_data.txt          : 模糊限制语词表（用于 HD 计算）
   - df_j.txt                : 腐蚀领域术语表（用于 JD 计算）

 用法示例：
   # 基本用法：从 JSON 输入，输出带特征的 CSV
   python extract_features.py \\
       --input dataset.json \\
       --output final_dataset.csv

   # 指定参考数据目录
   python extract_features.py \\
       --input dataset.json \\
       --output final_dataset.csv \\
       --data-dir ./data

   # 如果输入已经是包含 abstract 列的 CSV，可以直接处理
   python extract_features.py \\
       --input dataset_with_abstract.csv \\
       --output final_dataset.csv \\
       --from-csv

 作者：基于王淳、石洪旭原始 notebook 代码整合重构
 项目：腐蚀领域学术摘要文体特征与影响力关联分析
================================================================================
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================================
#  全局配置常量
# ============================================================================

# 标点符号列表 —— 用于 MWL（平均词长）和 LC（词汇高级度）计算时的清洗
PUNCTUATION = ',.\'"!-%~'

# 输出 CSV 的列顺序
OUTPUT_COLUMNS = [
    'doi', 'title', 'year', 'citations', 'is_oa', 'type',
    'source_journal',
    'abstract',
    'ASL',   # 平均句长
    'MWL',   # 平均词长
    'LD',    # 词汇密度
    'LC',    # 词汇高级度
    'JD',    # 术语密度
    'HD',    # 层次结构密度
    'NCC',   # 规范化引用计数
]


# ============================================================================
#  第一部分：摘要文本还原
# ============================================================================

def rebuild_abstract(inverted_index):
    """
    从 OpenAlex 的 abstract_inverted_index（倒排索引）还原为正常的摘要文本。

    OpenAlex 为了节省存储空间，将摘要以倒排索引形式存储：
      {"word": [position1, position2, ...], ...}
    例如：
      {"the": [0, 5], "corrosion": [1], "of": [2], "metal": [3], "in": [4]}
      → "the corrosion of metal in the"

    参数:
        inverted_index: dict 或 None — OpenAlex 的 abstract_inverted_index 字段

    返回:
        str — 还原后的摘要文本；若无摘要则返回空字符串
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    # 将 (位置, 词语) 对收集起来
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    # 按位置排序后拼接
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for pos, word in word_positions)


# ============================================================================
#  第二部分：参考数据加载
# ============================================================================

def load_stopwords(filepath):
    """
    加载停用词表（essential-word-list.txt）。

    文件格式：每行一个词，可能有逗号后缀或空白字符。
    处理方式与原 notebook 完全一致：
      - 逐行读取
      - 去除空白和逗号
      - 统一转为小写

    参数:
        filepath: str — 停用词文件路径

    返回:
        list[str] — 清洗后的停用词列表
    """
    stopwords = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                # 去除可能附带的逗号（原始文件中有 "your," "between," 等格式）
                clean_word = word.strip().lower().rstrip(',')
                if clean_word:
                    stopwords.append(clean_word)
    return stopwords


def load_hedge_words(filepath):
    """
    加载模糊限制语词表（hedge_data.txt）。

    文件格式：每行一个词或短语（如 "largely", "in general", "sort of"）。
    处理方式与原 notebook 一致：
      - 逐行读取，去除空白
      - 保留原始大小写形式但匹配时统一小写

    参数:
        filepath: str — 模糊限制语文件路径

    返回:
        list[str] — 模糊限制语列表
    """
    hedges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                hedges.append(word)
    return hedges


def load_jargon_list(filepath):
    """
    加载腐蚀领域术语表（df_j.txt）。

    文件格式：每行格式为 "中文术语\\xa0 英文术语"，
    其中 \\xa0 为不间断空格（non-breaking space）。
    例如："加速腐蚀试验\\xa0 accelerated corrosion test"

    处理方式与原 notebook 完全一致：
      - 以 \\xa0（不间断空格）分割中英文
      - 取英文部分（第二部分）
      - 去除首尾空白
      - 使用简单子串匹配（jargon_term in text）

    原 notebook 对应代码：
      df_jargon = pd.read_csv("df_j.txt", sep="\\xa0 ", engine='python', header=None)
      df_jargon = df_jargon[1]
      df_jargon = df_jargon.dropna()

    参数:
        filepath: str — 术语文件路径

    返回:
        list[str] — 英文腐蚀术语列表
    """
    jargons = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 以不间断空格 \\xa0 分割中英文部分
            if '\xa0' in line:
                # 取英文部分（后半部分），并去除前后的普通空格
                parts = line.split('\xa0')
                english_term = parts[-1].strip()  # 取最后一部分（英文）
                if english_term:
                    jargons.append(english_term)
            else:
                # 如果行中没有 \\xa0，可能是纯英文术语，直接使用整行
                jargons.append(line)
    return jargons


# ============================================================================
#  第三部分：特征提取核心函数
#  （以下函数严格遵循原始 notebook 中的实现逻辑）
# ============================================================================

def compute_word_count(text):
    """
    计算文本中的词数（以空格分词）。

    这是所有特征中共用的基础函数。
    词数 = 空格数 + 1

    参数:
        text: str — 输入文本

    返回:
        int — 词数
    """
    if not text or not isinstance(text, str):
        return 0
    return text.count(' ') + 1


def extract_ASL(df, abstract_col='abstract'):
    """
    特征1：平均句长 ASL (Average Sentence Length)

    计算方式（与原 notebook 完全一致）：
      - 句子数 = 摘要中 '.' 的数量
      - 词数 = 空格数 + 1
      - ASL = 词数 / 句子数

    注意：此方法简单粗暴，会把缩写中的 '.' 也当作句号。
    这是基线版本的设计选择，后续可以改进。
    """
    asl_values = []
    for text in df[abstract_col]:
        if not text or not isinstance(text, str) or text.strip() == '':
            asl_values.append(np.nan)
            continue
        sentence_count = text.count('.')
        word_count = text.count(' ') + 1
        if sentence_count > 0:
            asl_values.append(word_count / sentence_count)
        else:
            asl_values.append(np.nan)
    return asl_values


def extract_MWL(df, abstract_col='abstract'):
    """
    特征2：平均词长 MWL (Mean Word Length)

    计算方式（与原 notebook 完全一致）：
      1. 移除标点符号：,.'"!-%
      2. 统计去除标点后的字符总数
      3. 统计去除标点后的空格数
      4. 词数 = 空格数 + 1
      5. MWL = 去标点字符数 / 词数

    注意：此实现通过逐字符检查来移除标点，与原 notebook 方法相同。
    """
    mwl_values = []
    for text in df[abstract_col]:
        if not text or not isinstance(text, str) or text.strip() == '':
            mwl_values.append(np.nan)
            continue

        # 逐字符移除标点
        punc_count = 0
        text_nopunc = ""
        for char in text:
            if char in PUNCTUATION:
                punc_count += 1
            else:
                text_nopunc += char

        char_count_no_punc = len(text_nopunc)
        space_count = text_nopunc.count(' ')
        word_count = space_count + 1

        if word_count > 0:
            mwl_values.append(char_count_no_punc / word_count)
        else:
            mwl_values.append(np.nan)
    return mwl_values


def extract_LD(df, stopwords, abstract_col='abstract'):
    """
    特征3：词汇密度 LD (Lexical Density)

    计算方式（与原 notebook 完全一致）：
      1. 总词数 = 空格数 + 1
      2. 在文本前后加空格，对每个停用词统计 " " + word + " " 的出现次数
      3. 实词数 = 总词数 - 停用词出现次数
      4. LD = 实词数 / 总词数

    注意：使用空格包围来匹配完整单词边界，与原notebook方法一致。
    """
    ld_values = []
    for text in df[abstract_col]:
        if not text or not isinstance(text, str) or text.strip() == '':
            ld_values.append(np.nan)
            continue

        words_lower = text.lower()
        total_words = words_lower.count(' ') + 1

        # 在文本前后加空格以便进行 " word " 形式的边界匹配
        padded_text = " " + words_lower + " "
        stop_count = 0
        for w in stopwords:
            stop_count += padded_text.count(" " + w + " ")

        content_words = total_words - stop_count

        if total_words > 0:
            ld_values.append(content_words / total_words)
        else:
            ld_values.append(np.nan)
    return ld_values


def extract_LC(df, abstract_col='abstract'):
    """
    特征4：词汇高级度 LC (Lexical Complexity / Lexical Richness)

    计算方式（与原 notebook 完全一致）：
      - 使用 Type-Token Ratio (TTR) 来度量
      - 先转为小写
      - 逐字符扫描，过滤标点，累加当前词
      - 遇到空格/标点时判断当前词是否已出现过（通过与 seen_words 字符串比较）
      - LC = 不重复词类型数 / 总词令牌数

    注意：原 notebook 使用了一种独特的去重方式——将已见过的词拼接成
    一个用空格包围的长字符串，然后用 " " + word + " " in seen_words 判断。
    此实现保留这种逻辑以确保结果完全一致。
    """
    lc_values = []
    for text in df[abstract_col]:
        if not text or not isinstance(text, str) or text.strip() == '':
            lc_values.append(np.nan)
            continue

        words = text.strip().lower()
        total_words = words.count(' ') + 1

        # 前后加空格，统一处理边界
        padded_text = " " + words + " "

        unique_count = 0
        seen_words = " "   # 用空格包围已见词，确保完整匹配
        current_word = ""

        for char in padded_text:
            if char != ' ' and char not in PUNCTUATION:
                current_word += char
            else:
                if current_word:
                    # 用 " word " 形式检查是否已经出现过
                    if (" " + current_word + " ") not in seen_words:
                        unique_count += 1
                        seen_words += current_word + " "
                    current_word = ""

        # 处理最后一个词（如果文本不以空格/标点结尾）
        if current_word:
            if (" " + current_word + " ") not in seen_words:
                unique_count += 1

        if total_words > 0:
            lc_values.append(unique_count / total_words)
        else:
            lc_values.append(np.nan)
    return lc_values


def extract_JD(df, jargon_list, abstract_col='abstract'):
    """
    特征5：术语密度 JD (Jargon Density)

    计算方式（与原 notebook 完全一致）：
      1. 将摘要文本转为小写
      2. 对术语列表中的每个术语，检查是否出现在摘要中（简单子串匹配）
      3. 统计出现的术语总数
      4. JD = 术语出现次数 / 总词数

    注意：这里的匹配是简单的 `term in text` 子串匹配，
    而非词边界匹配。例如 "acid" 会匹配到 "acidity"。
    这是基线版本的设计，与原notebook一致。
    """
    jd_values = []
    # 预转为小写以提高效率
    jargon_lower = [j.lower() for j in jargon_list]

    for text in df[abstract_col]:
        if not text or not isinstance(text, str) or text.strip() == '':
            jd_values.append(np.nan)
            continue

        text_lower = text.lower()
        word_count = text_lower.count(' ') + 1

        jargon_count = 0
        for jargon_term in jargon_lower:
            if jargon_term in text_lower:
                jargon_count += 1

        if word_count > 0:
            jd_values.append(jargon_count / word_count)
        else:
            jd_values.append(np.nan)
    return jd_values


def extract_HD(df, hedge_list, abstract_col='abstract'):
    """
    特征6：层次结构密度 HD (Hedge Density)

    计算方式（与原 notebook 完全一致）：
      1. 将摘要文本转为小写
      2. 对模糊限制语列表中的每个词语，检查是否出现在摘要中（简单子串匹配）
      3. 统计出现的模糊限制语总数
      4. HD = 模糊限制语出现次数 / 总词数

    注意：和 JD 一样，使用简单的 `term in text` 子串匹配。
    """
    hd_values = []
    hedge_lower = [h.lower() for h in hedge_list]

    for text in df[abstract_col]:
        if not text or not isinstance(text, str) or text.strip() == '':
            hd_values.append(np.nan)
            continue

        text_lower = text.lower()
        word_count = text_lower.count(' ') + 1

        hedge_count = 0
        for hedge_term in hedge_lower:
            if hedge_term in text_lower:
                hedge_count += 1

        if word_count > 0:
            hd_values.append(hedge_count / word_count)
        else:
            hd_values.append(np.nan)
    return hd_values


def calculate_NCC(df):
    """
    规范化引用计数 NCC (Normalized Citation Count)
    —— 双维度归一化（期刊 × 年份）

    归一化策略：
      - 若数据包含 source_journal 列且存在多个期刊：
        NCC = citations / mean_citations_for(journal, year)
        → 消除期刊差异和年份累积效应的双重偏倚
      - 若数据仅含一个期刊或无 source_journal 列：
        NCC = citations / mean_citations_for(year)
        → 退化为仅年份归一化（向后兼容基线数据）

    处理细节：
      - 某 (期刊, 年份) 组合样本数 = 1 时，NCC = 1.0（该论文即为均值）
      - 某论文 citations 为 NaN 时，NCC = NaN
      - 某论文 year 或 source_journal 缺失时，NCC = NaN

    参数:
        df: pd.DataFrame — 需包含 'year', 'citations' 列，可选 'source_journal'

    返回:
        pd.DataFrame — 新增 'NCC', 'mean_citations', 'ncc_group' 列
    """
    has_journal = ('source_journal' in df.columns
                   and df['source_journal'].notna().sum() > 0)
    n_journals = df['source_journal'].nunique() if has_journal else 0

    if has_journal and n_journals > 1:
        # ---- 双维度归一化：期刊 × 年份 ----
        group_cols = ['source_journal', 'year']
        mode = 'dual'
    else:
        # ---- 单维度归一化：仅年份（向后兼容） ----
        group_cols = ['year']
        mode = 'year_only'

    # 计算每组平均引用数
    group_mean = df.groupby(group_cols)['citations'].mean().reset_index()
    mean_col_name = 'mean_citations'
    group_mean.columns = group_cols + [mean_col_name]

    # 标记分组
    df['ncc_group'] = ''
    if mode == 'dual':
        df['ncc_group'] = df['source_journal'].fillna('Unknown') + ' | ' + df['year'].astype(str)
    else:
        df['ncc_group'] = df['year'].astype(str)

    # 合并基准值
    df = df.merge(group_mean, on=group_cols, how='left')

    # 计算 NCC
    df['NCC'] = df['citations'] / df['mean_citations']

    # 记录归一化模式（用于报告）
    df.attrs['ncc_mode'] = mode
    df.attrs['ncc_n_groups'] = len(group_mean)
    df.attrs['ncc_n_journals'] = n_journals

    return df


# ============================================================================
#  第四部分：数据加载
# ============================================================================

def load_openalex_json(filepath):
    """
    从 OpenAlex JSON 文件中加载数据并解析为 DataFrame。

    从原始 JSON 嵌套结构中提取以下字段：
      - doi            : 论文 DOI
      - title          : 论文标题 (display_name)
      - year           : 发表年份 (publication_year)
      - citations      : 被引次数 (cited_by_count)
      - is_oa          : 是否开放获取 (open_access.is_oa)
      - type           : 文献类型 (type)
      - source_journal : 来源期刊名 (primary_location → source → display_name)
      - abstract       : 还原后的摘要文本

    参数:
        filepath: str — OpenAlex JSON 文件路径

    返回:
        pd.DataFrame — 包含上述字段的数据表
    """
    print(f"[加载] 正在读取 JSON 文件: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"[加载] JSON 中共有 {len(raw_data)} 条记录")

    rows = []
    no_abstract_count = 0
    journal_set = set()

    for item in raw_data:
        # 还原摘要
        abstract = rebuild_abstract(item.get('abstract_inverted_index'))

        if not abstract:
            no_abstract_count += 1

        # 提取来源期刊名
        source_name = None
        primary_loc = item.get('primary_location')
        if primary_loc and isinstance(primary_loc, dict):
            source = primary_loc.get('source')
            if source and isinstance(source, dict):
                source_name = source.get('display_name')

        if source_name:
            journal_set.add(source_name)

        rows.append({
            'doi':            item.get('doi'),
            'title':          item.get('display_name'),
            'year':           item.get('publication_year'),
            'citations':      item.get('cited_by_count'),
            'is_oa':          item.get('open_access', {}).get('is_oa'),
            'type':           item.get('type'),
            'source_journal': source_name,
            'abstract':       abstract,
        })

    df = pd.DataFrame(rows)

    # 报告数据质量
    total = len(df)
    with_abstract = total - no_abstract_count
    print(f"[加载] 摘要覆盖率: {with_abstract}/{total} "
          f"({100 * with_abstract / max(total, 1):.1f}%)")

    if journal_set:
        print(f"[加载] 检测到 {len(journal_set)} 个期刊: {', '.join(sorted(journal_set))}")
    else:
        print(f"[加载] ⚠ 未检测到期刊信息（source_journal 将为空）")

    if no_abstract_count > 0:
        print(f"[加载] ⚠ 有 {no_abstract_count} 条记录缺少摘要，"
              f"这些记录在特征计算中将产生 NaN 值")
        print(f"[加载] 提示：建议在 OpenAlex 查询时添加 has_abstract=True 筛选条件")

    return df


def load_csv(filepath):
    """
    从 CSV 文件中加载数据（适用于已经包含 abstract 列的数据集）。

    参数:
        filepath: str — CSV 文件路径

    返回:
        pd.DataFrame
    """
    print(f"[加载] 正在读取 CSV 文件: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[加载] CSV 中共有 {len(df)} 条记录")

    if 'abstract' not in df.columns:
        print("[加载] ⚠ CSV 中没有 'abstract' 列，无法计算文本特征！")
        sys.exit(1)

    return df


# ============================================================================
#  第五部分：特征提取流水线
# ============================================================================

def extract_all_features(df, data_dir, abstract_col='abstract'):
    """
    一站式执行所有六大特征的提取。

    这是整个脚本的核心流水线函数。按照以下顺序依次计算：
      1. ASL - 平均句长
      2. MWL - 平均词长
      3. LD  - 词汇密度
      4. LC  - 词汇高级度
      5. JD  - 术语密度
      6. HD  - 层次结构密度
      7. NCC - 规范化引用计数

    参数:
        df:        pd.DataFrame — 包含 abstract 列的数据表
        data_dir:  str — 参考数据文件所在目录
        abstract_col: str — 摘要列名

    返回:
        pd.DataFrame — 添加了所有特征列的数据表
    """
    total_start = time.time()

    # ---- 加载参考数据 ----
    print("\n" + "=" * 60)
    print(" 第一步：加载参考数据文件")
    print("=" * 60)

    stopwords_path = os.path.join(data_dir, 'essential-word-list.txt')
    hedges_path    = os.path.join(data_dir, 'hedge_data.txt')
    jargons_path   = os.path.join(data_dir, 'df_j.txt')

    for label, path in [("停用词表", stopwords_path),
                         ("模糊限制语词表", hedges_path),
                         ("腐蚀术语表", jargons_path)]:
        if not os.path.exists(path):
            print(f"[错误] ❌ {label} 不存在: {path}")
            print(f"[提示] 请确保参考数据文件位于 {data_dir}/ 目录下")
            sys.exit(1)

    stopwords = load_stopwords(stopwords_path)
    print(f"  ✓ 加载停用词: {len(stopwords)} 个")

    hedge_words = load_hedge_words(hedges_path)
    print(f"  ✓ 加载模糊限制语: {len(hedge_words)} 个")

    jargon_terms = load_jargon_list(jargons_path)
    print(f"  ✓ 加载腐蚀术语: {len(jargon_terms)} 个")

    # ---- 特征提取 ----
    print("\n" + "=" * 60)
    print(" 第二步：逐项提取文体特征")
    print("=" * 60)

    # 1. ASL - 平均句长
    t0 = time.time()
    df['ASL'] = extract_ASL(df, abstract_col)
    valid_asl = df['ASL'].notna().sum()
    print(f"  [1/6] ASL (平均句长)      → {valid_asl}/{len(df)} 条有效 "
          f"(均值={df['ASL'].mean():.2f}, 耗时={time.time()-t0:.1f}s)")

    # 2. MWL - 平均词长
    t0 = time.time()
    df['MWL'] = extract_MWL(df, abstract_col)
    valid_mwl = df['MWL'].notna().sum()
    print(f"  [2/6] MWL (平均词长)      → {valid_mwl}/{len(df)} 条有效 "
          f"(均值={df['MWL'].mean():.2f}, 耗时={time.time()-t0:.1f}s)")

    # 3. LD - 词汇密度
    t0 = time.time()
    df['LD'] = extract_LD(df, stopwords, abstract_col)
    valid_ld = df['LD'].notna().sum()
    print(f"  [3/6] LD  (词汇密度)      → {valid_ld}/{len(df)} 条有效 "
          f"(均值={df['LD'].mean():.3f}, 耗时={time.time()-t0:.1f}s)")

    # 4. LC - 词汇高级度
    t0 = time.time()
    df['LC'] = extract_LC(df, abstract_col)
    valid_lc = df['LC'].notna().sum()
    print(f"  [4/6] LC  (词汇高级度)    → {valid_lc}/{len(df)} 条有效 "
          f"(均值={df['LC'].mean():.3f}, 耗时={time.time()-t0:.1f}s)")

    # 5. JD - 术语密度
    t0 = time.time()
    df['JD'] = extract_JD(df, jargon_terms, abstract_col)
    valid_jd = df['JD'].notna().sum()
    print(f"  [5/6] JD  (术语密度)      → {valid_jd}/{len(df)} 条有效 "
          f"(均值={df['JD'].mean():.4f}, 耗时={time.time()-t0:.1f}s)")

    # 6. HD - 层次结构密度
    t0 = time.time()
    df['HD'] = extract_HD(df, hedge_words, abstract_col)
    valid_hd = df['HD'].notna().sum()
    print(f"  [6/6] HD  (层次结构密度)  → {valid_hd}/{len(df)} 条有效 "
          f"(均值={df['HD'].mean():.4f}, 耗时={time.time()-t0:.1f}s)")

    # ---- NCC 计算 ----
    print("\n" + "=" * 60)
    print(" 第三步：计算规范化引用计数 (NCC)")
    print("=" * 60)

    has_journal = ('source_journal' in df.columns
                   and df['source_journal'].notna().sum() > 0)
    n_journals = df['source_journal'].nunique() if has_journal else 0

    df = calculate_NCC(df)
    mode = df.attrs.get('ncc_mode', 'unknown')
    n_groups = df.attrs.get('ncc_n_groups', 0)

    if mode == 'dual':
        print(f"  归一化模式: 双维度（期刊 × 年份）")
        print(f"  期刊数: {n_journals}  归一化基准组数: {n_groups}\n")
        # 展示每个期刊每年的基准值
        pivot = df.groupby(['source_journal', 'year'])['mean_citations'].mean().unstack()
        print(pivot.round(1).to_string())
    else:
        print(f"  归一化模式: 单维度（仅年份）")
        if n_journals == 0:
            print(f"  ⚠ 未检测到 source_journal 列，已自动降级为仅年份归一化")
        else:
            print(f"  期刊数: {n_journals}（单期刊，退化为年份归一化）")
        print()
        yearly_stats = df.groupby('year').agg(
            论文数=('citations', 'count'),
            平均引用=('citations', 'mean')
        ).round(2)
        print(yearly_stats.to_string())

    valid_ncc = df['NCC'].notna().sum()
    print(f"\n  ✓ NCC 计算完成: {valid_ncc}/{len(df)} 条有效 "
          f"(均值={df['NCC'].mean():.2f})")

    # 标记低置信度分组（样本量 ≤ 2 的组）
    if mode == 'dual':
        group_sizes = df.groupby(['source_journal', 'year']).size()
        small_groups = group_sizes[group_sizes <= 2]
        if len(small_groups) > 0:
            print(f"  ⚠ {len(small_groups)} 个 (期刊, 年份) 组合样本量 ≤ 2，NCC 置信度较低")
            for (j, y), s in small_groups.items():
                print(f"      {j} ({y}): {s} 篇")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f" 特征提取全部完成！总耗时: {total_elapsed:.1f} 秒")
    print(f"{'=' * 60}")

    return df


# ============================================================================
#  第六部分：输出与报告
# ============================================================================

def generate_report(df):
    """
    生成特征提取结果摘要报告。
    """
    feature_cols = ['ASL', 'MWL', 'LD', 'LC', 'JD', 'HD', 'NCC']
    available_cols = [c for c in feature_cols if c in df.columns]

    print("\n" + "=" * 60)
    print(" 特征统计摘要")
    print("=" * 60)

    stats = df[available_cols].describe().T
    stats['missing'] = df[available_cols].isna().sum().values
    stats['missing_pct'] = (100 * df[available_cols].isna().sum() / len(df)).values
    print(stats[['count', 'mean', 'std', 'min', '50%', 'max', 'missing', 'missing_pct']]
          .to_string(float_format=lambda x: f'{x:.4f}'))

    print(f"\n 总记录数: {len(df)}")
    print(f" 有摘要的记录: {df['abstract'].notna().sum()}")
    print(f" 摘要为空: {df['abstract'].isna().sum() + (df['abstract'] == '').sum()}")
    print(f" 特征完全无缺失的记录: {df[available_cols].dropna().shape[0]}")


def save_output(df, output_path):
    """
    保存最终数据集为 CSV 文件。

    按标准列顺序输出，确保结果的可复现性。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 选择输出列（只保留存在的列）
    available_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    # 也保留不在标准列表中但可能存在的新列（如 mean_citations）
    extra_cols = [c for c in df.columns if c not in available_cols]

    output_df = df[available_cols + extra_cols]
    output_df.to_csv(output_path, index=False, encoding='utf-8')

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n 输出文件: {output_path}")
    print(f" 文件大小: {file_size_mb:.2f} MB")
    print(f" 总列数: {len(output_df.columns)}")
    print(f" 总行数: {len(output_df)}")


# ============================================================================
#  第七部分：命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="腐蚀领域学术摘要文体特征一站式提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从 OpenAlex JSON 提取特征
  python extract_features.py -i dataset.json -o final.csv

  # 从已有 CSV 提取特征
  python extract_features.py -i data.csv -o final.csv --from-csv

  # 指定参考数据目录
  python extract_features.py -i dataset.json -o final.csv --data-dir ./my_data
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入文件路径（OpenAlex JSON 或包含 abstract 列的 CSV）'
    )
    parser.add_argument(
        '-o', '--output',
        default='dataset_with_features.csv',
        help='输出 CSV 文件路径（默认: dataset_with_features.csv）'
    )
    parser.add_argument(
        '--data-dir',
        default=None,
        help='参考数据文件所在目录（默认: 脚本同级的 ../data 或 ./data）'
    )
    parser.add_argument(
        '--from-csv',
        action='store_true',
        help='输入文件是 CSV 格式而非 JSON（CSV 需包含 abstract 列）'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='不输出详细统计报告'
    )
    parser.add_argument(
        '--abstract-col',
        default='abstract',
        help='摘要列的名称（默认: abstract）'
    )

    args = parser.parse_args()

    # ---- 确定参考数据目录 ----
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # 自动查找：先尝试脚本同级的 ../data，再尝试 ./data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_data = os.path.join(script_dir, '..', 'data')
        if os.path.isdir(parent_data):
            data_dir = os.path.abspath(parent_data)
        elif os.path.isdir('./data'):
            data_dir = os.path.abspath('./data')
        else:
            data_dir = os.path.abspath('./data')

    print("=" * 60)
    print("  腐蚀领域学术摘要文体特征一站式提取工具")
    print("  One-stop Feature Extraction Pipeline")
    print("=" * 60)
    print(f"  输入文件:     {args.input}")
    print(f"  输出文件:     {args.output}")
    print(f"  参考数据目录: {data_dir}")
    print(f"  输入格式:     {'CSV' if args.from_csv else 'JSON (OpenAlex)'}")
    print("=" * 60)

    # ---- 加载数据 ----
    if args.from_csv:
        df = load_csv(args.input)
    else:
        df = load_openalex_json(args.input)

    if len(df) == 0:
        print("[错误] 输入数据集为空，无法继续。")
        sys.exit(1)

    # ---- 提取特征 ----
    df = extract_all_features(df, data_dir, args.abstract_col)

    # ---- 保存输出 ----
    save_output(df, args.output)

    # ---- 生成报告 ----
    if not args.no_report:
        generate_report(df)

    print("\n✓ 全部完成！")


if __name__ == '__main__':
    main()
