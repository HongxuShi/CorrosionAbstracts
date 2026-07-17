#!/usr/bin/env python3
"""
================================================================================
 元话语特征提取模块
 Metadiscourse Feature Extraction Module
================================================================================

功能说明：
  基于Hyland(2005)元话语模型，从摘要文本中提取10类元话语标记的使用密度。
  每个特征以比率形式表示（归一化到文本总词数）。

10类元话语特征：
  Interactive (引导元话语) — 帮助读者导航文本:
    transaction_ratio      逻辑过渡标记 (however, therefore, in addition...)
    frame_marker_ratio     文本结构标记 (first, finally, this paper aims to...)
    endophoric_ratio       内部指称标记 (as shown in Fig., see Table...)
    evidential_ratio       外部证据引用 (according to, previous studies...)
    code_gloss_ratio       语码注释标记 (namely, such as, in other words...)

  Interactional (互动元话语) — 表达作者立场和读者关系:
    hedge_ratio            模糊限制语 (may, possibly, approximately...)
    booster_ratio          增强语 (clearly, demonstrate, it is evident...)
    attitude_ratio         态度评价标记 (importantly, crucial, surprisingly...)
    self_mention_ratio     作者自我指称 (we, our, this study, the author...)
    engagement_ratio       读者介入标记 (note that, should, consider...)

与Biber方案的关键区别：
  - 不需要spaCy NLP管道（仅需基础tokenization）
  - 不需要PCA降维（特征本身已高度可解释）
  - 每个特征可以直接转化为写作建议

参考：
  Hyland, K. (2005). Metadiscourse: Exploring Interaction in Writing. Continuum.

作者：理论验证阶段 — Metadiscourse方案
日期：2026/07/17
================================================================================
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
#  MetadiscourseExtractor 类
# ============================================================================

class MetadiscourseExtractor:
    """
    元话语特征提取器

    从学术摘要文本中提取10类Hyland元话语标记的使用密度。

    属性:
        lexicon_dir (Path): 词典文件目录
        dictionaries (dict): {类别名: [(词条, 是否为多词短语), ...]}
        feature_names (list[str]): 10个特征名称列表
    """

    # 10个特征的名称（与词典文件对应）
    FEATURE_NAMES = [
        "transition_ratio",
        "frame_marker_ratio",
        "endophoric_ratio",
        "evidential_ratio",
        "code_gloss_ratio",
        "hedge_ratio",
        "booster_ratio",
        "attitude_ratio",
        "self_mention_ratio",
        "engagement_ratio",
    ]

    # 词典文件名 → 特征名的映射
    DICT_MAP = {
        "transitions":        "transition_ratio",
        "frame_markers":      "frame_marker_ratio",
        "endophorics":        "endophoric_ratio",
        "evidentials":        "evidential_ratio",
        "code_glosses":       "code_gloss_ratio",
        "hedges":             "hedge_ratio",
        "boosters":           "booster_ratio",
        "attitude_markers":   "attitude_ratio",
        "self_mentions":      "self_mention_ratio",
        "engagement_markers": "engagement_ratio",
    }

    # ==========================================================================
    #  初始化
    # ==========================================================================

    def __init__(self, lexicon_dir: Optional[str] = None):
        """
        初始化元话语特征提取器，加载所有词典。

        参数:
            lexicon_dir (str | None): 词典目录路径。默认相对于本模块定位。
        """
        # ---- 确定词典目录 ----
        if lexicon_dir is None:
            module_dir = Path(__file__).resolve().parent
            lexicon_dir = module_dir / "md_dictionaries"
        else:
            lexicon_dir = Path(lexicon_dir)

        self.lexicon_dir = lexicon_dir
        logger.info(f"元话语词典目录: {self.lexicon_dir}")

        # ---- 加载所有词典 ----
        # 每个词典存储为 [(term, is_multiword), ...] 列表
        # is_multiword=True 表示该词条包含空格，需要特殊匹配
        self.dictionaries: Dict[str, List[Tuple[str, bool, int]]] = {}
        self._load_all_dictionaries()

        # ---- 编译词边界正则（用于单/多词匹配） ----
        # 对单词使用 \b 边界匹配；对多词短语使用子串搜索
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, int]]] = {}
        self._compile_patterns()

        # ---- 统计 ----
        self.stats: Dict = {}
        self._total_terms_loaded = sum(
            len(terms) for terms in self.dictionaries.values()
        )

    # ==========================================================================
    #  词典加载
    # ==========================================================================

    def _load_all_dictionaries(self) -> None:
        """
        加载所有10个元话语词典。

        词典文件格式: 每行一个词条，# 开头为注释，空行忽略。
        支持多词短语（如 "in addition", "as a result"）。
        """
        for dict_name in self.DICT_MAP:
            filepath = self.lexicon_dir / f"{dict_name}.txt"
            try:
                terms = self._load_term_list(filepath)
                self.dictionaries[dict_name] = terms
                multi = sum(1 for _, is_mw, _ in terms if is_mw)
                single = len(terms) - multi
                logger.info(f"  Loaded {dict_name}: {len(terms)} terms "
                            f"({single} single-word, {multi} multi-word)")
            except FileNotFoundError:
                logger.warning(f"  Dictionary file not found: {filepath}")
                self.dictionaries[dict_name] = []

    def _load_term_list(self, filepath: Path) -> List[Tuple[str, bool, int]]:
        """
        从文本文件加载词条列表。

        返回: [(lowercase_term, is_multiword, term_length_in_words), ...]
        按词条长度降序排列（长词条优先匹配，避免 "in addition to" 被 "in addition" 抢先匹配）。

        参数:
            filepath (Path): 词典文件路径

        返回:
            list[tuple]: 词条列表
        """
        terms = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                term_lower = line.lower()
                # 判断是否为多词短语
                word_count = len(term_lower.split())
                is_multiword = word_count > 1

                terms.append((term_lower, is_multiword, word_count))

        # 按词条长度降序排列（最长优先，避免短词条抢先匹配长短语的子串）
        terms.sort(key=lambda x: (x[2], len(x[0])), reverse=True)
        return terms

    def _compile_patterns(self) -> None:
        """
        为每个词典编译正则表达式模式。

        单词语: \bword\b
        多词短语: 子串搜索（因为 "in addition" 前后可以是标点而不是空格边界）
        """
        for dict_name, terms in self.dictionaries.items():
            patterns = []
            for term, is_multiword, word_count in terms:
                if is_multiword:
                    # 多词短语: 使用转义后的子串匹配
                    escaped = re.escape(term)
                    patterns.append((re.compile(escaped, re.IGNORECASE), 1))
                else:
                    # 单词语: 使用词边界匹配
                    escaped = re.escape(term)
                    patterns.append((re.compile(r'\b' + escaped + r'\b', re.IGNORECASE), 1))
            self._compiled_patterns[dict_name] = patterns

    # ==========================================================================
    #  特征提取主入口
    # ==========================================================================

    def extract_all(self, abstracts: pd.DataFrame, text_col: str = "abstract") -> pd.DataFrame:
        """
        对所有摘要提取10类元话语特征。

        这是本模块的主入口方法。不需要spaCy预处理——
        直接读取原始文本。

        参数:
            abstracts (pd.DataFrame): 包含摘要文本的数据表
            text_col (str): 摘要文本列名

        返回:
            pd.DataFrame: 特征矩阵 (n_samples × 10 + doc_id)
        """
        if abstracts.empty:
            logger.warning("输入数据为空")
            return pd.DataFrame()

        n_total = len(abstracts)
        logger.info(f"开始提取 {n_total} 篇摘要的元话语特征...")

        # ---- 提取特征 ----
        feature_rows = []
        for i, (_, row) in enumerate(abstracts.iterrows()):
            text = row.get(text_col, '')
            if not isinstance(text, str) or not text.strip():
                # 空文本 → NaN
                feature_rows.append({f: np.nan for f in self.FEATURE_NAMES})
                continue

            features = self._extract_single(text)
            # 附加 doc_id（如果存在）
            if 'doi' in row:
                features['doc_id'] = row['doi']
            elif 'id' in row:
                features['doc_id'] = str(row['id'])
            else:
                features['doc_id'] = f"doc_{i}"

            feature_rows.append(features)

            if (i + 1) % 1000 == 0:
                logger.info(f"  进度: {i + 1}/{n_total}")

        # ---- 构建DataFrame ----
        df = pd.DataFrame(feature_rows)
        # 确保列顺序
        cols = ['doc_id'] + self.FEATURE_NAMES
        df = df[[c for c in cols if c in df.columns]]

        self.stats = {
            'n_documents': len(df),
            'n_features': len(self.FEATURE_NAMES),
            'total_terms_loaded': self._total_terms_loaded,
            'missing_rate': df[self.FEATURE_NAMES].isna().mean().to_dict(),
        }

        logger.info(f"元话语特征提取完成: {len(df)} 篇 × {len(self.FEATURE_NAMES)} 特征")
        return df

    # ==========================================================================
    #  单文本特征提取
    # ==========================================================================

    def _extract_single(self, text: str) -> Dict[str, float]:
        """
        从单篇摘要中提取10类元话语特征。

        参数:
            text (str): 摘要文本

        返回:
            dict: {feature_name: ratio, ...}
        """
        text_lower = text.lower()

        # ---- 计算总词数 ----
        # 简单按空格分词（对英文摘要足够）
        words = text_lower.split()
        word_count = len(words)
        if word_count == 0:
            return {f: 0.0 for f in self.FEATURE_NAMES}

        features = {}

        # ---- 逐类计算元话语密度 ----
        for dict_name, feature_name in self.DICT_MAP.items():
            count = self._count_matches(text_lower, dict_name)
            features[feature_name] = count / word_count

        return features

    # ==========================================================================
    #  匹配计数
    # ==========================================================================

    def _count_matches(self, text_lower: str, dict_name: str) -> int:
        """
        统计某个元话语类别在文本中的匹配次数。

        匹配策略:
          1. 对多词短语做子串搜索
          2. 对单词语做词边界匹配
          3. 已匹配的位置会被标记，避免重复计数
             （例如 "in addition to" 匹配后不再单独计 "in addition"）

        参数:
            text_lower (str): 小写化的文本
            dict_name (str): 词典名称

        返回:
            int: 匹配次数
        """
        if dict_name not in self._compiled_patterns:
            return 0

        matched_ranges: List[Tuple[int, int]] = []  # 已匹配的字符范围
        total_count = 0

        for pattern, _ in self._compiled_patterns[dict_name]:
            for match in pattern.finditer(text_lower):
                start, end = match.start(), match.end()

                # 检查是否与已匹配范围重叠
                overlapped = False
                for m_start, m_end in matched_ranges:
                    if start < m_end and end > m_start:
                        overlapped = True
                        break

                if not overlapped:
                    total_count += 1
                    matched_ranges.append((start, end))

        return total_count

    # ==========================================================================
    #  特征描述
    # ==========================================================================

    def get_feature_descriptions(self) -> Dict[str, str]:
        """返回每个元话语特征的描述和写作指导含义。"""
        return {
            "transition_ratio": (
                "逻辑过渡标记密度 (transitions)。"
                "高频表示文本包含较多显性逻辑连接（因果、对比、递进）。"
                "写作建议：适当使用过渡标记使论证更连贯。"
            ),
            "frame_marker_ratio": (
                "文本结构标记密度 (frame markers)。"
                "高频表示文本明确标示了结构（如'first...finally'、'this paper aims to'）。"
                "写作建议：清晰标示摘要结构有助于读者快速定位信息。"
            ),
            "endophoric_ratio": (
                "内部指称密度 (endophoric markers)。"
                "高频表示文本频繁指向图表、公式、前后文。"
                "写作建议：摘要通常不需要内部指称——这是期刊正式论文中才常见的特征。"
            ),
            "evidential_ratio": (
                "外部证据引用密度 (evidentials)。"
                "高频表示文本大量引用前人研究作为依据。"
                "写作建议：适当引用前人工作展示研究定位，但摘要不宜过度堆砌引用。"
            ),
            "code_gloss_ratio": (
                "语码注释密度 (code glosses)。"
                "高频表示文本使用了较多的解释性表达（'such as'、'namely'、'i.e.'）。"
                "写作建议：适当的注释有助于读者理解专业概念。"
            ),
            "hedge_ratio": (
                "模糊限制语密度 (hedges)。"
                "高频表示作者在表达中留下了谨慎的空间（'may'、'possibly'、'suggest'）。"
                "写作建议：适度的模糊限制是学术写作的规范，但过多会显得论证不自信。"
            ),
            "booster_ratio": (
                "增强语密度 (boosters)。"
                "高频表示作者使用了强烈的确定性表达（'clearly'、'demonstrate'、'prove'）。"
                "写作建议：重要发现应使用增强语，但过度使用可能显得武断。"
            ),
            "attitude_ratio": (
                "态度标记密度 (attitude markers)。"
                "高频表示文本包含较多评价性表达（'important'、'crucial'、'remarkably'）。"
                "写作建议：适度的评价可以突出研究价值，但过多显得主观。"
            ),
            "self_mention_ratio": (
                "作者自我指称密度 (self-mentions)。"
                "高频表示作者在文本中显性出现（'we'、'our study'、'this paper'）。"
                "写作建议：适度的自我指称可以增强作者声音，STEM领域通常偏低。"
            ),
            "engagement_ratio": (
                "读者介入密度 (engagement markers)。"
                "高频表示文本直接呼唤读者（'note that'、'should be noted'）。"
                "写作建议：过多直接介入表达可能不适合学术摘要的客观性传统。"
            ),
        }

    def get_stats(self) -> Dict:
        """返回最近一次提取的统计信息。"""
        return self.stats.copy()
