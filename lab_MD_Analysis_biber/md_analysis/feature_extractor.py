#!/usr/bin/env python3
"""
================================================================================
 语言特征提取模块 — MD分析管道第二步
 Linguistic Feature Extraction Module for Multi-Dimensional Analysis
================================================================================

功能说明：
  基于Biber (1988)多维分析框架，从预处理后的科技论文摘要中提取15个语言特征。
  每个特征以比率形式表示（归一化到文本总词数），确保不同长度的摘要可比较。

特征列表及归属维度：
  ┌──────────────────────────┬──────────┬──────────────────────┐
  │ 特征名称                 │ 维度     │ 检测方式             │
  ├──────────────────────────┼──────────┼──────────────────────┤
  │ past_tense_ratio         │ Dim 1    │ POS: VBD             │
  │ present_tense_ratio      │ Dim 1    │ POS: VBP, VBZ        │
  │ passive_ratio            │ Dim 1    │ dep: auxpass         │
  │ stance_verb_ratio        │ Dim 1    │ stance_verb + that   │
  │ mental_verb_ratio        │ Dim 1    │ 词典匹配             │
  │ modal_possibility_ratio  │ Dim 2    │ 词典匹配             │
  │ modal_prediction_ratio   │ Dim 2    │ 词典匹配             │
  │ relative_clause_ratio    │ Dim 2    │ dep: relcl           │
  │ communication_verb_ratio │ Dim 3    │ 词典匹配             │
  │ suasive_verb_ratio       │ Dim 3    │ 词典匹配             │
  │ noun_modifier_ratio      │ Dim 4    │ NOUN+NOUN序列        │
  │ abstract_noun_ratio      │ Dim 4    │ 词典匹配             │
  │ nominalization_ratio     │ Dim 4    │ 后缀检测 (-tion等)   │
  │ human_noun_ratio         │ (辅助)   │ 词典匹配             │
  │ word_length              │ (辅助)   │ 平均字符长度         │
  │ word_count               │ (辅助)   │ 总词数               │
  └──────────────────────────┴──────────┴──────────────────────┘

参考：
  Biber, D. (1988). Variation across Speech and Writing. Cambridge University Press.
  Biber, D. (2006). University Language. John Benjamins.

作者：基于PRD规范开发
日期：2026/07/17
================================================================================
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
#  FeatureExtractor 类
# ============================================================================

class FeatureExtractor:
    """
    语言特征提取器

    从spaCy预处理后的文档中提取15个Biber风格的语言特征。
    所有频率型特征均除以文本总词数以获得比率，实现跨文本长度可比性。

    属性:
        feature_names (list[str]): 所有特征名称列表
        lexicon_dir (Path): 词典文件目录
        dictionaries (dict): 已加载的词典 {名称: set}
    """

    # ==========================================================================
    #  特征定义常量
    # ==========================================================================

    # 15个特征的名称列表（按PRD定义的顺序排列）
    FEATURE_NAMES = [
        # Dimension 1: Involved vs. Informational Production
        "past_tense_ratio",
        "present_tense_ratio",
        "passive_ratio",
        "stance_verb_ratio",
        "mental_verb_ratio",
        # Dimension 2: Narrative vs. Non-Narrative
        "modal_possibility_ratio",
        "modal_prediction_ratio",
        "relative_clause_ratio",
        # Dimension 3: Context-Independent vs. Context-Dependent
        "communication_verb_ratio",
        "suasive_verb_ratio",
        # Dimension 4: (Academic) Abstract vs. Concrete Style
        "noun_modifier_ratio",
        "abstract_noun_ratio",
        "nominalization_ratio",
        # 辅助特征
        "human_noun_ratio",
        "word_length",
    ]

    # 名词化后缀列表 —— 用于 nominalization_ratio
    # 这些后缀是英语名词化的典型标记（将动词/形容词转化为名词）
    NOMINALIZATION_SUFFIXES = (
        'tion',    # investigation, determination
        'ment',    # development, measurement
        'ance',    # performance, resistance
        'ence',    # dependence, difference
        'ity',     # ability, reactivity
        'ness',    # hardness, effectiveness
        'sion',    # corrosion, discussion
        'ship',    # relationship
        'age',     # coverage, leakage
        'ism',     # mechanism
        'ure',     # procedure, failure
        'al',      # removal, appraisal
        'sis',     # analysis, synthesis
    )

    # ==========================================================================
    #  初始化
    # ==========================================================================

    def __init__(self, lexicon_dir: Optional[str] = None):
        """
        初始化特征提取器，加载所有词典。

        参数:
            lexicon_dir (str | None): 词典文件目录路径。
                                      默认为 ../data/md_dictionaries （相对本模块位置）
        """
        # ---- 确定词典目录 ----
        if lexicon_dir is None:
            # 从本模块所在位置向上找到项目根目录，再定位到 data/md_dictionaries
            module_dir = Path(__file__).resolve().parent.parent  # CorrosionAbstracts/
            lexicon_dir = module_dir / "data" / "md_dictionaries"
        else:
            lexicon_dir = Path(lexicon_dir)

        self.lexicon_dir = lexicon_dir
        logger.info(f"词典目录: {self.lexicon_dir}")

        # ---- 加载所有词典 ----
        self.dictionaries: Dict[str, Set[str]] = {}
        self._load_all_dictionaries()

        # ---- 统计信息 ----
        self.stats: Dict[str, Any] = {}

    # ==========================================================================
    #  词典加载
    # ==========================================================================

    def _load_all_dictionaries(self) -> None:
        """
        加载所有MD分析所需的词典文件。

        词典文件格式: 每行一个词/短语，空行和 # 开头的注释行被忽略。
        所有词典项统一转为小写以便进行大小写不敏感的匹配。
        """
        # 词典文件映射: {字典键名: (文件名, 描述)}
        dict_map = {
            "mental_verbs":        ("mental_verbs.txt",        "心理动词"),
            "communication_verbs": ("communication_verbs.txt", "交流动词"),
            "suasive_verbs":       ("suasive_verbs.txt",       "劝说性动词"),
            "modal_possibility":   ("modal_possibility.txt",   "可能性情态词"),
            "modal_prediction":    ("modal_prediction.txt",    "预测性情态词"),
            "abstract_nouns":      ("abstract_nouns.txt",      "抽象名词"),
            "human_nouns":         ("human_nouns.txt",         "人类指称名词"),
            "stance_verbs":        ("stance_verbs.txt",        "立场动词"),
        }

        for dict_key, (filename, desc) in dict_map.items():
            filepath = self.lexicon_dir / filename
            try:
                word_set = self._load_word_list(filepath)
                self.dictionaries[dict_key] = word_set
                logger.info(f"  ✓ 加载{desc}词典: {len(word_set)} 个词条")
            except FileNotFoundError:
                logger.warning(f"  ✗ {desc}词典文件不存在: {filepath}")
                self.dictionaries[dict_key] = set()

    def _load_word_list(self, filepath: Path) -> Set[str]:
        """
        从文本文件加载词表，返回小写词集合。

        每行一个词条，忽略:
          - 空行
          - # 开头的注释行
          - 行首尾空白

        参数:
            filepath (Path): 词典文件路径

        返回:
            set[str]: 所有词条的小写形式集合
        """
        words = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    continue
                # 转为小写并加入集合
                words.add(line.lower())
        return words

    # ==========================================================================
    #  特征提取入口
    # ==========================================================================

    def extract_all_features(
        self,
        processed_docs: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        对所有预处理后的文档提取全部15个语言特征。

        这是特征提取模块的主入口方法。

        参数:
            processed_docs (list[dict]): TextPreprocessor.process_file() 的返回值，
                                        每篇文档一个字典

        返回:
            pd.DataFrame: 特征矩阵，行为文档，列为特征变量
                          shape = (n_documents, n_features)

                          ┌──────────────┬─────────────────┬─────┬───────────┐
                          │ doc_id       │ past_tense_ratio│ ... │ word_count│
                          ├──────────────┼─────────────────┼─────┼───────────┤
                          │ doc_0        │        0.042    │ ... │    187    │
                          │ doc_1        │        0.031    │ ... │    223    │
                          │ ...          │         ...     │ ... │    ...    │
                          └──────────────┴─────────────────┴─────┴───────────┘
        """
        if not processed_docs:
            logger.warning("输入文档列表为空！")
            return pd.DataFrame()

        logger.info(f"开始提取 {len(processed_docs)} 篇文档的语言特征...")

        # ---- 逐文档提取特征 ----
        feature_rows = []
        for i, doc in enumerate(processed_docs):
            features = self._extract_single_doc(doc)
            feature_rows.append(features)

            # 进度日志（每100篇输出一次）
            if (i + 1) % 100 == 0:
                logger.info(f"  特征提取进度: {i + 1}/{len(processed_docs)}")

        # ---- 构建DataFrame ----
        df = pd.DataFrame(feature_rows)

        # 确保列顺序一致
        columns = ["doc_id"] + self.FEATURE_NAMES
        # 只保留存在的列（防止某些特征提取失败导致缺失）
        available_cols = [c for c in columns if c in df.columns]
        df = df[available_cols]

        # ---- 记录统计 ----
        self.stats = {
            "n_documents": len(df),
            "n_features": len(self.FEATURE_NAMES),
            "features_extracted": len(available_cols) - 1,  # 减去doc_id
            "missing_rate": df[self.FEATURE_NAMES].isna().mean().to_dict()
            if all(f in df.columns for f in self.FEATURE_NAMES) else {},
        }

        logger.info(f"特征提取完成: {self.stats['n_documents']} 篇 × "
                    f"{self.stats['features_extracted']} 个特征")

        return df

    # ==========================================================================
    #  单文档特征提取
    # ==========================================================================

    def _extract_single_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        从单篇预处理后的文档中提取所有语言特征。

        参数:
            doc (dict): 单篇文档的预处理结果字典

        返回:
            dict: 包含 doc_id 和所有15个特征值的字典
        """
        # ---- 获取基础数据 ----
        doc_id = doc["id"]
        word_count = doc["word_count"]
        text_lower = doc["abstract_full"].lower()

        # 从spaCy Doc对象中提取结构化信息
        spacy_doc = doc["spacy_doc"]

        # ---- 构建特征字典（逐个计算每个特征） ----
        features = {"doc_id": doc_id}

        # Dimension 1 相关特征
        features["past_tense_ratio"] = self._compute_past_tense(spacy_doc, word_count)
        features["present_tense_ratio"] = self._compute_present_tense(spacy_doc, word_count)
        features["passive_ratio"] = self._compute_passive(doc["dependencies"], word_count)
        features["stance_verb_ratio"] = self._compute_stance_complement(
            spacy_doc, text_lower, word_count
        )
        features["mental_verb_ratio"] = self._compute_lexicon_ratio(
            text_lower, "mental_verbs", word_count
        )

        # Dimension 2 相关特征
        features["modal_possibility_ratio"] = self._compute_lexicon_ratio(
            text_lower, "modal_possibility", word_count
        )
        features["modal_prediction_ratio"] = self._compute_lexicon_ratio(
            text_lower, "modal_prediction", word_count
        )
        features["relative_clause_ratio"] = self._compute_relative_clause(
            doc["dependencies"], word_count
        )

        # Dimension 3 相关特征
        features["communication_verb_ratio"] = self._compute_lexicon_ratio(
            text_lower, "communication_verbs", word_count
        )
        features["suasive_verb_ratio"] = self._compute_lexicon_ratio(
            text_lower, "suasive_verbs", word_count
        )

        # Dimension 4 相关特征
        features["noun_modifier_ratio"] = self._compute_noun_modifier(
            doc["pos_tags"], word_count
        )
        features["abstract_noun_ratio"] = self._compute_lexicon_ratio(
            text_lower, "abstract_nouns", word_count
        )
        features["nominalization_ratio"] = self._compute_nominalization(
            doc["tokens"], word_count
        )

        # 辅助特征
        features["human_noun_ratio"] = self._compute_lexicon_ratio(
            text_lower, "human_nouns", word_count
        )
        features["word_length"] = self._compute_word_length(doc["tokens"])
        # 注意: word_count 作为额外信息添加到DataFrame中但不在FEATURE_NAMES中
        # 有些后续分析可能需要它

        return features

    # ==========================================================================
    #  单个特征计算方法
    #  以下每个方法计算一个独立的语言特征
    #  所有频率型特征均除以 word_count 以归一化
    # ==========================================================================

    def _compute_past_tense(self, doc, word_count: int) -> float:
        """
        F1: 过去时比率 (past_tense_ratio)

        统计所有带有 VBD (Past Tense Verb) 标签的词数，除以总词数。
        VBD 是 Penn Treebank 标签集中表示动词过去式的标签。

        例如:
          "The results indicated that corrosion occurred rapidly."
          → "indicated" (VBD), "occurred" (VBD) → count=2

        参数:
            doc: spaCy Doc对象
            word_count (int): 总词数

        返回:
            float: past_tense_ratio = VBD词数 / 总词数
        """
        vbd_count = sum(1 for token in doc if token.tag_ == "VBD")
        return vbd_count / max(word_count, 1)

    def _compute_present_tense(self, doc, word_count: int) -> float:
        """
        F2: 现在时比率 (present_tense_ratio)

        统计 VBP (Present Tense Verb, non-3rd person singular)
        和 VBZ (Present Tense Verb, 3rd person singular) 词数。
        除以总词数。

        例如:
          "This paper presents a novel approach. The results show improvements."
          → "presents" (VBZ), "show" (VBP) → count=2

        参数:
            doc: spaCy Doc对象
            word_count (int): 总词数

        返回:
            float: present_tense_ratio
        """
        present_count = sum(1 for token in doc if token.tag_ in ("VBP", "VBZ"))
        return present_count / max(word_count, 1)

    def _compute_passive(self, dependencies: List[Tuple], word_count: int) -> float:
        """
        F3: 被动语态比率 (passive_ratio)

        检测依存关系中的 auxpass (Auxiliary Passive) 标签。
        auxpass 表示助动词用于被动语态结构。

        例如:
          "The experiment was conducted at room temperature."
          → "was" → head="conducted", dep="auxpass"
          → 这表示 "was conducted" 是被动结构

        参数:
            dependencies (list): 依存关系列表 [(head, dep, token), ...]
            word_count (int): 总词数

        返回:
            float: passive_ratio
        """
        passive_count = sum(1 for _, dep, _ in dependencies if dep == "auxpass")
        return passive_count / max(word_count, 1)

    def _compute_stance_complement(
        self, doc, text_lower: str, word_count: int
    ) -> float:
        """
        F4: 立场动词 + that从句比率 (stance_verb_ratio)

        检测 "立场动词 + that" 的结构模式，表示作者的主观立场/态度。

        实现方式:
          1. 获取立场动词词形还原形式 (lemma)
          2. 在文本中查找 "<stance_verb> that" 模式
          3. 统计匹配次数

        例如:
          "We suggest that further research is needed."
          "The results indicate that the mechanism is complex."
          → suggest that, indicate that → count=2

        参数:
            doc: spaCy Doc对象
            text_lower (str): 小写化的完整文本
            word_count (int): 总词数

        返回:
            float: stance_verb_ratio
        """
        stance_verbs = self.dictionaries.get("stance_verbs", set())
        if not stance_verbs:
            return 0.0

        stance_count = 0
        # 方法: 遍历所有token，检测是否为立场动词且后跟 "that"
        for token in doc:
            if token.lemma_.lower() in stance_verbs:
                # 检查下一个非空格token是否为 "that"
                next_token = token.nbor(1) if token.i + 1 < len(doc) else None
                # 有时 "that" 可能被副词隔开，如 "suggest strongly that"
                # 简化处理：检查后2-3个位置内是否有 "that"
                for offset in range(1, 4):
                    if token.i + offset < len(doc):
                        candidate = doc[token.i + offset]
                        if candidate.text.lower() == "that":
                            stance_count += 1
                            break
                        if candidate.pos_ not in ("ADV", "PART"):
                            # 如果遇到非副词/非小品词，停止查找
                            break

        return stance_count / max(word_count, 1)

    def _compute_lexicon_ratio(
        self, text_lower: str, dict_key: str, word_count: int
    ) -> float:
        """
        通用词典匹配比率方法。

        对给定的词典在文本中进行子串匹配，统计匹配次数。
        用于: mental_verb_ratio, communication_verb_ratio, suasive_verb_ratio,
              modal_possibility_ratio, modal_prediction_ratio, abstract_noun_ratio,
              human_noun_ratio

        实现说明:
          使用简单的子串匹配 (term in text) 而非词边界匹配。
          这是有意为之的设计选择：
          - 科技文本中的术语可能有屈折变化（如 "indicates" vs "indicate"）
          - 子串匹配可以捕获这些变体
          - 对于多词短语（如 "point out"），子串匹配也更鲁棒

        参数:
            text_lower (str): 小写化的文本
            dict_key (str): 词典键名（self.dictionaries中的key）
            word_count (int): 总词数

        返回:
            float: 词典匹配比率
        """
        lexicon = self.dictionaries.get(dict_key, set())
        if not lexicon:
            return 0.0

        match_count = 0
        for term in lexicon:
            if term in text_lower:
                match_count += 1

        return match_count / max(word_count, 1)

    def _compute_relative_clause(
        self, dependencies: List[Tuple], word_count: int
    ) -> float:
        """
        F8: 关系从句比率 (relative_clause_ratio)

        检测依存关系中的 relcl (Relative Clause) 标签。
        relcl 表示一个从句是它所修饰名词的关系从句。

        例如:
          "The method that we proposed is effective."
          → "proposed" 的 dep="relcl", head="method"
          → 表示 "that we proposed" 是修饰 "method" 的关系从句

        参数:
            dependencies (list): 依存关系列表
            word_count (int): 总词数

        返回:
            float: relative_clause_ratio
        """
        relcl_count = sum(1 for _, dep, _ in dependencies if dep == "relcl")
        return relcl_count / max(word_count, 1)

    def _compute_noun_modifier(
        self, pos_tags: List[Tuple], word_count: int
    ) -> float:
        """
        F11: 前置名词修饰比率 (noun_modifier_ratio)

        检测连续两个名词 (NOUN + NOUN) 的模式，即名词作前置修饰语。
        这在科技英语中非常常见。

        例如:
          "corrosion resistance analysis" → 3个连续名词
          "metal surface treatment" → 3个连续名词
          "steel corrosion inhibitor" → 3个连续名词

        实现方式:
          遍历POS标签序列，检测 (NOUN, NOUN) 连续出现的位置。
          每遇到一个 NOUN→NOUN 的边界就计数一次。
          例如 NOUN NOUN NOUN 有2个边界，计数为2。

        参数:
            pos_tags (list): 词性标签列表 [(text, tag, pos), ...]
            word_count (int): 总词数

        返回:
            float: noun_modifier_ratio = NOUN+NOUN序列数 / 总词数
        """
        nn_count = 0
        prev_is_noun = False

        for _, _, pos in pos_tags:
            is_noun = (pos == "NOUN")
            if is_noun and prev_is_noun:
                # 检测到 NOUN + NOUN 边界
                nn_count += 1
            prev_is_noun = is_noun

        return nn_count / max(word_count, 1)

    def _compute_nominalization(
        self, tokens: List[str], word_count: int
    ) -> float:
        """
        F13: 名词化比率 (nominalization_ratio)

        统计以特定名词化后缀结尾的词数。
        名词化是将动词或形容词转换为名词的语言过程，
        在学术写作中非常普遍，表示抽象化和客观化。

        检测的后缀:
          -tion (investigation, determination, corrosion)
          -ment (development, measurement, treatment)
          -ance (performance, resistance, importance)
          -ence (dependence, difference, existence)
          -ity  (ability, reactivity, conductivity)
          -ness (hardness, effectiveness, thickness)
          -sion (discussion, conclusion, adhesion)
          -ship (relationship, workmanship)
          -age  (coverage, leakage, percentage)
          -ism  (mechanism, organism)
          -ure  (procedure, failure, exposure)
          -al   (removal, appraisal, arrival)
          -sis  (analysis, synthesis, hypothesis)

        注意:
          为了排除过短词的误匹配，仅统计长度 ≥ 5 的词。
          例如 "ural" 以 -al 结尾但不是名词化。

        参数:
            tokens (list[str]): 词列表
            word_count (int): 总词数

        返回:
            float: nominalization_ratio = 名词化词数 / 总词数
        """
        nominalization_count = 0
        for token in tokens:
            token_lower = token.lower()
            # 排除过短词，减少误匹配
            if len(token_lower) < 5:
                continue
            for suffix in self.NOMINALIZATION_SUFFIXES:
                if token_lower.endswith(suffix):
                    nominalization_count += 1
                    break  # 一个词只计数一次

        return nominalization_count / max(word_count, 1)

    def _compute_word_length(self, tokens: List[str]) -> float:
        """
        F15: 平均词长 (word_length)

        计算所有词的平均字符数。这是文体分析中的基本指标。

        学术文本倾向于使用更长的词（因为术语和名词化），
        因此平均词长越高，文本可能越偏向学术/抽象风格。

        参数:
            tokens (list[str]): 词列表

        返回:
            float: 平均字符长度
        """
        if not tokens:
            return 0.0
        total_chars = sum(len(t) for t in tokens)
        return total_chars / len(tokens)

    # ==========================================================================
    #  辅助方法
    # ==========================================================================

    def get_feature_description(self) -> Dict[str, str]:
        """
        返回每个特征的描述和计算方法说明。

        用于报告生成，向用户解释每个特征的含义。

        返回:
            dict: {特征名: 描述}
        """
        descriptions = {
            "past_tense_ratio": (
                "过去时动词比率。统计VBD标签词数/总词数。"
                "高频值表示文本偏向叙事/报告风格，属于Dimension 1。"
            ),
            "present_tense_ratio": (
                "现在时动词比率。统计VBP+VBZ标签词数/总词数。"
                "高频值表示文本偏向即时/互动风格，属于Dimension 1。"
            ),
            "passive_ratio": (
                "被动语态比率。统计auxpass依存关系数/总词数。"
                "高频值表示文本使用大量被动结构，常见于学术写作。属于Dimension 1。"
            ),
            "stance_verb_ratio": (
                "立场动词+that从句比率。检测如'suggest that'、'indicate that'等结构。"
                "表示作者对所述内容表达了主观立场。属于Dimension 1。"
            ),
            "mental_verb_ratio": (
                "心理动词比率。词典匹配（如believe, assume, consider等）。"
                "表示认知/心理过程的提及。属于Dimension 1。"
            ),
            "modal_possibility_ratio": (
                "可能性情态比率。词典匹配（如may, might, could, possibly等）。"
                "表示作者对所述命题的不确定性态度。属于Dimension 2。"
            ),
            "modal_prediction_ratio": (
                "预测性情态比率。词典匹配（如will, likely, predict, expected等）。"
                "表示对未来事态或结果的预测。属于Dimension 2。"
            ),
            "relative_clause_ratio": (
                "关系从句比率。统计relcl依存关系数/总词数。"
                "关系从句用于提供补充信息，使文本更复杂。属于Dimension 2。"
            ),
            "communication_verb_ratio": (
                "交流动词比率。词典匹配（如report, state, argue, describe等）。"
                "高频值表示文本关注学术交流行为。属于Dimension 3。"
            ),
            "suasive_verb_ratio": (
                "劝说性动词比率。词典匹配（如suggest, recommend, propose等）。"
                "表示文本包含建议、推荐等劝说性言语行为。属于Dimension 3。"
            ),
            "noun_modifier_ratio": (
                "前置名词修饰比率。检测NOUN+NOUN序列。"
                "高频值是科技英语的典型特征（如'corrosion resistance'）。属于Dimension 4。"
            ),
            "abstract_noun_ratio": (
                "抽象名词比率。词典匹配（如method, process, system, analysis等）。"
                "表示文本的概念/抽象层面。属于Dimension 4。"
            ),
            "nominalization_ratio": (
                "名词化比率。统计以-tion/-ment/-ance/-ity等后缀结尾的词。"
                "高频值表示文本高度名词化（学术写作的标志）。属于Dimension 4。"
            ),
            "human_noun_ratio": (
                "人类指称名词比率。词典匹配（如researcher, author, we等）。"
                "表示文本中对人类主体的指称程度。辅助特征。"
            ),
            "word_length": (
                "平均词长。所有词的字符数均值。"
                "较高值通常表示更学术/正式的写作风格。辅助特征。"
            ),
        }
        return descriptions

    def get_stats(self) -> Dict[str, Any]:
        """返回最近一次特征提取的统计信息。"""
        return self.stats.copy()
