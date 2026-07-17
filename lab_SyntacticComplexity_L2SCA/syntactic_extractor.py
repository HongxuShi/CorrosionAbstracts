#!/usr/bin/env python3
"""
================================================================================
 句法复杂度特征提取模块 (基于Lu 2010 L2SCA框架)
 Syntactic Complexity Feature Extraction — L2SCA Implementation
================================================================================

14个句法复杂度指标，分5个维度：

1. Length of Production Unit (产出单位长度)
   MLS: Mean Length of Sentence (平均句长/词)
   MLT: Mean Length of T-unit (平均T-unit长度/词)
   MLC: Mean Length of Clause (平均从句长度/词)

2. Sentence Complexity (句子复杂度)
   C_S: Clauses per Sentence (每句从句数)

3. Subordination (从属度)
   C_T: Clauses per T-unit (每T-unit从句数)
   CT_T: Complex T-units per T-unit (复杂T-unit比例)
   DC_C: Dependent Clauses per Clause (从属从句比)
   DC_T: Dependent Clauses per T-unit (每T-unit从属从句数)

4. Coordination (并列度)
   CP_C: Coordinate Phrases per Clause (每从句并列短语数)
   CP_T: Coordinate Phrases per T-unit (每T-unit并列短语数)
   T_S: T-units per Sentence (每句T-unit数)

5. Particular Structures (特殊结构)
   CN_C: Complex Nominals per Clause (每从句复杂名词短语数)
   CN_T: Complex Nominals per T-unit (每T-unit复杂名词短语数)
   VP_T: Verb Phrases per T-unit (每T-unit动词短语数)

参考: Lu, X. (2010). Automatic analysis of syntactic complexity in second language writing.
     International Journal of Corpus Linguistics, 15(4), 474-496.

作者：理论验证阶段 — Syntactic Complexity方案
日期：2026/07/17
================================================================================
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, Token

logger = logging.getLogger(__name__)


class SyntacticComplexityExtractor:
    """
    句法复杂度特征提取器 — 实现Lu(2010)的14个L2SCA指标。

    需要spaCy的en_core_web_sm模型（已安装）。
    """

    FEATURE_NAMES = [
        "MLS", "MLT", "MLC",           # Length of Production Unit
        "C_S",                          # Sentence Complexity
        "C_T", "CT_T", "DC_C", "DC_T", # Subordination
        "CP_C", "CP_T", "T_S",         # Coordination
        "CN_C", "CN_T", "VP_T",        # Particular Structures
    ]

    # 并列连词
    COORD_CONJ = {'and', 'or', 'but', 'nor', 'yet', 'so', 'for'}

    def __init__(self, model_name: str = "en_core_web_sm"):
        logger.info(f"Loading spaCy model: {model_name}")
        self.nlp = spacy.load(model_name, disable=["ner", "entity_linker", "textcat"])
        self.stats: Dict = {}

    # =========================================================================
    # 主入口
    # =========================================================================

    def extract_all(self, abstracts: pd.DataFrame, text_col: str = "abstract") -> pd.DataFrame:
        """对所有摘要提取14个句法复杂度指标。"""
        n_total = len(abstracts)
        logger.info(f"Extracting syntactic complexity from {n_total} abstracts...")

        feature_rows = []
        for i, (_, row) in enumerate(abstracts.iterrows()):
            text = row.get(text_col, '')
            if not isinstance(text, str) or not text.strip():
                feature_rows.append({f: np.nan for f in self.FEATURE_NAMES})
                continue

            doc = self.nlp(text)
            features = self._extract_single(doc)
            features['doc_id'] = row.get('doi', f"doc_{i}")
            feature_rows.append(features)

            if (i + 1) % 1000 == 0:
                logger.info(f"  Progress: {i+1}/{n_total}")

        df = pd.DataFrame(feature_rows)
        cols = ['doc_id'] + self.FEATURE_NAMES
        df = df[[c for c in cols if c in df.columns]]

        self.stats = {'n_documents': len(df), 'n_features': len(self.FEATURE_NAMES)}
        logger.info(f"Done: {len(df)} x {len(self.FEATURE_NAMES)}")
        return df

    # =========================================================================
    # 单文本提取
    # =========================================================================

    def _extract_single(self, doc: Doc) -> Dict[str, float]:
        """从单个spaCy Doc提取14个指标。"""
        sentences = list(doc.sents)
        n_sentences = len(sentences)

        if n_sentences == 0:
            return {f: 0.0 for f in self.FEATURE_NAMES}

        # 计算基础单元
        n_words = len([t for t in doc if not t.is_space and not t.is_punct])
        t_units = self._find_t_units(doc, sentences)
        n_t_units = len(t_units)
        clauses = self._find_clauses(doc)
        n_clauses = len(clauses)
        dependent_clauses = self._find_dependent_clauses(doc, clauses)
        n_dc = len(dependent_clauses)
        complex_t_units = self._count_complex_t_units(t_units, clauses)
        coordinate_phrases = self._count_coordinate_phrases(doc)
        complex_nominals = self._count_complex_nominals(doc)
        verb_phrases = self._count_verb_phrases(doc)

        features = {}

        # 1. Length
        features['MLS'] = n_words / n_sentences if n_sentences > 0 else 0
        features['MLT'] = n_words / n_t_units if n_t_units > 0 else 0
        features['MLC'] = n_words / n_clauses if n_clauses > 0 else 0

        # 2. Sentence Complexity
        features['C_S'] = n_clauses / n_sentences if n_sentences > 0 else 0

        # 3. Subordination
        features['C_T'] = n_clauses / n_t_units if n_t_units > 0 else 0
        features['CT_T'] = complex_t_units / n_t_units if n_t_units > 0 else 0
        features['DC_C'] = n_dc / n_clauses if n_clauses > 0 else 0
        features['DC_T'] = n_dc / n_t_units if n_t_units > 0 else 0

        # 4. Coordination
        features['CP_C'] = coordinate_phrases / n_clauses if n_clauses > 0 else 0
        features['CP_T'] = coordinate_phrases / n_t_units if n_t_units > 0 else 0
        features['T_S'] = n_t_units / n_sentences if n_sentences > 0 else 0

        # 5. Particular Structures
        features['CN_C'] = complex_nominals / n_clauses if n_clauses > 0 else 0
        features['CN_T'] = complex_nominals / n_t_units if n_t_units > 0 else 0
        features['VP_T'] = verb_phrases / n_t_units if n_t_units > 0 else 0

        return features

    # =========================================================================
    # T-unit Detection
    # =========================================================================

    def _find_t_units(self, doc: Doc, sentences) -> List[List[Token]]:
        """
        识别T-units。

        简化方法: 以每个根动词(clausal root)为锚点，
        将其及其所有从属token归为一个T-unit。
        对于没有根动词的片段，视为独立T-unit。
        """
        t_units = []
        assigned = set()

        for sent in sentences:
            sent_tokens = list(sent)
            # Find root-level clausal anchors
            roots_in_sent = []
            for token in sent_tokens:
                if token.dep_ == "ROOT":
                    roots_in_sent.append(token)
                # Also treat conj-linked verbs as separate T-unit anchors
                elif (token.pos_ == "VERB" and token.dep_ == "conj"
                      and token.head.dep_ == "ROOT"):
                    roots_in_sent.append(token)

            if not roots_in_sent:
                # No clear root — treat whole sentence as one T-unit
                if sent_tokens:
                    t_units.append(sent_tokens)
                continue

            # For each root, collect its subtree as a T-unit
            for root in roots_in_sent:
                if root.i in assigned:
                    continue
                tu_tokens = list(root.subtree)
                for t in tu_tokens:
                    assigned.add(t.i)
                t_units.append(tu_tokens)

            # Remaining unassigned tokens form their own T-unit
            leftover = [t for t in sent_tokens if t.i not in assigned]
            if leftover:
                t_units.append(leftover)
                for t in leftover:
                    assigned.add(t.i)

        return t_units

    # =========================================================================
    # Clause Detection
    # =========================================================================

    def _find_clauses(self, doc: Doc) -> List[List[Token]]:
        """
        识别从句(Clauses)。

        每个有限动词(finite verb)及其主语/宾语/修饰语构成一个从句。
        """
        clauses = []
        assigned = set()

        # Find all finite verbs (excluding auxiliaries that are just helping)
        finite_verbs = []
        for token in doc:
            if token.pos_ == "VERB" and token.tag_ in {
                "VB", "VBD", "VBP", "VBZ",  # finite forms
                "VBG", "VBN",                # non-finite (counted but as dependent clauses)
            }:
                finite_verbs.append(token)

        for verb in finite_verbs:
            if verb.i in assigned:
                continue

            # Collect verb + its dependents as a clause
            clause_tokens = [verb]
            assigned.add(verb.i)

            for child in verb.children:
                if child.i not in assigned and child.dep_ not in ("cc", "conj"):
                    clause_tokens.append(child)
                    assigned.add(child.i)
                    # Also include grandchildren for complex structures
                    for grandchild in child.children:
                        if grandchild.i not in assigned and grandchild.dep_ not in ("cc", "conj", "punct"):
                            clause_tokens.append(grandchild)
                            assigned.add(grandchild.i)

            clauses.append(clause_tokens)

        return clauses

    # =========================================================================
    # Dependent Clauses
    # =========================================================================

    def _find_dependent_clauses(self, doc: Doc, clauses: List) -> List:
        """
        识别从属从句(Dependent Clauses)。

        包括: 关系从句(relcl)、状语从句(advcl)、补语从句(ccomp, xcomp)、
        非限定从句(infinitival/participial)。
        """
        dc_indices = set()
        dep_clause_labels = {"advcl", "relcl", "ccomp", "xcomp", "csubj", "csubjpass"}

        for token in doc:
            if token.dep_ in dep_clause_labels:
                # Mark the head of the dependent clause
                dc_indices.add(token.i)
                for child in token.subtree:
                    dc_indices.add(child.i)

        # Match to clause segments
        dependent = []
        for clause in clauses:
            if any(t.i in dc_indices for t in clause):
                dependent.append(clause)

        return dependent

    # =========================================================================
    # Complex T-units
    # =========================================================================

    def _count_complex_t_units(self, t_units: List, clauses: List) -> int:
        """复杂T-unit = 包含多于1个从句的T-unit。"""
        # Build mapping: token index -> clause index
        token_to_clause = {}
        for ci, clause in enumerate(clauses):
            for token in clause:
                if token.i not in token_to_clause:
                    token_to_clause[token.i] = set()
                token_to_clause[token.i].add(ci)

        count = 0
        for tu in t_units:
            clause_ids = set()
            for token in tu:
                if token.i in token_to_clause:
                    clause_ids.update(token_to_clause[token.i])
            if len(clause_ids) > 1:
                count += 1

        return count

    # =========================================================================
    # Coordinate Phrases
    # =========================================================================

    def _count_coordinate_phrases(self, doc: Doc) -> int:
        """
        统计并列短语数量。

        检测: 由并列连词(and/or/but)连接的相同POS标签序列。
        方法: 统计带有 conj 依存关系的token（表示与前一个并列项的关系）。
        """
        count = 0
        for token in doc:
            if token.dep_ == "conj" and token.pos_ in ("NOUN", "ADJ", "ADV", "VERB", "PROPN"):
                count += 1

        return count

    # =========================================================================
    # Complex Nominals
    # =========================================================================

    def _count_complex_nominals(self, doc: Doc) -> int:
        """
        统计复杂名词短语。

        复杂名词短语 = 名词后跟修饰语的情况:
          1. 名词 + 形容词修饰
          2. 名词 + 介词短语
          3. 名词 + 关系从句
          4. 名词 + 分词修饰
          5. 名词 + 所有格
          6. 名词 + 同位语
        """
        count = 0
        complex_modifiers = {"amod", "prep", "relcl", "acl", "poss", "appos", "nummod"}

        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                # Check if any child modifies this noun
                for child in token.children:
                    if child.dep_ in complex_modifiers:
                        count += 1
                        break  # Count each noun only once

        return count

    # =========================================================================
    # Verb Phrases
    # =========================================================================

    def _count_verb_phrases(self, doc: Doc) -> int:
        """
        统计动词短语数量。

        每个有限动词(finite verb)或动词群(verb cluster)计为一个VP。
        """
        count = 0
        for token in doc:
            # Count main verbs (finite + non-finite),
            # excluding auxiliaries that are dependents of other verbs
            if token.pos_ == "VERB":
                # Exclude verbs that are in auxiliary position
                if token.dep_ not in ("aux", "auxpass"):
                    count += 1
            elif token.pos_ == "AUX" and token.dep_ == "ROOT":
                # Count auxiliary as root (e.g., "is" as main verb)
                count += 1

        return count

    # =========================================================================
    # Utility
    # =========================================================================

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "MLS": "Mean Length of Sentence (words/sentence). Higher = longer sentences.",
            "MLT": "Mean Length of T-unit (words/T-unit). Higher = more information per main clause.",
            "MLC": "Mean Length of Clause (words/clause). Higher = more elaborated clauses.",
            "C_S": "Clauses per Sentence. Higher = more clausal embedding per sentence.",
            "C_T": "Clauses per T-unit. Higher = more subordination.",
            "CT_T": "Complex T-units per T-unit. Higher = more sentences with multiple clauses.",
            "DC_C": "Dependent Clauses per Clause. Higher = more subordination.",
            "DC_T": "Dependent Clauses per T-unit. Higher = more embedding depth.",
            "CP_C": "Coordinate Phrases per Clause. Higher = more parallel structures.",
            "CP_T": "Coordinate Phrases per T-unit. Higher = more coordination.",
            "T_S": "T-units per Sentence. Higher = more independent clauses per sentence.",
            "CN_C": "Complex Nominals per Clause. Higher = more information-packed noun phrases.",
            "CN_T": "Complex Nominals per T-unit. Higher = denser nominal modification.",
            "VP_T": "Verb Phrases per T-unit. Higher = more verbal structures per main clause.",
        }

    def get_stats(self) -> Dict:
        return self.stats.copy()
