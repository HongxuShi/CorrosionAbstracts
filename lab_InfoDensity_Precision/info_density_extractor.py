#!/usr/bin/env python3
"""
信息密度与精确性特征提取
Information Density & Precision Feature Extraction

10个特征 — 全正则/简单NLP, 不需要外部模型:
  1. term_density         术语密度 (词边界匹配)
  2. numeric_density      定量信息密度 (含数字句子比例)
  3. abbreviation_ratio   缩写使用率
  4. redundancy_ratio     冗余套话占比
  5. sentence_length_cv   句长变异系数 (std/mean)
  6. short_sentence_ratio 短句比例 (<8词)
  7. long_sentence_ratio  长句比例 (>35词)
  8. question_ratio       问句比例 (修辞问句)
  9. formulaic_open_ratio 公式化开头比例
  10. avg_word_length     平均词长 (与Biber的word_length类似但独立计算)
"""
import re, logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class InfoDensityExtractor:
    FEATURE_NAMES = [
        "term_density", "numeric_density", "abbreviation_ratio",
        "redundancy_ratio", "sentence_length_cv", "short_sentence_ratio",
        "long_sentence_ratio", "question_ratio", "formulaic_open_ratio",
        "avg_word_length",
    ]

    # Boilerplate / formulaic phrases (redundancy detection)
    REDUNDANCY_PATTERNS = [
        r'this paper (aims|investigates|presents|reports|describes|discusses|focuses|deals|is concerned)',
        r'in this (paper|study|research|work|article)',
        r'the (present|current) (paper|study|research|work)',
        r'we (present|report|describe|discuss|investigate|propose|introduce)',
        r'the results (show|indicate|demonstrate|reveal|suggest|confirm)',
        r'it (is|was) (found|observed|shown|demonstrated|concluded|suggested) that',
        r'this (study|research|investigation|work) (aims|focuses|is aimed)',
        r'the (aim|purpose|objective|goal) of this',
        r'to the best of our knowledge',
        r'further (research|investigation|study|work) is (needed|required|necessary)',
        r'in (conclusion|summary|brief)',
    ]

    # Formulaic sentence openings
    FORMULAIC_OPENINGS = [
        r'^(in )?this (paper|study|research|work|article|investigation)',
        r'^we (present|propose|report|describe|investigate|introduce|demonstrate|show)',
        r'^the (present|current) (paper|study|research|work)',
        r'^(recently|currently|nowadays|today)',
        r'^it is (well |widely |generally )?(known|recognized|accepted|believed|assumed)',
        r'^over the (past|last) (few |several |)(years|decades)',
        r'^(corrosion|metal|material)',
    ]

    def __init__(self, jargon_path: Optional[str] = None):
        if jargon_path is None:
            jargon_path = Path(__file__).resolve().parent.parent / "lab_MD_Analysis_biber" / "md_dictionaries"
        # Load jargon terms from existing file
        self.jargon_terms: List[str] = []
        jp = Path(jargon_path) / ".." / ".." / "data" / "df_j.txt"
        # Try multiple paths
        for p in [
            Path(__file__).resolve().parent.parent / "data" / "df_j.txt",
            Path(__file__).resolve().parent.parent.parent / "data" / "df_j.txt",
        ]:
            if p.exists():
                self._load_jargon(p)
                break
        if not self.jargon_terms:
            logger.warning("Jargon file not found, term_density will be 0")
        logger.info(f"Loaded {len(self.jargon_terms)} jargon terms")

    def _load_jargon(self, path: Path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if '\xa0' in line:
                    eng = line.split('\xa0')[-1].strip()
                    if eng: self.jargon_terms.append(eng.lower())
                else:
                    self.jargon_terms.append(line.lower())

    def extract_all(self, abstracts: pd.DataFrame, text_col: str = "abstract") -> pd.DataFrame:
        n = len(abstracts)
        logger.info(f"Extracting from {n} abstracts...")
        rows = []
        for i, (_, row) in enumerate(abstracts.iterrows()):
            text = row.get(text_col, '')
            if not isinstance(text, str) or not text.strip():
                rows.append({f: np.nan for f in self.FEATURE_NAMES})
                continue
            f = self._extract_single(text)
            f['doc_id'] = row.get('doi', f"doc_{i}")
            rows.append(f)
            if (i+1)%1000==0: logger.info(f"  {i+1}/{n}")
        df = pd.DataFrame(rows)
        cols = ['doc_id']+self.FEATURE_NAMES
        return df[[c for c in cols if c in df.columns]]

    def _extract_single(self, text: str) -> Dict[str, float]:
        text_clean = re.sub(r'<[^>]+>', '', text).strip()
        words = text_clean.split()
        wc = len(words)
        if wc == 0:
            return {f: 0.0 for f in self.FEATURE_NAMES}

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text_clean)
        sentences = [s.strip() for s in sentences if s.strip()]
        sc = len(sentences)

        features = {}

        # 1. Term density (word-boundary matching)
        text_lower = ' ' + text_clean.lower() + ' '
        term_count = 0
        for term in self.jargon_terms:
            if ' ' in term:
                if term in text_lower: term_count += 1
            else:
                if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                    term_count += 1
        features['term_density'] = term_count / wc

        # 2. Numeric/quantitative density
        numeric_sentences = sum(1 for s in sentences if re.search(r'\d+', s))
        features['numeric_density'] = numeric_sentences / sc if sc > 0 else 0

        # 3. Abbreviation ratio (ALL CAPS or parenthetical definitions)
        abbrev_count = len(re.findall(r'\b[A-Z]{2,}\b', text_clean))
        # Also count defined abbreviations "X (Y)" patterns
        abbrev_count += len(re.findall(r'\([A-Z]{2,}\)', text_clean))
        features['abbreviation_ratio'] = abbrev_count / wc

        # 4. Redundancy ratio
        redundancy_count = 0
        for pat in self.REDUNDANCY_PATTERNS:
            redundancy_count += len(re.findall(pat, text_lower))
        features['redundancy_ratio'] = redundancy_count / wc

        # 5. Sentence length CV
        sent_lens = [len(s.split()) for s in sentences]
        mean_sl = np.mean(sent_lens) if sent_lens else 1
        features['sentence_length_cv'] = np.std(sent_lens) / mean_sl if mean_sl > 0 else 0

        # 6-7. Short/long sentence ratios
        features['short_sentence_ratio'] = sum(1 for l in sent_lens if l < 8) / sc if sc > 0 else 0
        features['long_sentence_ratio'] = sum(1 for l in sent_lens if l > 35) / sc if sc > 0 else 0

        # 8. Question ratio
        features['question_ratio'] = sum(1 for s in sentences if '?' in s) / sc if sc > 0 else 0

        # 9. Formulaic opening ratio
        formulaic_count = 0
        for s in sentences:
            for pat in self.FORMULAIC_OPENINGS:
                if re.search(pat, s.lower().strip()):
                    formulaic_count += 1
                    break
        features['formulaic_open_ratio'] = formulaic_count / sc if sc > 0 else 0

        # 10. Average word length
        chars = sum(len(w.strip('.,;:!?\'\"()[]{}')) for w in words)
        features['avg_word_length'] = chars / wc

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "term_density": "术语密度(w/词边界)。比值越高，文本包含越多的领域专业术语。",
            "numeric_density": "含数字的句子比例。高值表示摘要包含定量数据。",
            "abbreviation_ratio": "缩写使用密度。STEM常见，但过多影响可读性。",
            "redundancy_ratio": "套话占比('This paper investigates...'等)。高值=缺乏信息性。",
            "sentence_length_cv": "句长变异系数(std/mean)。高值表示句长变化大(风格多变)。",
            "short_sentence_ratio": "短句(<8词)比例。高值=碎片化，低值=复杂句为主。",
            "long_sentence_ratio": "长句(>35词)比例。高值=句子过于复杂，影响可读性。",
            "question_ratio": "修辞问句比例。某些学科常用，STEM少见。",
            "formulaic_open_ratio": "公式化句子开头比例。高值=缺乏变化。",
            "avg_word_length": "平均词长。学术写作通常较高(5-7字符)。",
        }
