#!/usr/bin/env python3
"""
================================================================================
 文本预处理模块 — MD分析管道第一步
 Text Preprocessing Module for Multi-Dimensional Analysis
================================================================================

功能说明：
  对输入的科技论文摘要数据集执行以下NLP预处理步骤：
  1. 读取CSV/Excel格式的摘要数据
  2. 去除空文本（abstract为空或仅含空白字符）
  3. 使用spaCy进行英文NLP处理：
     - Tokenization（分词）
     - POS Tagging（词性标注）
     - Dependency Parsing（依存句法分析）
     - Lemmatization（词形还原）
  4. 输出结构化的预处理结果，供后续特征提取使用

依赖：
  - spacy (en_core_web_sm 模型)
  - pandas
  - numpy

使用方式：
  from md_analysis.preprocessor import TextPreprocessor

  preprocessor = TextPreprocessor()
  result = preprocessor.process_file("abstracts.csv", text_col="abstract")
  # result 是一个列表，每篇摘要对应一个字典：
  # {
  #   "id": ...,
  #   "abstract": ...,
  #   "tokens": [...],
  #   "pos_tags": [...],
  #   "dependencies": [...],
  #   "lemmas": [...],
  #   "word_count": int,
  #   "sentence_count": int
  # }

作者：基于PRD规范开发
日期：2026/07/17
================================================================================
"""

import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import pandas as pd
import numpy as np

# spaCy NLP处理管道
import spacy
from spacy.tokens import Doc

# ============================================================================
#  日志配置
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
#  TextPreprocessor 类
# ============================================================================

class TextPreprocessor:
    """
    文本预处理类

    封装了从原始CSV/Excel数据到结构化NLP分析结果的完整预处理流程。

    属性:
        nlp (spacy.Language): 加载的spaCy语言模型
        min_word_count (int): 最小词数阈值，低于此值的摘要将被过滤

    使用示例:
        >>> preprocessor = TextPreprocessor()
        >>> docs = preprocessor.process_file("data.csv")
        >>> print(f"成功处理 {len(docs)} 篇摘要")
    """

    # ==========================================================================
    #  初始化
    # ==========================================================================

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        min_word_count: int = 30,
        disable_components: Optional[List[str]] = None
    ):
        """
        初始化文本预处理器。

        参数:
            model_name (str): spaCy模型名称，默认为 "en_core_web_sm"
                              (轻量级英文模型，适合批量处理)
            min_word_count (int): 最小词数阈值。短于此值的摘要可能不包含
                                  足够的语言信息来分析文体特征，默认30词。
            disable_components (list[str] | None): 需要禁用的spaCy管道组件。
                                                  默认禁用 "ner"（命名实体识别），
                                                  因为MD分析不需要NER，禁用可加速处理。
        """
        # ---- 加载spaCy模型 ----
        logger.info(f"正在加载spaCy模型: {model_name} ...")
        try:
            if disable_components is None:
                # 默认禁用NER和Entity Linker — MD分析不需要这些，但可大幅加速
                # 只禁用不需要的组件。保留 attribute_ruler，
                # 因为它负责从 fine-grained tag 映射到 coarse-grained POS (如 NOUN, VERB)
                disable_components = ["ner", "entity_linker", "textcat", "sentencizer"]

            self.nlp = spacy.load(
                model_name,
                disable=disable_components
            )
            logger.info(f"spaCy模型加载成功。管道组件: {self.nlp.pipe_names}")
        except OSError:
            logger.error(
                f"找不到spaCy模型 '{model_name}'。"
                f"请运行: python -m spacy download {model_name}"
            )
            raise

        self.min_word_count = min_word_count
        self.model_name = model_name

        # ---- 统计信息（每次处理后更新） ----
        self.stats: Dict[str, Any] = {}

    # ==========================================================================
    #  数据加载
    # ==========================================================================

    def load_data(
        self,
        file_path: Union[str, Path],
        text_col: str = "abstract",
        id_col: Optional[str] = "id"
    ) -> pd.DataFrame:
        """
        从CSV或Excel文件中加载摘要数据。

        支持的文件格式:
          - .csv  (逗号分隔，UTF-8编码)
          - .xlsx / .xls (Excel格式)

        参数:
            file_path (str | Path): 输入文件路径
            text_col (str): 包含摘要文本的列名，默认为 "abstract"
            id_col (str | None): 用作文档ID的列名。为None时使用行号作为ID。

        返回:
            pd.DataFrame: 至少包含 id_col 和 text_col 两列的数据表

        异常:
            FileNotFoundError: 文件不存在
            ValueError: 未找到指定的文本列
        """
        file_path = Path(file_path)

        logger.info(f"正在加载数据文件: {file_path}")

        # ---- 根据扩展名选择读取方式 ----
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.suffix.lower() in ('.xlsx', '.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(
                f"不支持的文件格式: {file_path.suffix}。"
                f"请使用 .csv 或 .xlsx/.xls 文件。"
            )

        logger.info(f"成功读取 {len(df)} 条记录。列名: {list(df.columns)}")

        # ---- 验证文本列存在 ----
        if text_col not in df.columns:
            # 尝试模糊匹配（如 "Abstract" vs "abstract"）
            candidates = [c for c in df.columns if c.lower() == text_col.lower()]
            if candidates:
                text_col = candidates[0]
                logger.info(f"文本列名自动匹配为: '{text_col}'")
            else:
                raise ValueError(
                    f"未在数据中找到文本列 '{text_col}'。"
                    f"可用列: {list(df.columns)}"
                )

        # ---- 处理ID列 ----
        if id_col and id_col in df.columns:
            df['_doc_id'] = df[id_col].astype(str)
        else:
            # 使用行号作为ID
            df['_doc_id'] = [f"doc_{i}" for i in range(len(df))]

        # ---- 统一列名 ----
        df['_text'] = df[text_col]

        # ---- 保留可能存在的可选字段 ----
        self._optional_cols = {}
        for col in ['year', 'citation', 'journal', 'discipline']:
            if col in df.columns:
                self._optional_cols[col] = col

        return df[['_doc_id', '_text'] + list(self._optional_cols.values())]

    # ==========================================================================
    #  文本清洗与过滤
    # ==========================================================================

    def _clean_text(self, text: str) -> Optional[str]:
        """
        清洗单条摘要文本。

        清洗步骤:
          1. 去除首尾空白字符
          2. 将多个连续空白符合并为单个空格
          3. 移除可能的HTML标签残留
          4. 检查清洗后文本是否有效

        参数:
            text (str): 原始摘要文本

        返回:
            str | None: 清洗后的文本；如果文本为空或过短则返回None
        """
        # ---- 类型检查 ----
        if not isinstance(text, str):
            return None

        # ---- 去除首尾空白 ----
        text = text.strip()

        if not text:
            return None

        # ---- 合并连续空白 ----
        text = re.sub(r'\s+', ' ', text)

        # ---- 移除HTML标签残留（如 <i>, </i>, <br> 等） ----
        # 注意：科技论文摘要中常见斜体标签如 <i>Phoenix dactylifera</i>
        text = re.sub(r'<[^>]+>', '', text)

        # ---- 再次检查是否还有内容 ----
        text = text.strip()
        if not text:
            return None

        return text

    # ==========================================================================
    #  spaCy处理
    # ==========================================================================

    def _process_single_doc(self, doc_id: str, text: str) -> Optional[Dict[str, Any]]:
        """
        对单篇摘要执行完整的spaCy NLP处理。

        处理步骤:
          1. 运行spaCy管道（tokenization → POS tagging → dependency parsing → lemmatization）
          2. 提取tokens、词性标签、依存关系、词形还原结果
          3. 统计基本指标（词数、句数）

        参数:
            doc_id (str): 文档ID
            text (str): 清洗后的摘要文本

        返回:
            dict | None: 包含以下键的字典：
              - id: 文档ID
              - abstract: 原始文本（截断至200字符用于日志）
              - tokens: 词列表
              - pos_tags: 词性标签列表
              - dependencies: 依存关系列表，格式为 [(head_text, dep_label, token_text), ...]
              - lemmas: 词形还原结果列表
              - word_count: 词数
              - sentence_count: 句数
            若处理失败则返回None
        """
        try:
            # ---- 运行spaCy处理管道 ----
            # 注意：spaCy的__call__方法会依次执行管道中的所有组件
            doc: Doc = self.nlp(text)
        except Exception as e:
            logger.warning(f"spaCy处理失败 [ID={doc_id}]: {e}")
            return None

        # ---- 提取词列表 ----
        # 过滤掉纯标点和空白符，仅保留有实际文本内容的token
        tokens = [
            token.text for token in doc
            if not token.is_space and not (token.is_punct and len(token.text.strip()) == 0)
        ]

        # ---- 词数检查 ----
        word_count = len([t for t in doc if not t.is_space and not t.is_punct])
        if word_count < self.min_word_count:
            logger.debug(f"跳过过短摘要 [ID={doc_id}]: 仅{word_count}词（阈值={self.min_word_count}）")
            return None

        # ---- 提取词性标签 ----
        # 使用Penn Treebank标签集（spaCy的tag_属性）
        pos_tags = [
            (token.text, token.tag_, token.pos_)
            for token in doc
            if not token.is_space and not token.is_punct
        ]

        # ---- 提取依存关系 ----
        # 格式: (支配词文本, 依存关系标签, 当前词文本)
        # 例如: ("conducted", "auxpass", "was") 表示被动语态
        dependencies = [
            (token.head.text, token.dep_, token.text)
            for token in doc
            if not token.is_space and not token.is_punct
        ]

        # ---- 提取词形还原结果 ----
        lemmas = [
            (token.text, token.lemma_)
            for token in doc
            if not token.is_space and not token.is_punct
        ]

        # ---- 构建结果字典 ----
        result = {
            "id": doc_id,
            "abstract": text[:200] + "..." if len(text) > 200 else text,
            "abstract_full": text,  # 保留完整文本，供特征提取使用
            "tokens": tokens,
            "pos_tags": pos_tags,          # List[Tuple[str, str, str]]: (text, tag, pos)
            "dependencies": dependencies,  # List[Tuple[str, str, str]]: (head, dep, token)
            "lemmas": lemmas,              # List[Tuple[str, str]]: (text, lemma)
            "word_count": word_count,
            "sentence_count": len(list(doc.sents)),
            "spacy_doc": doc,              # 保留spaCy Doc对象供特征提取使用
        }

        return result

    # ==========================================================================
    #  批量处理
    # ==========================================================================

    def process_file(
        self,
        file_path: Union[str, Path],
        text_col: str = "abstract",
        id_col: Optional[str] = None,
        batch_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        加载文件并批量处理所有摘要。

        这是整个预处理模块的主入口方法，串联了数据加载、清洗和NLP处理。

        参数:
            file_path (str | Path): 输入文件路径
            text_col (str): 摘要文本列名
            id_col (str | None): ID列名
            batch_size (int): spaCy管道批处理大小。增大可提升速度但增加内存占用。

        返回:
            list[dict]: 处理后的文档列表，每个元素为一个字典
                        （详见 _process_single_doc 的返回值说明）

        处理流程:
          CSV/Excel → 数据清洗 → 空文本过滤 → spaCy管道 → 短文本过滤 → 输出
        """
        # ---- 第一步：加载数据 ----
        df = self._load_dataframe(file_path, text_col, id_col)

        total_input = len(df)
        logger.info(f"加载完成: {total_input} 条记录")

        # ---- 第二步：清洗文本 ----
        df['_cleaned'] = df['_text'].apply(self._clean_text)

        # 统计空文本
        empty_mask = df['_cleaned'].isna()
        n_empty = empty_mask.sum()
        logger.info(f"空文本/无效文本: {n_empty} 条 ({100*n_empty/max(total_input,1):.1f}%)")

        # 过滤空文本
        df_valid = df[~empty_mask].copy()
        logger.info(f"清洗后有效文本: {len(df_valid)} 条")

        # ---- 第三步：批量spaCy处理 ----
        docs = []
        nlp_failed = 0
        nlp_too_short = 0

        # 将文本分批送入spaCy管道
        texts = df_valid['_cleaned'].tolist()
        doc_ids = df_valid['_doc_id'].tolist()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = doc_ids[i:i + batch_size]

            # 使用spaCy的pipe方法进行批量处理（比逐条处理快）
            for doc_id, text, spacy_doc in zip(
                batch_ids, batch_texts,
                self.nlp.pipe(batch_texts, batch_size=batch_size)
            ):
                result = self._build_result_from_doc(doc_id, text, spacy_doc)
                if result is None:
                    nlp_failed += 1
                elif result['word_count'] < self.min_word_count:
                    nlp_too_short += 1
                else:
                    docs.append(result)

            # 进度日志
            if (i // batch_size) % 10 == 0:
                logger.info(
                    f"处理进度: {min(i + batch_size, len(texts))}/{len(texts)} "
                    f"({100 * min(i + batch_size, len(texts)) / len(texts):.1f}%)"
                )

        # ---- 第四步：汇总统计 ----
        self.stats = {
            "total_input": total_input,
            "empty_or_invalid": n_empty,
            "nlp_failed": nlp_failed,
            "too_short": nlp_too_short,
            "successful": len(docs),
            "success_rate": len(docs) / max(total_input, 1) * 100,
        }

        logger.info("=" * 60)
        logger.info(" 文本预处理完成")
        logger.info(f"  输入记录: {total_input}")
        logger.info(f"  空/无效:  {n_empty}")
        logger.info(f"  NLP失败: {nlp_failed}")
        logger.info(f"  过短:     {nlp_too_short}")
        logger.info(f"  有效输出: {len(docs)} ({self.stats['success_rate']:.1f}%)")
        logger.info("=" * 60)

        return docs

    def _load_dataframe(
        self,
        file_path: Union[str, Path],
        text_col: str,
        id_col: Optional[str]
    ) -> pd.DataFrame:
        """加载数据（process_file的辅助方法）"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.suffix.lower() in ('.xlsx', '.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        if text_col not in df.columns:
            candidates = [c for c in df.columns if c.lower() == text_col.lower()]
            if candidates:
                text_col = candidates[0]
            else:
                raise ValueError(f"未找到文本列 '{text_col}'。可用列: {list(df.columns)}")

        if id_col and id_col in df.columns:
            df['_doc_id'] = df[id_col].astype(str)
        else:
            df['_doc_id'] = [f"doc_{i}" for i in range(len(df))]

        df['_text'] = df[text_col]
        return df

    def _build_result_from_doc(
        self,
        doc_id: str,
        text: str,
        spacy_doc: Doc
    ) -> Optional[Dict[str, Any]]:
        """从spaCy Doc对象构建结构化结果（process_file的辅助方法）"""
        try:
            pos_tags = [
                (token.text, token.tag_, token.pos_)
                for token in spacy_doc
                if not token.is_space and not token.is_punct
            ]

            dependencies = [
                (token.head.text, token.dep_, token.text)
                for token in spacy_doc
                if not token.is_space and not token.is_punct
            ]

            lemmas = [
                (token.text, token.lemma_)
                for token in spacy_doc
                if not token.is_space and not token.is_punct
            ]

            tokens = [t[0] for t in pos_tags]
            word_count = len(tokens)

            if word_count == 0:
                return None

            return {
                "id": doc_id,
                "abstract": text[:200] + "..." if len(text) > 200 else text,
                "abstract_full": text,
                "tokens": tokens,
                "pos_tags": pos_tags,
                "dependencies": dependencies,
                "lemmas": lemmas,
                "word_count": word_count,
                "sentence_count": len(list(spacy_doc.sents)),
                "spacy_doc": spacy_doc,
            }
        except Exception as e:
            logger.warning(f"构建结果失败 [ID={doc_id}]: {e}")
            return None

    # ==========================================================================
    #  辅助方法
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """返回最近一次处理的统计信息。"""
        return self.stats.copy()

    def get_word_count_distribution(self, docs: List[Dict]) -> Dict[str, float]:
        """
        计算文档词数分布统计。

        参数:
            docs (list[dict]): process_file 返回的文档列表

        返回:
            dict: 包含 mean, median, std, min, max 的字典
        """
        word_counts = [d['word_count'] for d in docs]
        return {
            "mean": float(np.mean(word_counts)),
            "median": float(np.median(word_counts)),
            "std": float(np.std(word_counts)),
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "total": int(np.sum(word_counts)),
        }
