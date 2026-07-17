#!/usr/bin/env python3
"""
================================================================================
 MD分析主流程编排模块
 Main Pipeline Orchestration Module
================================================================================

功能说明：
  串联MD分析的所有模块，提供一站式分析接口。

流程：
  CSV/Excel → [TextPreprocessor] → [FeatureExtractor] → [PCAAnalyzer]
                                                              ↓
                                          [SuitabilityChecker] ← [Feature Matrix]
                                                              ↓
                                          [ReportGenerator] → MD_analysis_report.html

使用方式：
  from md_analysis.pipeline import MDPipeline

  pipeline = MDPipeline()
  pipeline.run("abstracts.csv")
  # 报告自动保存到 results/md_analysis/MD_analysis_report.html

作者：基于PRD规范开发
日期：2026/07/17
================================================================================
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from .preprocessor import TextPreprocessor
from .feature_extractor import FeatureExtractor
from .pca_analyzer import PCAAnalyzer
from .suitability_checker import SuitabilityChecker
from .report_generator import ReportGenerator

# ============================================================================
#  日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
#  MDPipeline 类
# ============================================================================

class MDPipeline:
    """
    MD分析主流程编排器

    串联文本预处理 → 特征提取 → PCA分析 → 适配性检测 → 报告生成
    的完整分析管道。

    属性:
        preprocessor (TextPreprocessor): 文本预处理器
        extractor (FeatureExtractor): 特征提取器
        analyzer (PCAAnalyzer): PCA分析器
        checker (SuitabilityChecker): 适配性检测器
        reporter (ReportGenerator): 报告生成器
        results (dict): 完整分析结果
    """

    # ==========================================================================
    #  初始化
    # ==========================================================================

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        n_components: int = 4,
        min_word_count: int = 30,
        output_dir: Optional[str] = None,
        lexicon_dir: Optional[str] = None
    ):
        """
        初始化MD分析管道。

        参数:
            spacy_model (str): spaCy模型名称
            n_components (int): PCA维度数
            min_word_count (int): 最小词数阈值
            output_dir (str | None): 报告输出目录
            lexicon_dir (str | None): 词典目录
        """
        logger.info("=" * 60)
        logger.info(" 初始化 MD 分析管道")
        logger.info("=" * 60)

        # ---- 初始化各模块 ----
        self.preprocessor = TextPreprocessor(
            model_name=spacy_model,
            min_word_count=min_word_count
        )

        self.extractor = FeatureExtractor(lexicon_dir=lexicon_dir)

        self.analyzer = PCAAnalyzer(n_components=n_components)

        self.checker = SuitabilityChecker()

        self.reporter = ReportGenerator(output_dir=output_dir)

        # ---- 结果存储 ----
        self.results: Dict[str, Any] = {}
        self.feature_df: Optional[pd.DataFrame] = None
        self.processed_docs: list = []

        logger.info("所有模块初始化完成")

    # ==========================================================================
    #  主运行方法
    # ==========================================================================

    def run(
        self,
        input_file: str,
        text_col: str = "abstract",
        id_col: Optional[str] = None,
        apply_rotation: bool = True,
        save_features: bool = True
    ) -> Dict[str, Any]:
        """
        执行完整的MD分析流程。

        这是本模块的主入口方法。依次执行:
          1. 文本预处理
          2. 语言特征提取
          3. PCA分析 + Promax旋转
          4. 数据适配性检测
          5. HTML报告生成

        参数:
            input_file (str): 输入文件路径（CSV或Excel）
            text_col (str): 摘要文本列名，默认"abstract"
            id_col (str | None): ID列名，None则自动生成
            apply_rotation (bool): 是否进行Promax旋转，默认True
            save_features (bool): 是否保存特征矩阵CSV，默认True

        返回:
            dict: 完整分析结果，包含以下键：
              - preprocessing_stats: 预处理统计
              - feature_matrix: 特征矩阵DataFrame
              - pca_results: PCA分析结果
              - suitability_results: 适配性检测结果
              - report_path: 报告文件路径
        """
        input_path = Path(input_file)
        logger.info(f"\n{'=' * 60}")
        logger.info(f" 开始分析: {input_path.name}")
        logger.info(f"{'=' * 60}\n")

        # ======================================================================
        #  第一步：文本预处理
        # ======================================================================
        logger.info("[Step 1/5] 文本预处理...")
        self.processed_docs = self.preprocessor.process_file(
            str(input_path),
            text_col=text_col,
            id_col=id_col
        )
        preprocessing_stats = self.preprocessor.get_stats()
        word_count_dist = self.preprocessor.get_word_count_distribution(
            self.processed_docs
        )

        if not self.processed_docs:
            logger.error("预处理后无有效文档！请检查输入数据。")
            raise ValueError("预处理后无有效文档。请检查输入数据是否包含有效摘要。")

        # ======================================================================
        #  第二步：语言特征提取
        # ======================================================================
        logger.info("\n[Step 2/5] 提取语言特征...")
        self.feature_df = self.extractor.extract_all_features(self.processed_docs)
        extractor_stats = self.extractor.get_stats()

        # 可选：保存特征矩阵
        if save_features:
            feature_path = self.reporter.output_dir / "feature_matrix.csv"
            self.feature_df.to_csv(feature_path, index=False, encoding='utf-8')
            logger.info(f"  特征矩阵已保存: {feature_path}")

        # ======================================================================
        #  第三步：PCA分析
        # ======================================================================
        logger.info("\n[Step 3/5] PCA分析 + Promax旋转...")
        pca_results = self.analyzer.analyze(
            self.feature_df,
            apply_rotation=apply_rotation
        )

        # 可选：保存维度得分
        if save_features and self.analyzer.dimension_scores is not None:
            scores_path = self.reporter.output_dir / "dimension_scores.csv"
            self.analyzer.dimension_scores.to_csv(
                scores_path, index=False, encoding='utf-8'
            )
            logger.info(f"  维度得分已保存: {scores_path}")

        # ======================================================================
        #  第四步：数据适配性检测
        # ======================================================================
        logger.info("\n[Step 4/5] 数据适配性检测...")
        suitability_results = self.checker.check_all(self.feature_df)

        # ======================================================================
        #  第五步：生成HTML报告
        # ======================================================================
        logger.info("\n[Step 5/5] 生成分析报告...")
        self.reporter.collect_data(
            preprocessing_stats=preprocessing_stats,
            feature_df=self.feature_df,
            feature_stats=extractor_stats,
            pca_results=pca_results,
            suitability_results=suitability_results,
            word_count_distribution=word_count_dist,
            feature_descriptions=self.extractor.get_feature_description(),
            input_file=input_path.name
        )
        report_path = self.reporter.generate()

        # ======================================================================
        #  汇总结果
        # ======================================================================
        self.results = {
            "input_file": str(input_path),
            "preprocessing_stats": preprocessing_stats,
            "word_count_distribution": word_count_dist,
            "n_documents": len(self.feature_df),
            "n_features": extractor_stats.get("n_features", 0),
            "feature_matrix": self.feature_df,
            "pca_results": pca_results,
            "suitability_results": suitability_results,
            "report_path": report_path,
        }

        # ======================================================================
        #  控制台输出摘要
        # ======================================================================
        self._print_summary()

        return self.results

    # ==========================================================================
    #  摘要输出
    # ==========================================================================

    def _print_summary(self) -> None:
        """在控制台打印分析结果摘要。"""
        results = self.results
        pca = results.get("pca_results", {})
        suit = results.get("suitability_results", {})

        print("\n")
        print("=" * 60)
        print(f" [OK] MD分析完成!")
        print("=" * 60)
        print(f" 输入文件:   {results.get('input_file', 'N/A')}")
        print(f" 有效文档:   {results.get('n_documents', 0)} 篇")
        print(f" 语言特征:   {results.get('n_features', 0)} 个")
        print(f" PCA维度:    {pca.get('n_components', 4)}")
        print(f" 累计方差:   {pca.get('cumulative_variance', 0)*100:.1f}%")
        print(f" 因子旋转:   {'Promax' if pca.get('rotation_applied') else '无'}")
        verdict_short = suit.get('verdict', '未知')[:60]
        print(f" 适配性:     {verdict_short}...")
        print(f" 报告位置:   {results.get('report_path', 'N/A')}")
        print("=" * 60)

        # 打印维度信息
        explained = pca.get("explained_variance", {})
        for i in range(1, pca.get("n_components", 4) + 1):
            var = explained.get(i, 0) * 100
            print(f"   Dimension {i}: {var:.1f}% 解释方差")

        print("=" * 60)

    # ==========================================================================
    #  便捷方法
    # ==========================================================================

    def get_feature_matrix(self) -> pd.DataFrame:
        """返回特征矩阵。"""
        if self.feature_df is None:
            raise ValueError("尚未运行分析。请先调用 run() 方法。")
        return self.feature_df.copy()

    def get_dimension_scores(self) -> pd.DataFrame:
        """返回维度得分。"""
        if self.analyzer.dimension_scores is None:
            raise ValueError("尚未运行PCA分析。请先调用 run() 方法。")
        return self.analyzer.dimension_scores.copy()

    def get_loadings(self) -> pd.DataFrame:
        """返回因子载荷矩阵。"""
        if self.analyzer.loadings is None:
            raise ValueError("尚未运行PCA分析。请先调用 run() 方法。")
        return self.analyzer.loadings.copy()

    def get_results(self) -> Dict[str, Any]:
        """返回完整分析结果。"""
        if not self.results:
            raise ValueError("尚未运行分析。请先调用 run() 方法。")
        return self.results.copy()
