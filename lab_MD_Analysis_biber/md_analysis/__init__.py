"""
================================================================================
 MD分析工具包 - 包初始化文件
 Multi-Dimensional Analysis Feasibility Testing Toolkit
================================================================================

本包实现基于Biber风格的科技论文摘要MD分析可行性测试工具。

模块结构：
  preprocessor.py      — 文本预处理（spaCy NLP管道）
  feature_extractor.py — 语言特征提取（15个Biber风格特征）
  pca_analyzer.py      — PCA降维 + Promax旋转
  suitability_checker.py — 数据适配性检测（KMO/Bartlett/样本量）
  report_generator.py  — HTML报告生成
  pipeline.py          — 主流程编排

使用方式：
  from md_analysis.pipeline import MDPipeline
  pipeline = MDPipeline(input_file="abstracts.csv")
  pipeline.run()
  pipeline.generate_report()

作者：基于PRD规范开发
项目：腐蚀领域学术摘要文体特征与影响力关联分析
================================================================================
"""

__version__ = "0.1.0"
__author__ = "MD Analysis Toolkit"

# 导出主要类和函数，方便外部调用
from .pipeline import MDPipeline
from .preprocessor import TextPreprocessor
from .feature_extractor import FeatureExtractor
from .pca_analyzer import PCAAnalyzer
from .suitability_checker import SuitabilityChecker
from .report_generator import ReportGenerator
