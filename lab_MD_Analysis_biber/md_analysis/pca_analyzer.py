#!/usr/bin/env python3
"""
================================================================================
 PCA分析模块 — MD分析管道第三步
 PCA Analysis & Promax Rotation Module
================================================================================

功能说明：
  对提取的语言特征矩阵执行以下统计分析：
  1. 标准化（StandardScaler） — 使不同量纲的特征具有可比性
  2. 主成分分析（PCA） — 降维到4个维度，对应Biber MD分析的4个功能维度
  3. Promax旋转 — 使因子载荷更易解释（允许因素间存在相关性）
  4. 输出维度得分 — 每篇摘要在每个维度上的得分

技术路线（参考PRD和Biber方法论文献）：
  Feature Matrix → StandardScaler → PCA(4) → Promax Rotation → Dimension Scores

参考资料：
  Biber, D. (1988). Variation across Speech and Writing. Cambridge University Press.
  论文参考：四个因素累计解释约38%的方差（对于学术摘要语料是合理的期望值）

依赖：
  - sklearn.preprocessing.StandardScaler
  - sklearn.decomposition.PCA
  - factor_analyzer.FactorAnalyzer

作者：基于PRD规范开发
日期：2026/07/17
================================================================================
"""

import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer

logger = logging.getLogger(__name__)


# ============================================================================
#  PCAAnalyzer 类
# ============================================================================

class PCAAnalyzer:
    """
    PCA分析器

    对语言特征矩阵进行主成分分析和Promax旋转，输出解释方差、
    因子载荷矩阵和每篇文档的维度得分。

    属性:
        n_components (int): 保留的维度数，默认为4
        scaler (StandardScaler): 标准化器
        pca (PCA): PCA模型
        fa (FactorAnalyzer): 因子分析器（用于Promax旋转）
        loadings (pd.DataFrame): 因子载荷矩阵
        dimension_scores (pd.DataFrame): 维度得分矩阵
        explained_variance (dict): 各维度的解释方差
        cumulative_variance (float): 累计解释方差
        results (dict): 所有分析结果的汇总字典
    """

    # ==========================================================================
    #  初始化
    # ==========================================================================

    def __init__(self, n_components: int = 4):
        """
        初始化PCA分析器。

        参数:
            n_components (int): 降维目标维度数。Biber MD分析通常使用4-7个维度。
                                对于科技论文摘要的轻量化分析，默认使用4个维度。
        """
        self.n_components = n_components

        # ---- 模型对象（在fit时初始化） ----
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.fa: Optional[FactorAnalyzer] = None

        # ---- 结果存储 ----
        self.loadings: Optional[pd.DataFrame] = None          # 因子载荷矩阵
        self.dimension_scores: Optional[pd.DataFrame] = None  # 维度得分
        self.explained_variance: Dict[int, float] = {}        # 各维度解释方差
        self.cumulative_variance: float = 0.0                 # 累计解释方差
        self.results: Dict[str, Any] = {}                     # 完整结果汇总

        # 特征名称列表（在fit时记录）
        self.feature_names: List[str] = []

    # ==========================================================================
    #  主分析流程
    # ==========================================================================

    def analyze(
        self,
        feature_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        doc_id_col: str = "doc_id",
        apply_rotation: bool = True
    ) -> Dict[str, Any]:
        """
        执行完整的PCA分析流程。

        流程:
          1. 提取特征矩阵
          2. 标准化 (Z-score normalization)
          3. PCA降维 (n_components=4)
          4. Promax旋转 (可选，使载荷更易解释)
          5. 计算因子载荷矩阵
          6. 计算维度得分

        参数:
            feature_df (pd.DataFrame): 特征矩阵（FeatureExtractor的输出）
            feature_columns (list[str] | None): 需要分析的特征列名。
                                               为None时使用所有非doc_id的数值列。
            doc_id_col (str): 文档ID列名
            apply_rotation (bool): 是否进行Promax旋转。默认True。

        返回:
            dict: 包含以下键的汇总结果：
              - explained_variance: 各维度解释方差
              - cumulative_variance: 累计解释方差
              - loadings: 因子载荷DataFrame
              - dimension_scores: 维度得分DataFrame
              - n_samples: 样本量
              - n_features: 特征数
        """
        # ---- 第一步：准备特征矩阵 ----
        X, self.feature_names, doc_ids = self._prepare_matrix(
            feature_df, feature_columns, doc_id_col
        )

        n_samples, n_features = X.shape
        logger.info(f"特征矩阵形状: {n_samples} 篇 × {n_features} 个特征")

        # ---- 第二步：标准化 ----
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        logger.info("标准化完成 (Z-score normalization)")

        # ---- 第三步：PCA降维 ----
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        # 记录PCA解释方差
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            self.explained_variance[i + 1] = var
            logger.info(f"  Dimension {i+1}: {var*100:.2f}% 解释方差")

        self.cumulative_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"  累计解释方差: {self.cumulative_variance*100:.2f}%")

        # ---- 第四步：计算载荷矩阵 ----
        # PCA载荷: 特征与主成分之间的相关系数
        # loading = sqrt(eigenvalue) * eigenvector / std
        # 等价于: component matrix × sqrt(n_components)
        loadings_pca = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)

        # ---- 第五步：Promax旋转（可选） ----
        if apply_rotation and n_features >= self.n_components:
            try:
                self.fa = FactorAnalyzer(
                    n_factors=self.n_components,
                    rotation='promax',
                    method='principal'  # 使用主成分提取方法
                )
                self.fa.fit(X_scaled)

                # 获取旋转后的载荷矩阵
                loadings_rotated = self.fa.loadings_

                # 构建载荷DataFrame
                loadings_df = self._build_loadings_df(
                    loadings_rotated, self.feature_names, "Rotated"
                )
                logger.info("Promax旋转完成")

            except Exception as e:
                logger.warning(f"Promax旋转失败: {e}。将使用未旋转的PCA载荷。")
                loadings_df = self._build_loadings_df(
                    loadings_pca, self.feature_names, "Unrotated"
                )
        else:
            # 不使用旋转，直接使用PCA载荷
            if not apply_rotation:
                logger.info("已跳过Promax旋转（apply_rotation=False）")
            else:
                logger.warning(f"特征数({n_features})小于维度数({self.n_components})，"
                               f"跳过旋转")
            loadings_df = self._build_loadings_df(
                loadings_pca, self.feature_names, "Unrotated"
            )

        self.loadings = loadings_df

        # ---- 第六步：计算维度得分 ----
        # 维度得分 = 标准化后的特征矩阵 × 载荷矩阵
        # 使用旋转后的载荷（如果有的话）
        if self.fa is not None:
            scores = self.fa.transform(X_scaled)
        else:
            # 使用PCA得分（标准化）
            scores = X_pca / np.sqrt(self.pca.explained_variance_)

        scores_df = pd.DataFrame(
            scores,
            columns=[f"Dim{i+1}" for i in range(self.n_components)]
        )
        scores_df.insert(0, doc_id_col, doc_ids)
        self.dimension_scores = scores_df

        # ---- 汇总结果 ----
        self.results = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": self.n_components,
            "explained_variance": self.explained_variance.copy(),
            "cumulative_variance": self.cumulative_variance,
            "loadings": self.loadings,
            "dimension_scores": self.dimension_scores,
            "feature_names": self.feature_names.copy(),
            "rotation_applied": self.fa is not None,
            "pca_components": self.pca.components_.copy() if self.pca else None,
        }

        logger.info("=" * 60)
        logger.info(" PCA分析完成")
        logger.info(f"  样本量:    {n_samples}")
        logger.info(f"  特征数:    {n_features}")
        logger.info(f"  维度数:    {self.n_components}")
        logger.info(f"  累计方差:  {self.cumulative_variance*100:.2f}%")
        logger.info(f"  旋转:      {'Promax' if self.fa else '无'}")
        logger.info("=" * 60)

        return self.results

    # ==========================================================================
    #  辅助方法
    # ==========================================================================

    def _prepare_matrix(
        self,
        feature_df: pd.DataFrame,
        feature_columns: Optional[List[str]],
        doc_id_col: str
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        从DataFrame提取特征矩阵。

        执行以下处理:
          1. 选择特征列
          2. 检查并移除方差为0的列（常量特征）
          3. 处理缺失值（用列均值填充）
          4. 转为numpy数组

        参数:
            feature_df (pd.DataFrame): 特征DataFrame
            feature_columns (list[str] | None): 特征列名
            doc_id_col (str): ID列名

        返回:
            (X, feature_names, doc_ids): 特征矩阵、特征名列表、文档ID列表
        """
        # ---- 选择特征列 ----
        if feature_columns is None:
            # 自动排除ID列、非数值列
            exclude = {doc_id_col}
            feature_columns = [
                c for c in feature_df.columns
                if c not in exclude and np.issubdtype(feature_df[c].dtype, np.number)
            ]

        # ---- 提取数据 ----
        doc_ids = feature_df[doc_id_col].tolist() if doc_id_col in feature_df.columns else []
        df_features = feature_df[feature_columns].copy()

        # ---- 移除方差为0的列 ----
        variances = df_features.var()
        zero_var_cols = variances[variances == 0].index.tolist()
        if zero_var_cols:
            logger.warning(f"移除 {len(zero_var_cols)} 个方差为0的特征: {zero_var_cols}")
            df_features = df_features.drop(columns=zero_var_cols)
            feature_columns = [c for c in feature_columns if c not in zero_var_cols]

        # ---- 处理缺失值 ----
        # 用列均值填充（对于比率型特征，均值填充是合理的）
        missing_count = df_features.isna().sum().sum()
        if missing_count > 0:
            logger.warning(f"检测到 {missing_count} 个缺失值，将用列均值填充")
            df_features = df_features.fillna(df_features.mean())

        # ---- 处理无穷值 ----
        # 将 inf/-inf 替换为列均值
        inf_mask = np.isinf(df_features)
        if inf_mask.any().any():
            logger.warning(f"检测到 {(inf_mask).sum().sum()} 个无穷值，将用列均值替换")
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            df_features = df_features.fillna(df_features.mean())

        # ---- 构建numpy矩阵 ----
        X = df_features.values.astype(np.float64)
        final_feature_names = list(df_features.columns)

        return X, final_feature_names, doc_ids

    def _build_loadings_df(
        self,
        loadings_matrix: np.ndarray,
        feature_names: List[str],
        label: str = ""
    ) -> pd.DataFrame:
        """
        构建格式化的因子载荷DataFrame。

        输出格式类似论文中的载荷矩阵表：

        ┌─────────────────────┬────────┬────────┬────────┬────────┐
        │ Feature             │  Dim1  │  Dim2  │  Dim3  │  Dim4  │
        ├─────────────────────┼────────┼────────┼────────┼────────┤
        │ past_tense_ratio    │  0.72  │ -0.05  │ -0.32  │ -0.08  │
        │ passive_ratio       │  0.43  │ -0.15  │ -0.28  │ -0.33  │
        │ ...                 │  ...   │  ...   │  ...   │  ...   │
        └─────────────────────┴────────┴────────┴────────┴────────┘

        参数:
            loadings_matrix (ndarray): 载荷矩阵 (n_features × n_components)
            feature_names (list[str]): 特征名称列表
            label (str): 标签（用于区分旋转前后）

        返回:
            pd.DataFrame: 格式化的载荷表
        """
        n_cols = loadings_matrix.shape[1]
        columns = [f"Dim{i+1}" for i in range(n_cols)]

        df = pd.DataFrame(loadings_matrix, columns=columns)
        df.insert(0, "Feature", feature_names)

        # 四舍五入到4位小数（便于阅读）
        for col in columns:
            df[col] = df[col].round(4)

        return df

    # ==========================================================================
    #  结果查询方法
    # ==========================================================================

    def get_explained_variance_summary(self) -> pd.DataFrame:
        """
        返回解释方差的格式化汇总表。

        返回:
            pd.DataFrame: 包含各维度的解释方差和累计方差
        """
        rows = []
        cumsum = 0
        for i in range(1, self.n_components + 1):
            var = self.explained_variance.get(i, 0)
            cumsum += var
            rows.append({
                "Dimension": f"Dimension {i}",
                "Explained Variance (%)": f"{var*100:.2f}%",
                "Cumulative (%)": f"{cumsum*100:.2f}%",
            })

        return pd.DataFrame(rows)

    def get_top_features_per_dimension(
        self, threshold: float = 0.30, n_top: int = 5
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        获取每个维度上载荷最高的特征（绝对值最大的）。

        这个方法用于自动解读维度含义。对于每个维度，返回载荷绝对值
        大于threshold的特征列表，按载荷绝对值降序排列。

        参数:
            threshold (float): 载荷阈值（绝对值），低于此值的特征被忽略
            n_top (int): 每个维度最多返回几个特征

        返回:
            dict: {维度号: [{"feature": 名称, "loading": 载荷值}, ...]}
        """
        if self.loadings is None:
            return {}

        top_features = {}
        for i in range(1, self.n_components + 1):
            dim_col = f"Dim{i}"
            if dim_col not in self.loadings.columns:
                continue

            # 按载荷绝对值降序排列
            dim_loadings = self.loadings[["Feature", dim_col]].copy()
            dim_loadings["abs_loading"] = dim_loadings[dim_col].abs()
            dim_loadings = dim_loadings.sort_values("abs_loading", ascending=False)

            # 过滤并取top N
            selected = dim_loadings[dim_loadings["abs_loading"] >= threshold].head(n_top)
            top_features[i] = [
                {"feature": row["Feature"], "loading": row[dim_col]}
                for _, row in selected.iterrows()
            ]

        return top_features

    def interpret_dimension(self, dim_number: int) -> str:
        """
        自动生成维度含义的初步解读。

        基于每个维度上载荷最高的特征（正载荷和负载荷分别解读），
        生成该维度可能代表的功能/文体含义。

        参数:
            dim_number (int): 维度编号 (1-4)

        返回:
            str: 维度含义解读文本
        """
        if self.loadings is None:
            return "尚未进行分析。"

        dim_col = f"Dim{dim_number}"
        if dim_col not in self.loadings.columns:
            return f"维度 {dim_number} 不存在。"

        # 获取正载荷（特征对维度有正向贡献）
        positive = self.loadings[self.loadings[dim_col] > 0.20]
        positive = positive.sort_values(dim_col, ascending=False)

        # 获取负载荷（特征对维度有负向贡献）
        negative = self.loadings[self.loadings[dim_col] < -0.20]
        negative = negative.sort_values(dim_col, ascending=True)

        # ---- 构建解读 ----
        interpretation_parts = [f"Dimension {dim_number} 相关特征分析："]

        if not positive.empty:
            pos_features = positive["Feature"].head(5).tolist()
            interpretation_parts.append(
                f"  正载荷特征 ({len(positive)} 个超过阈值): "
                + ", ".join(pos_features)
            )

        if not negative.empty:
            neg_features = negative["Feature"].head(5).tolist()
            interpretation_parts.append(
                f"  负载荷特征 ({len(negative)} 个超过阈值): "
                + ", ".join(neg_features)
            )

        # ---- 根据Biber MD分析框架给出可能的维度假说 ----
        # Dimension 1: Involved vs. Informational (互动性 vs. 信息性)
        # Dimension 2: Narrative vs. Non-Narrative (叙事性 vs. 非叙事性)
        # Dimension 3: Context-Independent vs. Context-Dependent
        # Dimension 4: Abstract vs. Concrete (抽象 vs. 具体)

        dim_hypotheses = {
            1: ('可能解释: 互动/立场表达 vs. 信息传递维度。'
                '如果passive_ratio和nominalization为正载荷，past_tense为正载荷，'
                '则可能对应「程序性/报告性话语」。'),
            2: ('可能解释: 叙事vs.非叙事维度。'
                '如果past_tense和relative_clause为正载荷，'
                '则可能对应「叙事/时间导向话语」。'),
            3: ('可能解释: 立场相关维度。'
                '如果communication_verb和suasive_verb为正载荷，'
                '则可能对应「学术论证/劝说性话语」。'),
            4: ('可能解释: 抽象vs.具体维度。'
                '如果noun_modifier、nominalization和abstract_noun为正载荷，'
                '则可能对应「科技信息包装/抽象风格」。'),
        }

        interpretation_parts.append(
            dim_hypotheses.get(dim_number, "需要进一步分析以确定维度含义。")
        )

        return "\n".join(interpretation_parts)

    def get_all_interpretations(self) -> Dict[int, str]:
        """
        获取所有维度的含义解读。

        返回:
            dict: {维度号: 解读文本}
        """
        return {
            i: self.interpret_dimension(i)
            for i in range(1, self.n_components + 1)
        }

    def get_results(self) -> Dict[str, Any]:
        """返回完整的分析结果汇总。"""
        return self.results.copy()
