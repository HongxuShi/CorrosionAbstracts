#!/usr/bin/env python3
"""
================================================================================
 数据适配性检测模块 — MD分析管道第四步
 Data Suitability Assessment Module for MD Analysis
================================================================================

功能说明：
  评估输入数据集是否适合进行多维分析（MD Analysis）。
  调用此模块前需要先完成特征提取和PCA分析。

检测项目：
  ┌─────────────────────┬──────────────────────────────────────┐
  │ 检测项              │ 判断标准                             │
  ├─────────────────────┼──────────────────────────────────────┤
  │ 样本量              │ < 100: 不建议                        │
  │ (Sample Size)       │ 100-300: 谨慎                        │
  │                     │ 500+: 较适合                         │
  │                     │ 1000+: 较稳定                        │
  ├─────────────────────┼──────────────────────────────────────┤
  │ 特征方差            │ 方差 ≈ 0 的特征需要被移除            │
  │ (Feature Variance)  │ （说明该特征在所有文本中几乎相同）   │
  ├─────────────────────┼──────────────────────────────────────┤
  │ KMO检验             │ ≥ 0.9: 非常好                        │
  │ (Kaiser-Meyer-Olkin)│ 0.8-0.9: 良好                        │
  │                     │ 0.7-0.8: 一般                         │
  │                     │ 0.6-0.7: 可接受                       │
  │                     │ < 0.6: 不适合因子分析                 │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Bartlett球形检验    │ p < 0.05: 适合因子分析               │
  │ (Bartlett's Test)   │ p ≥ 0.05: 数据可能不适合             │
  └─────────────────────┴──────────────────────────────────────┘

依赖：
  - numpy, scipy
  - factor_analyzer (用于KMO和Bartlett检验)
  - pandas

作者：基于PRD规范开发
日期：2026/07/17
================================================================================
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

logger = logging.getLogger(__name__)


# ============================================================================
#  SuitabilityChecker 类
# ============================================================================

class SuitabilityChecker:
    """
    数据适配性检测器

    对特征矩阵执行一系列统计检验，判断数据是否适合进行因子分析/MD分析。

    属性:
        results (dict): 检测结果汇总
        verdict (str): 综合判断结论
        warnings (list[str]): 警告信息列表
    """

    # ==========================================================================
    #  检测阈值常量
    # ==========================================================================

    # 样本量评估阈值
    SAMPLE_THRESHOLDS = {
        "not_recommended": 100,    # < 100: 不建议
        "cautious": 300,            # 100-300: 谨慎
        "adequate": 500,            # 300-500: 可接受
        "good": 1000,               # 500-1000: 较适合
        # >= 1000: 较稳定
    }

    # KMO评估阈值
    KMO_THRESHOLDS = {
        "marvelous": 0.9,
        "meritorious": 0.8,
        "middling": 0.7,
        "mediocre": 0.6,
        # < 0.6: unacceptable
    }

    # Bartlett检验显著性水平
    BARTLETT_ALPHA = 0.05

    # 特征低方差阈值（方差低于此值的特征可能需要移除）
    LOW_VARIANCE_THRESHOLD = 1e-6

    # ==========================================================================
    #  初始化
    # ==========================================================================

    def __init__(self):
        """初始化检测器。"""
        self.results: Dict[str, Any] = {}
        self.verdict: str = ""
        self.warnings: List[str] = []
        self.recommendations: List[str] = []

    # ==========================================================================
    #  主检测流程
    # ==========================================================================

    def check_all(
        self,
        feature_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        doc_id_col: str = "doc_id"
    ) -> Dict[str, Any]:
        """
        执行所有数据适配性检测。

        这是本模块的主入口方法。依次执行:
          1. 样本量检查
          2. 特征方差检查
          3. KMO检验
          4. Bartlett球形检验

        参数:
            feature_df (pd.DataFrame): 特征矩阵
            feature_columns (list[str] | None): 特征列名列表
            doc_id_col (str): ID列名

        返回:
            dict: 包含所有检测结果的字典
        """
        logger.info("=" * 60)
        logger.info(" 开始数据适配性检测")
        logger.info("=" * 60)

        # ---- 准备数据 ----
        X, feature_names = self._prepare_matrix(feature_df, feature_columns, doc_id_col)
        n_samples, n_features = X.shape
        logger.info(f"  样本量: {n_samples}")
        logger.info(f"  特征数: {n_features}")

        # ---- 1. 样本量检查 ----
        sample_check = self._check_sample_size(n_samples)
        self.results["sample_size"] = sample_check
        logger.info(f"  样本量评估: {sample_check['level']}")

        # ---- 2. 特征方差检查 ----
        variance_check = self._check_feature_variance(X, feature_names)
        self.results["feature_variance"] = variance_check
        n_low_var = len(variance_check["low_variance_features"])
        logger.info(f"  低方差特征: {n_low_var} 个")

        # ---- 2b. 过滤低方差特征（用于后续KMO和Bartlett检验） ----
        # KMO和Bartlett检验要求协方差矩阵可逆，因此需要先移除常量特征
        low_var_features = variance_check["low_variance_features"]
        if low_var_features:
            keep_mask = ~np.isin(feature_names, low_var_features)
            X_filtered = X[:, keep_mask]
            filtered_names = [f for f in feature_names if f not in low_var_features]
            logger.info(f"  过滤后用于检验的特征: {len(filtered_names)} 个 "
                        f"(移除了 {len(low_var_features)} 个低方差特征)")
        else:
            X_filtered = X
            filtered_names = feature_names

        # ---- 3. KMO检验（使用过滤后的矩阵） ----
        kmo_result = self._perform_kmo_test(X_filtered, filtered_names)
        self.results["kmo"] = kmo_result
        if kmo_result.get('kmo_score') is not None:
            logger.info(f"  KMO: {kmo_result['kmo_score']:.4f} ({kmo_result['level']})")
        else:
            logger.info(f"  KMO: 计算失败")

        # ---- 4. Bartlett球形检验（使用过滤后的矩阵） ----
        bartlett_result = self._perform_bartlett_test(X_filtered)
        self.results["bartlett"] = bartlett_result
        if bartlett_result.get('chi2') is not None:
            logger.info(f"  Bartlett: chi2={bartlett_result['chi2']:.2f}, "
                        f"p={bartlett_result['p_value']:.6f} "
                        f"({'通过' if bartlett_result['passed'] else '未通过'})")
        else:
            logger.info(f"  Bartlett: 计算失败")

        # ---- 生成综合判断 ----
        self._generate_verdict()

        logger.info("=" * 60)
        logger.info(f" 综合判断: {self.verdict}")
        logger.info("=" * 60)

        return self.results

    # ==========================================================================
    #  单项检测方法
    # ==========================================================================

    def _check_sample_size(self, n_samples: int) -> Dict[str, Any]:
        """
        检查样本量是否适合MD分析。

        MD分析（尤其是因子分析）对样本量有一定要求。
        样本量越大，相关矩阵越稳定，因子结构的可泛化性越好。

        Biber (1988) 的研究使用了约1000篇文本。对于轻量化MD分析原型:
          - < 100: 样本量过少，因子结构不稳定，不建议继续
          - 100-300: 可尝试但结果需谨慎解释
          - 300-500: 基本可接受
          - 500-1000: 较适合
          - ≥ 1000: 较稳定，接近Biber研究的规模

        参数:
            n_samples (int): 文档/摘要数量

        返回:
            dict: 包含 level, description, is_adequate 等字段
        """
        if n_samples < self.SAMPLE_THRESHOLDS["not_recommended"]:
            level = "不建议"
            description = (
                f"样本量 ({n_samples} 篇) 不足 {self.SAMPLE_THRESHOLDS['not_recommended']} 篇。"
                f"此规模下因子结构可能不稳定，建议收集更多数据后再进行分析。"
            )
            is_adequate = False
        elif n_samples < self.SAMPLE_THRESHOLDS["cautious"]:
            level = "谨慎使用"
            description = (
                f"样本量 ({n_samples} 篇) 在 {self.SAMPLE_THRESHOLDS['not_recommended']}"
                f"-{self.SAMPLE_THRESHOLDS['cautious']} 之间。可以进行分析，但结果需要谨慎解读。"
                f"建议收集更多数据以提升稳定性。"
            )
            is_adequate = True  # 可以尝试，但要警告
        elif n_samples < self.SAMPLE_THRESHOLDS["adequate"]:
            level = "基本可接受"
            description = (
                f"样本量 ({n_samples} 篇) 在 {self.SAMPLE_THRESHOLDS['cautious']}"
                f"-{self.SAMPLE_THRESHOLDS['adequate']} 之间。可以进行初步分析。"
            )
            is_adequate = True
        elif n_samples < self.SAMPLE_THRESHOLDS["good"]:
            level = "较适合"
            description = (
                f"样本量 ({n_samples} 篇) 在 {self.SAMPLE_THRESHOLDS['adequate']}"
                f"-{self.SAMPLE_THRESHOLDS['good']} 之间。适合进行MD分析。"
            )
            is_adequate = True
        else:
            level = "较稳定"
            description = (
                f"样本量 ({n_samples} 篇) 超过 {self.SAMPLE_THRESHOLDS['good']} 篇。"
                f"规模与Biber (1988) 研究相近，因子结构预期较稳定。"
            )
            is_adequate = True

        return {
            "n_samples": n_samples,
            "level": level,
            "description": description,
            "is_adequate": is_adequate,
        }

    def _check_feature_variance(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        检查各特征的方差，标记方差接近0的特征。

        如果某个语言特征在所有文档中的取值几乎相同（方差 ≈ 0），
        说明该特征没有区分度，不应纳入因子分析。

        例如:
          如果所有摘要中都几乎没有过去时动词（过去时比率全为0），
          那么 past_tense_ratio 就没有分析价值。

        参数:
            X (ndarray): 特征矩阵 (n_samples × n_features)
            feature_names (list[str]): 特征名称列表

        返回:
            dict: 包含各特征的方差和低方差标记
        """
        # ---- 计算每个特征的方差 ----
        variances = np.var(X, axis=0, ddof=1)  # 样本方差 (ddof=1)
        stds = np.std(X, axis=0, ddof=1)

        # ---- 构建方差详情 ----
        variance_details = []
        low_variance_features = []

        for i, name in enumerate(feature_names):
            var = variances[i]
            is_low = var < self.LOW_VARIANCE_THRESHOLD

            variance_details.append({
                "feature": name,
                "variance": float(var),
                "std": float(stds[i]),
                "mean": float(np.mean(X[:, i])),
                "is_low_variance": is_low,
            })

            if is_low:
                low_variance_features.append(name)

        return {
            "details": variance_details,
            "low_variance_features": low_variance_features,
            "n_low_variance": len(low_variance_features),
            "n_total": len(feature_names),
        }

    def _perform_kmo_test(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        执行KMO (Kaiser-Meyer-Olkin) 检验。

        KMO检验测量变量间的偏相关性，判断数据是否适合因子分析。

        KMO值的含义:
          ≥ 0.9: 非常好 (marvelous)
          0.8-0.9: 良好 (meritorious)
          0.7-0.8: 一般 (middling)
          0.6-0.7: 可接受 (mediocre)
          < 0.6: 不适合 (unacceptable)

        原理:
          KMO比较了观测到的相关系数与偏相关系数的大小。
          如果变量之间存在潜在的共同因素（适合因子分析），
          则偏相关系数应该较小，KMO值接近1。

        参数:
            X (ndarray): 特征矩阵
            feature_names (list[str]): 特征名称列表

        返回:
            dict: KMO检验结果
        """
        try:
            # 使用factor_analyzer的KMO计算函数
            kmo_all, kmo_total = calculate_kmo(X)

            # ---- 评估总体KMO ----
            if kmo_total >= self.KMO_THRESHOLDS["marvelous"]:
                level = "非常好"
            elif kmo_total >= self.KMO_THRESHOLDS["meritorious"]:
                level = "良好"
            elif kmo_total >= self.KMO_THRESHOLDS["middling"]:
                level = "一般"
            elif kmo_total >= self.KMO_THRESHOLDS["mediocre"]:
                level = "可接受"
            else:
                level = "不适合"

            is_acceptable = kmo_total >= self.KMO_THRESHOLDS["mediocre"]

            # ---- 各特征的KMO值（MSA: Measure of Sampling Adequacy） ----
            per_feature_kmo = {}
            if isinstance(kmo_all, np.ndarray) and len(kmo_all) == len(feature_names):
                for i, name in enumerate(feature_names):
                    per_feature_kmo[name] = float(kmo_all[i])

            return {
                "kmo_score": float(kmo_total),
                "level": level,
                "is_acceptable": is_acceptable,
                "per_feature_kmo": per_feature_kmo,
                "description": (
                    f"总体KMO值为 {kmo_total:.4f}，"
                    f'表明采样充分性处于「{level}」水平。'
                    f"{'' if is_acceptable else ' 数据不太适合因子分析，建议收集更多样本来改进。'}"
                ),
            }

        except Exception as e:
            logger.warning(f"KMO检验失败: {e}")
            return {
                "kmo_score": None,
                "level": "计算失败",
                "is_acceptable": None,
                "per_feature_kmo": {},
                "description": f"KMO检验无法完成: {e}",
                "error": str(e),
            }

    def _perform_bartlett_test(self, X: np.ndarray) -> Dict[str, Any]:
        """
        执行Bartlett球形检验。

        Bartlett球形检验的原假设是：相关矩阵是单位矩阵（即所有变量之间
        都不相关）。如果p值小于显著性水平（0.05），则拒绝原假设，说明
        变量之间存在相关性，适合进行因子分析。

        原理:
          Bartlett检验比较相关矩阵的行列式与单位矩阵的行列式。
          如果变量之间完全独立，相关矩阵就是单位矩阵，行列式为1。
          行列式越小，变量间的相关性越强。

        参数:
            X (ndarray): 特征矩阵

        返回:
            dict: Bartlett检验结果
        """
        try:
            chi2, p_value = calculate_bartlett_sphericity(X)

            passed = p_value < self.BARTLETT_ALPHA

            if passed:
                description = (
                    f"Bartlett球形检验显著 (χ²={chi2:.2f}, p={p_value:.6f})，"
                    f'拒绝「变量之间互不相关」的原假设。'
                    f"数据适合进行因子分析。"
                )
            else:
                description = (
                    f"Bartlett球形检验不显著 (χ²={chi2:.2f}, p={p_value:.6f})，"
                    f'无法拒绝「变量之间互不相关」的原假设。'
                    f"数据可能不太适合因子分析，建议检查特征相关性或增加样本量。"
                )

            return {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "passed": passed,
                "alpha": self.BARTLETT_ALPHA,
                "description": description,
            }

        except Exception as e:
            logger.warning(f"Bartlett检验失败: {e}")
            return {
                "chi2": None,
                "p_value": None,
                "passed": None,
                "alpha": self.BARTLETT_ALPHA,
                "description": f"Bartlett检验无法完成: {e}",
                "error": str(e),
            }

    # ==========================================================================
    #  综合判断
    # ==========================================================================

    def _generate_verdict(self) -> None:
        """
        基于所有检测结果生成综合判断。

        综合考量:
          1. 样本量是否足够
          2. KMO是否可接受
          3. Bartlett是否通过
          4. 低方差特征的数量

        判断结果写入 self.verdict, self.warnings, self.recommendations
        """
        self.warnings = []
        self.recommendations = []

        # ---- 收集各项结果 ----
        sample = self.results.get("sample_size", {})
        kmo = self.results.get("kmo", {})
        bartlett = self.results.get("bartlett", {})
        variance = self.results.get("feature_variance", {})

        # ---- 逐项判断 ----
        checks_passed = 0
        checks_total = 3  # 样本量、KMO、Bartlett

        # 1. 样本量
        if sample.get("is_adequate", False):
            checks_passed += 1
        else:
            self.warnings.append(f"⚠ 样本量不足: {sample.get('description', '')}")

        # 2. KMO
        if kmo.get("is_acceptable", False):
            checks_passed += 1
        elif kmo.get("is_acceptable") is None:
            checks_total -= 1  # KMO无法计算，降低标准
            self.warnings.append("KMO检验无法执行，跳过此项评估。")
        else:
            self.warnings.append(f"⚠ KMO值偏低: {kmo.get('description', '')}")

        # 3. Bartlett
        if bartlett.get("passed", False):
            checks_passed += 1
        elif bartlett.get("passed") is None:
            checks_total -= 1
            self.warnings.append("Bartlett检验无法执行，跳过此项评估。")
        else:
            self.warnings.append(f"⚠ Bartlett检验未通过: {bartlett.get('description', '')}")

        # 4. 低方差特征
        n_low_var = variance.get("n_low_variance", 0)
        if n_low_var > 0:
            low_var_features = variance.get("low_variance_features", [])
            self.warnings.append(
                f"检测到 {n_low_var} 个低方差特征: {', '.join(low_var_features)}。"
                f"建议在分析前移除这些特征。"
            )
            self.recommendations.append(
                "移除低方差特征（它们在不同文本中几乎没有变化，无法提供有效信息）。"
            )

        # ---- 生成综合判断 ----
        adequacy_ratio = checks_passed / max(checks_total, 1)

        if adequacy_ratio >= 1.0:
            self.verdict = "✅ 适合: 数据集通过了所有必要检验，适合进行MD分析。"
        elif adequacy_ratio >= 0.67:
            self.verdict = (
                f"⚠️ 有条件适合: 数据集通过了 {checks_passed}/{checks_total} 项检验。"
                f"可以尝试MD分析，但结果需要谨慎解读。"
            )
        elif adequacy_ratio >= 0.33:
            self.verdict = (
                f"⚠️ 勉强可行: 数据集仅通过了 {checks_passed}/{checks_total} 项检验。"
                f"MD分析结果仅供参考，不建议基于此做重要推断。"
            )
        else:
            self.verdict = (
                f"❌ 不适合: 数据集未通过必要检验 ({checks_passed}/{checks_total})。"
                f"当前数据不适合进行MD分析。建议: (1) 收集更多数据; "
                f"(2) 检查特征定义是否合理; (3) 确保数据质量。"
            )

        # ---- 添加通用建议 ----
        if n_low_var > 0:
            self.recommendations.append(
                "考虑增加更多样化的摘要数据（不同期刊、不同年份），"
                "以增加特征之间的方差。"
            )

        if sample.get("n_samples", 0) < self.SAMPLE_THRESHOLDS["cautious"]:
            self.recommendations.append(
                "考虑使用更多期刊的数据，或放宽检索条件以获取更多摘要。"
                f"Biber (1988) 使用了约1000篇文本，建议至少有"
                f"{self.SAMPLE_THRESHOLDS['adequate']}篇以上。"
            )

    # ==========================================================================
    #  结果查询方法
    # ==========================================================================

    def get_results(self) -> Dict[str, Any]:
        """返回全部检测结果。"""
        return {
            **self.results,
            "verdict": self.verdict,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def get_summary_table(self) -> pd.DataFrame:
        """生成检测结果汇总表（用于报告）。"""
        rows = []

        sample = self.results.get("sample_size", {})
        rows.append({
            "检测项目": "样本量 (Sample Size)",
            "结果": f"{sample.get('n_samples', 'N/A')} 篇",
            "评估": sample.get("level", "N/A"),
            "状态": "✅" if sample.get("is_adequate") else "⚠️",
        })

        kmo = self.results.get("kmo", {})
        rows.append({
            "检测项目": "KMO检验",
            "结果": f"{kmo.get('kmo_score', 'N/A'):.4f}" if kmo.get('kmo_score') else "N/A",
            "评估": kmo.get("level", "N/A"),
            "状态": "✅" if kmo.get("is_acceptable") else ("⚠️" if kmo.get("is_acceptable") is False else "❓"),
        })

        bartlett = self.results.get("bartlett", {})
        p_val = bartlett.get("p_value")
        rows.append({
            "检测项目": "Bartlett球形检验",
            "结果": f"p={p_val:.6f}" if p_val is not None else "N/A",
            "评估": "通过" if bartlett.get("passed") else ("未通过" if bartlett.get("passed") is False else "N/A"),
            "状态": "✅" if bartlett.get("passed") else ("⚠️" if bartlett.get("passed") is False else "❓"),
        })

        variance = self.results.get("feature_variance", {})
        rows.append({
            "检测项目": "特征方差检查",
            "结果": f"{variance.get('n_low_variance', 0)} 个低方差特征",
            "评估": "正常" if variance.get('n_low_variance', 0) == 0 else "需注意",
            "状态": "✅" if variance.get('n_low_variance', 0) == 0 else "⚠️",
        })

        return pd.DataFrame(rows)

    # ==========================================================================
    #  辅助方法
    # ==========================================================================

    @staticmethod
    def _prepare_matrix(
        feature_df: pd.DataFrame,
        feature_columns: Optional[List[str]],
        doc_id_col: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        从DataFrame中提取特征矩阵（用于检验）。

        参数:
            feature_df (pd.DataFrame): 特征DataFrame
            feature_columns (list[str] | None): 特征列名
            doc_id_col (str): ID列名

        返回:
            (X, feature_names): 特征矩阵和名称列表
        """
        if feature_columns is None:
            exclude = {doc_id_col}
            feature_columns = [
                c for c in feature_df.columns
                if c not in exclude and np.issubdtype(feature_df[c].dtype, np.number)
            ]

        df_features = feature_df[feature_columns].copy()

        # 处理缺失值
        df_features = df_features.fillna(df_features.mean())
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(df_features.mean())

        X = df_features.values.astype(np.float64)
        return X, list(df_features.columns)
