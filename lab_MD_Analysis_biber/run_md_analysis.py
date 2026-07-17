#!/usr/bin/env python3
"""
================================================================================
 MD分析命令行入口脚本
 Command-Line Entry Point for MD Analysis Pipeline
================================================================================

功能说明：
  提供命令行接口，方便用户在终端中直接运行MD分析流程。

用法示例：
  # 基本用法：分析CSV文件中的摘要
  python scripts/run_md_analysis.py processed/merged_all_features.csv

  # 指定摘要列名
  python scripts/run_md_analysis.py data.csv --text-col abstract

  # 不进行因子旋转
  python scripts/run_md_analysis.py data.csv --no-rotation

  # 指定输出目录
  python scripts/run_md_analysis.py data.csv -o results/my_analysis

  # 使用更多维度
  python scripts/run_md_analysis.py data.csv --n-components 5

  # 查看帮助
  python scripts/run_md_analysis.py --help

依赖：
  md_analysis 包（位于本脚本的上级目录）

作者：基于PRD规范开发
日期：2026/07/17
================================================================================
"""

import sys
import argparse
from pathlib import Path

# ============================================================================
#  确保能找到 md_analysis 包
#  脚本位于 scripts/ 目录下，包在项目根目录的 md_analysis/ 中
# ============================================================================

# 将项目根目录添加到 sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from md_analysis.pipeline import MDPipeline


# ============================================================================
#  命令行参数解析
# ============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    返回:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="MD分析可行性测试工具 — "
                    "测试科技论文摘要数据集是否适合进行Multi-Dimensional Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本分析
  python scripts/run_md_analysis.py processed/merged_all_features.csv

  # 使用Excel文件
  python scripts/run_md_analysis.py data.xlsx --text-col Abstract

  # 自定义参数
  python scripts/run_md_analysis.py data.csv --n-components 5 --no-rotation

更多信息请参见项目 README.md 和 docs/ 目录下的文档。
        """
    )

    # ---- 必需参数 ----
    parser.add_argument(
        "input",
        metavar="INPUT_FILE",
        help="输入文件路径（CSV或Excel格式）。文件需包含摘要文本列。"
    )

    # ---- 输入相关 ----
    parser.add_argument(
        "--text-col",
        default="abstract",
        metavar="COLUMN",
        help="摘要文本所在的列名 (默认: abstract)"
    )
    parser.add_argument(
        "--id-col",
        default=None,
        metavar="COLUMN",
        help="文档ID所在的列名 (默认: 自动生成行号)"
    )

    # ---- 分析参数 ----
    parser.add_argument(
        "--n-components",
        type=int,
        default=4,
        metavar="N",
        help="PCA降维的维度数 (默认: 4，即Biber MD分析的四个维度)"
    )
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=30,
        metavar="N",
        help="最小词数阈值，短于此值的摘要将被过滤 (默认: 30)"
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="不进行Promax因子旋转 (默认进行旋转以增强可解释性)"
    )

    # ---- 输出相关 ----
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        metavar="DIR",
        help="报告输出目录 (默认: results/md_analysis/)"
    )
    parser.add_argument(
        "--no-save-features",
        action="store_true",
        help="不保存中间特征矩阵 (默认会保存到输出目录)"
    )

    # ---- 高级选项 ----
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        metavar="MODEL",
        help="spaCy模型名称 (默认: en_core_web_sm)"
    )

    return parser


# ============================================================================
#  主函数
# ============================================================================

def main():
    """
    命令行入口主函数。

    解析命令行参数 → 初始化管道 → 运行分析 → 输出结果
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    # ---- 检查输入文件 ----
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    # ---- 打印启动信息 ----
    print("=" * 60)
    print(" MD分析可行性测试工具 v0.1.0")
    print(" Multi-Dimensional Analysis Feasibility Testing")
    print("=" * 60)
    print(f"  输入文件:   {input_path}")
    print(f"  文本列:     {args.text_col}")
    print(f"  PCA维度:    {args.n_components}")
    print(f"  最小词数:   {args.min_word_count}")
    print(f"  Promax旋转: {'否' if args.no_rotation else '是'}")
    print(f"  输出目录:   {args.output_dir or 'results/md_analysis/'}")
    print("=" * 60)
    print()

    # ---- 初始化管道 ----
    pipeline = MDPipeline(
        spacy_model=args.spacy_model,
        n_components=args.n_components,
        min_word_count=args.min_word_count,
        output_dir=args.output_dir,
    )

    # ---- 运行分析 ----
    try:
        results = pipeline.run(
            input_file=str(input_path),
            text_col=args.text_col,
            id_col=args.id_col,
            apply_rotation=not args.no_rotation,
            save_features=not args.no_save_features,
        )

        # ---- 输出关键结论 ----
        report_path = results.get("report_path", "")
        suitability = results.get("suitability_results", {})

        print(f"\n[Report] 详细报告: {report_path}")
        verdict = suitability.get('verdict', '未知')
        print(f"[Verdict] 适配性判断: {verdict}")

        return 0

    except Exception as e:
        print(f"\n[ERROR] 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ============================================================================
#  入口
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
