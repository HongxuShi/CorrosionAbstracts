#!/usr/bin/env python3
"""
================================================================================
 多期刊特征数据集汇合脚本
 Merge Multi-Journal Feature Datasets
================================================================================

 功能：
   读取 processed/ 目录下所有 *_features.csv，
   纵向拼接 → DOI 去重 → 质量检查 → 输出统一数据集。

 用法：
   python merge_datasets.py

   或指定输入输出路径：
   python merge_datasets.py --input-dir ./processed --output ./processed/merged_all_features.csv

 汇合逻辑：
   1. 扫描并读取所有 CSV
   2. 检查列一致性，不兼容时报警
   3. 按 DOI 去重（保留首次出现的记录）
   4. 按 year + source_journal 排序
   5. 输出合并文件 + 汇合报告
================================================================================
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import pandas as pd


def merge_datasets(input_dir, output_path):
    """
    扫描目录、读取所有特征 CSV、汇合为单个数据集。

    参数:
        input_dir:   str — 包含 *_features.csv 的目录
        output_path: str — 输出文件路径

    返回:
        pd.DataFrame — 汇合后的 DataFrame
    """
    # ---- 扫描文件 ----
    pattern = os.path.join(input_dir, "*_features.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[错误] 在 {input_dir}/ 下未找到任何 *_features.csv 文件")
        print(f"[提示] 请先将特征提取结果放入 {input_dir}/ 目录")
        sys.exit(1)

    print(f"[扫描] 找到 {len(files)} 个特征文件:")
    for f in files:
        size_kb = os.path.getsize(f) / 1024
        print(f"  - {os.path.basename(f):<50s} ({size_kb:.0f} KB)")

    # ---- 逐文件读取 ----
    print(f"\n[读取] 正在逐个加载...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            journal = os.path.basename(f).replace("_features.csv", "")
            print(f"  ✓ {journal:<45s} {len(df):>6d} 条, {len(df.columns):>2d} 列")
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ {os.path.basename(f)}: {e}")

    if not dfs:
        print("[错误] 没有成功加载任何文件")
        sys.exit(1)

    # ---- 列一致性检查 ----
    print(f"\n[检查] 列一致性...")
    base_cols = set(dfs[0].columns)
    all_ok = True
    for i, df in enumerate(dfs[1:], 1):
        current_cols = set(df.columns)
        missing = base_cols - current_cols
        extra = current_cols - base_cols
        fname = os.path.basename(files[i])
        if missing:
            print(f"  ⚠ {fname}: 缺少列 {missing}")
            all_ok = False
        if extra:
            print(f"  ⚠ {fname}: 多余列 {extra}")
    if all_ok:
        print(f"  ✓ 所有文件列结构一致 ({len(base_cols)} 列)")

    # ---- 纵向拼接 ----
    print(f"\n[合并] 正在纵向拼接...")
    merged = pd.concat(dfs, ignore_index=True)
    total_before = len(merged)
    print(f"  合并后总计: {total_before} 条")

    # ---- 期刊分布 ----
    if 'source_journal' in merged.columns:
        print(f"\n  各期刊记录数:")
        for journal, count in merged['source_journal'].value_counts().items():
            print(f"    {journal:<45s} {count:>6d} 条")

    # ---- DOI 去重 ----
    if 'doi' in merged.columns:
        dup_count = merged['doi'].duplicated().sum()
        if dup_count > 0:
            print(f"\n[去重] 发现 {dup_count} 条重复 DOI，保留首次出现")
            merged = merged.drop_duplicates(subset='doi', keep='first')
        else:
            print(f"\n[去重] ✓ 无重复 DOI")
    else:
        print(f"\n[去重] ⚠ 未找到 doi 列，跳过去重")

    # ---- 排序 ----
    sort_cols = []
    for col in ['year', 'source_journal']:
        if col in merged.columns:
            sort_cols.append(col)
    if sort_cols:
        merged = merged.sort_values(sort_cols, ignore_index=True)

    # ---- 保存 ----
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    merged.to_csv(output_path, index=False, encoding='utf-8')
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # ---- 汇总报告 ----
    print(f"\n{'=' * 60}")
    print(f" 汇合完成")
    print(f"{'=' * 60}")
    print(f"  输入文件数:   {len(files)}")
    print(f"  合并前总条数: {total_before}")
    print(f"  去重后条数:   {len(merged)}")
    print(f"  去除重复:     {total_before - len(merged)}")
    print(f"  输出文件:     {output_path}")
    print(f"  文件大小:     {file_size_mb:.2f} MB")

    if 'year' in merged.columns:
        year_range = f"{int(merged['year'].min())}–{int(merged['year'].max())}"
        print(f"  年份范围:     {year_range}")

    if 'source_journal' in merged.columns:
        print(f"  期刊数:       {merged['source_journal'].nunique()}")

    feature_cols = ['ASL', 'MWL', 'LD', 'LC', 'JD', 'HD', 'NCC']
    available_features = [c for c in feature_cols if c in merged.columns]
    complete = merged[available_features].dropna().shape[0]
    print(f"  特征完整记录: {complete}/{len(merged)} "
          f"({100 * complete / max(len(merged), 1):.1f}%)")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="多期刊特征数据集汇合工具",
    )
    parser.add_argument(
        '--input-dir',
        default=None,
        help='包含 *_features.csv 的目录（默认: 脚本同级的 ../processed）',
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='输出文件路径（默认: ../processed/merged_all_features.csv）',
    )

    args = parser.parse_args()

    # 默认路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.join(script_dir, '..', 'processed')
    output_path = args.output or os.path.join(script_dir, '..', 'processed',
                                               'merged_all_features.csv')

    input_dir = os.path.abspath(input_dir)
    output_path = os.path.abspath(output_path)

    print("=" * 60)
    print("  多期刊特征数据集汇合工具")
    print("=" * 60)
    print(f"  输入目录: {input_dir}")
    print(f"  输出文件: {output_path}")
    print("=" * 60)

    merge_datasets(input_dir, output_path)


if __name__ == '__main__':
    main()
