#!/usr/bin/env python3
"""
================================================================================
 单期刊 OpenAlex 数据采集脚本
 Single-Journal Data Acquisition from OpenAlex API
================================================================================

 功能：
   - 从 OpenAlex API 按期刊 ISSN 批量拉取论文元数据与摘要
   - 支持断点续传（意外中断后可从上次位置继续）
   - 内置请求限速与自动重试
   - 干运行模式：先查看预计获取量再决定是否执行
   - 输出标准 JSON 格式，可直接对接 extract_features.py

 预设期刊（--journal 参数）：
   ┌──────────────────────────────────────────────────┬─────────────┐
   │ 期刊名                                           │ ISSN        │
   ├──────────────────────────────────────────────────┼─────────────┤
   │ corrosion_science              (Corrosion Sci.)  │ 0010-938X   │
   │ corrosion                      (Corrosion)       │ 0010-9312   │
   │ corrosion_engineering          (Corros. Eng. ...)│ 1478-422X   │
   │ materials_corrosion            (Mater. Corros.)  │ 0947-5117   │
   │ anti_corrosion_methods         (Anti-Corros. ...)│ 0003-5599   │
   │ corrosion_materials_degradation(Corros. Mater. …)│ 2624-5558   │
   └──────────────────────────────────────────────────┴─────────────┘

 用法示例：
   # 干运行：仅查看某期刊预计有多少条数据
   python fetch_journal.py --journal corrosion_science --dry-run

   # 正式采集 Corrosion Science 2015-2026
   python fetch_journal.py --journal corrosion_science \
       --output raw_CorrosionScience.json

   # 自定义 ISSN 和日期范围
   python fetch_journal.py --issn 0010-938X \
       --from 2015-01-01 --to 2026-12-31 \
       --output raw_CorrosionScience.json

   # 断点续传（使用已有的检查点文件恢复）
   python fetch_journal.py --journal corrosion_science \
       --output raw_CorrosionScience.json --resume

 API Key 配置：
   方式1：设置环境变量 OPENALEX_API_KEY
   方式2：在命令行传 --api-key YOUR_KEY
   不设置也能用，但会进入"礼貌池"（访问速度较慢）

 作者：基于王淳、石洪旭原始 notebook 代码整合重构
 项目：腐蚀领域学术摘要文体特征与影响力关联分析
================================================================================
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# ============================================================================
#  预设期刊配置
# ============================================================================

JOURNALS = {
    "corrosion_science": {
        "name": "Corrosion Science",
        "issn": "0010-938X",
    },
    "corrosion": {
        "name": "Corrosion",
        "issn": "0010-9312",
    },
    "corrosion_engineering": {
        "name": "Corrosion Engineering, Science and Technology",
        "issn": "1478-422X",
    },
    "materials_corrosion": {
        "name": "Materials and Corrosion",
        "issn": "0947-5117",
    },
    "anti_corrosion_methods": {
        "name": "Anti-Corrosion Methods and Materials",
        "issn": "0003-5599",
    },
    "corrosion_materials_degradation": {
        "name": "Corrosion and Materials Degradation",
        "issn": "2624-5558",
    },
}

# 每 N 页保存一次检查点
CHECKPOINT_INTERVAL = 5

# 每页最大记录数（OpenAlex 上限为 200）
PER_PAGE = 200

# 请求间隔（秒），避免触发速率限制
REQUEST_DELAY = 0.3


# ============================================================================
#  工具函数
# ============================================================================

def load_api_key(key_arg):
    """
    按优先级获取 API Key：命令行参数 > 环境变量 > 无
    """
    if key_arg:
        return key_arg
    return os.environ.get("OPENALEX_API_KEY", None)


def format_duration(seconds):
    """将秒数格式化为人类可读的时间字符串"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}分{seconds % 60:.0f}秒"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h:.0f}小时{m:.0f}分"


def list_journals():
    """打印所有预设期刊"""
    print("\n可用的预设期刊：")
    print("-" * 70)
    for key, info in JOURNALS.items():
        print(f"  {key:<36s}  ISSN: {info['issn']:<12s}  {info['name']}")
    print("-" * 70)
    print()


def save_json(data, filepath):
    """保存数据为 JSON 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath):
    """从 JSON 文件加载数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_checkpoint_path(output_path):
    """根据输出文件路径生成检查点文件路径"""
    p = Path(output_path)
    checkpoint_name = f".checkpoint_{p.stem}.json"
    return str(p.parent / checkpoint_name)


# ============================================================================
#  核心采集逻辑
# ============================================================================

def count_records(issn, from_date, to_date):
    """
    干运行：仅查询匹配记录的总数，不实际下载数据。

    返回匹配条件的预计记录数。
    """
    from pyalex import Works

    query = Works().filter(
        primary_location={"source": {"issn": issn}},
        from_publication_date=from_date,
        to_publication_date=to_date,
        language="en",
        has_abstract=True,
        type="article",
    )

    total = query.count()
    return total


def fetch_journal(issn, journal_name, from_date, to_date,
                  output_path, api_key=None, resume=False):
    """
    从 OpenAlex 分页拉取指定期刊的全部论文数据。

    参数:
        issn:         str  — 期刊 ISSN
        journal_name: str  — 期刊显示名（用于日志）
        from_date:    str  — 起始日期 "YYYY-MM-DD"
        to_date:      str  — 结束日期 "YYYY-MM-DD"
        output_path:  str  — 输出 JSON 文件路径
        api_key:      str  — OpenAlex API Key（可选）
        resume:       bool — 是否从检查点恢复

    返回:
        list — 所有论文的 JSON 对象列表
    """
    from pyalex import Works, config

    # ---- 配置 API ----
    config.max_retries = 5
    config.retry_backoff_factor = 2
    config.retry_http_codes = [429, 500, 502, 503, 504]

    if api_key:
        config.api_key = api_key
        print("[配置] 已设置 API Key（优先访问池）")
    else:
        print("[配置] 未设置 API Key（使用礼貌池，速度较慢）")

    # ---- 检查点处理 ----
    checkpoint_path = get_checkpoint_path(output_path)
    collected_ids = set()
    results = []

    if resume and os.path.exists(checkpoint_path):
        print(f"[断点续传] 正在从检查点恢复: {checkpoint_path}")
        checkpoint_data = load_json(checkpoint_path)
        results = checkpoint_data.get("results", [])
        collected_ids = set(checkpoint_data.get("collected_ids", []))
        print(f"[断点续传] 已恢复 {len(results)} 条记录，继续采集...")
    else:
        if resume:
            print("[断点续传] 未找到检查点文件，将从头开始采集")

    # ---- 构建查询 ----
    print(f"\n[查询] 正在构建查询条件...")
    print(f"  期刊: {journal_name}")
    print(f"  ISSN: {issn}")
    print(f"  日期范围: {from_date} ~ {to_date}")
    print(f"  筛选条件: language=en, has_abstract=True, type=article")

    query = Works().filter(
        primary_location={"source": {"issn": issn}},
        from_publication_date=from_date,
        to_publication_date=to_date,
        language="en",
        has_abstract=True,
        type="article",
    )

    # ---- 获取总数 ----
    print(f"\n[统计] 正在查询符合条件的总记录数...")
    total = query.count()
    print(f"[统计] 预计总记录数: {total}")

    if total == 0:
        print("[结果] 没有找到符合条件的记录，请检查 ISSN 和日期范围是否正确。")
        return []

    # 预估时间
    pages = (total + PER_PAGE - 1) // PER_PAGE
    est_seconds = pages * REQUEST_DELAY
    print(f"[统计] 约需 {pages} 页 (每页{PER_PAGE}条)")
    print(f"[统计] 预计耗时: {format_duration(est_seconds)}（不含网络延迟）")

    # ---- 分页拉取 ----
    print(f"\n[采集] 开始分页拉取数据...")
    print("=" * 60)

    start_time = time.time()
    page_num = 0
    new_count = 0
    error_pages = 0
    max_retry_pages = 3  # 单页最大重试次数

    # 如果从检查点恢复，计算已经完成的页数
    if resume and results:
        page_num = len(results) // PER_PAGE
        print(f"[采集] 从第 {page_num + 1} 页继续...")

    try:
        for page_data in query.paginate(per_page=PER_PAGE):
            page_num += 1

            # 检查并去重（OpenAlex 偶尔会在分页间出现重复）
            page_new = 0
            for item in page_data:
                item_id = item.get("id", "")
                if item_id and item_id not in collected_ids:
                    collected_ids.add(item_id)
                    results.append(item)
                    page_new += 1

            new_count += page_new

            # 进度信息
            elapsed = time.time() - start_time
            eta = (elapsed / page_num) * (pages - page_num) if page_num > 0 else 0
            has_abstract_count = sum(
                1 for item in page_data
                if item.get("abstract_inverted_index")
                and isinstance(item["abstract_inverted_index"], dict)
            )

            print(f"  第 {page_num}/{pages} 页 | "
                  f"本页新增 {page_new} 条 (含摘要 {has_abstract_count}) | "
                  f"累计 {len(results)}/{total} | "
                  f"耗时 {format_duration(elapsed)} | "
                  f"预计剩余 {format_duration(eta)}")

            # 定期保存检查点
            if page_num % CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    "issn": issn,
                    "journal_name": journal_name,
                    "from_date": from_date,
                    "to_date": to_date,
                    "total_expected": total,
                    "page_progress": page_num,
                    "collected_count": len(results),
                    "collected_ids": list(collected_ids),
                    "results": results,
                    "last_updated": datetime.now().isoformat(),
                }
                save_json(checkpoint, checkpoint_path)
                print(f"  [检查点] 已保存 (第 {page_num} 页)")

            # 请求间隔
            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print(f"\n[中断] 用户手动中断。正在保存检查点...")
        checkpoint = {
            "issn": issn,
            "journal_name": journal_name,
            "from_date": from_date,
            "to_date": to_date,
            "total_expected": total,
            "page_progress": page_num,
            "collected_count": len(results),
            "collected_ids": list(collected_ids),
            "results": results,
            "last_updated": datetime.now().isoformat(),
        }
        save_json(checkpoint, checkpoint_path)
        print(f"[中断] 检查点已保存至: {checkpoint_path}")
        print(f"[中断] 已采集 {len(results)}/{total} 条记录")
        print(f"[中断] 下次运行添加 --resume 参数可从当前位置继续")

        # 仍然保存当前结果
        save_json(results, output_path)
        print(f"[中断] 部分结果已保存至: {output_path}")
        return results

    except Exception as e:
        print(f"\n[错误] 采集过程中出现异常: {e}")
        print(f"[错误] 正在保存检查点...")
        checkpoint = {
            "issn": issn,
            "journal_name": journal_name,
            "from_date": from_date,
            "to_date": to_date,
            "total_expected": total,
            "page_progress": page_num,
            "collected_count": len(results),
            "collected_ids": list(collected_ids),
            "results": results,
            "last_updated": datetime.now().isoformat(),
        }
        save_json(checkpoint, checkpoint_path)
        print(f"[错误] 检查点已保存至: {checkpoint_path}")
        raise

    # ---- 完成 ----
    total_elapsed = time.time() - start_time

    print("=" * 60)
    print(f"\n[完成] 数据采集完毕！")
    print(f"  总耗时:    {format_duration(total_elapsed)}")
    print(f"  采集记录:  {len(results)} 条")
    print(f"  预计总数:  {total} 条")
    print(f"  去重移除:  {new_count - len(results) if new_count > len(results) else 0} 条")

    # 保存最终结果
    save_json(results, output_path)
    print(f"  输出文件:  {output_path}")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小:  {file_size_mb:.2f} MB")

    # 清理检查点
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  [清理] 检查点文件已删除")

    # ---- 数据质量报告 ----
    print(f"\n[质量报告]")
    with_abstract_inverted = sum(
        1 for item in results
        if item.get("abstract_inverted_index")
        and isinstance(item["abstract_inverted_index"], dict)
        and len(item["abstract_inverted_index"]) > 0
    )
    with_doi = sum(1 for item in results if item.get("doi"))
    years = [item.get("publication_year") for item in results if item.get("publication_year")]

    print(f"  有 DOI:        {with_doi}/{len(results)} ({100*with_doi/max(len(results),1):.1f}%)")
    print(f"  有摘要(倒排):  {with_abstract_inverted}/{len(results)} "
          f"({100*with_abstract_inverted/max(len(results),1):.1f}%)")

    if years:
        from collections import Counter
        year_dist = Counter(years)
        print(f"  年份分布:")
        for year in sorted(year_dist.keys()):
            bar = "█" * max(1, year_dist[year] // max(1, max(year_dist.values()) // 30))
            print(f"    {year}: {year_dist[year]:<6d} {bar}")

    return results


# ============================================================================
#  命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="单期刊 OpenAlex 数据采集工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 查看所有预设期刊
  python fetch_journal.py --list

  # 干运行：仅查看数量不下载
  python fetch_journal.py --journal corrosion_science --dry-run

  # 正式采集
  python fetch_journal.py --journal corrosion_science -o raw_CorrosionScience.json

  # 断点续传
  python fetch_journal.py --journal corrosion_science -o raw_CorrosionScience.json --resume

  # 自定义 ISSN 和日期范围
  python fetch_journal.py --issn 0010-938X --from 2020-01-01 --to 2026-12-31 -o output.json

API Key 设置:
  export OPENALEX_API_KEY="your_key_here"
  或在命令行传 --api-key
        """,
    )

    # 数据源参数
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--journal",
        choices=list(JOURNALS.keys()),
        help="预设期刊的标识符",
    )
    src_group.add_argument(
        "--issn",
        help="自定义 ISSN（当不使用预设期刊时）",
    )
    src_group.add_argument(
        "--list",
        action="store_true",
        help="列出所有预设期刊并退出",
    )

    # 输出参数
    parser.add_argument(
        "-o", "--output",
        help="输出 JSON 文件路径（采集模式必需）",
    )

    # 日期范围
    parser.add_argument(
        "--from",
        dest="from_date",
        default="2015-01-01",
        help="起始日期 YYYY-MM-DD（默认: 2015-01-01）",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        default="2026-12-31",
        help="结束日期 YYYY-MM-DD（默认: 2026-12-31）",
    )

    # 模式
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式：仅查询数量，不下载数据",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次的检查点文件恢复采集",
    )

    # API 配置
    parser.add_argument(
        "--api-key",
        help="OpenAlex API Key（也可以通过环境变量 OPENALEX_API_KEY 设置）",
    )

    args = parser.parse_args()

    # ---- 列出期刊 ----
    if args.list:
        list_journals()
        return

    # ---- 确定 ISSN 和期刊名 ----
    if args.journal:
        info = JOURNALS[args.journal]
        issn = info["issn"]
        journal_name = info["name"]
    elif args.issn:
        issn = args.issn
        journal_name = f"ISSN:{issn}"
    else:
        print("[错误] 请指定 --journal 或 --issn，使用 --list 查看可用期刊")
        sys.exit(1)

    # ---- 获取 API Key ----
    api_key = load_api_key(args.api_key)

    # ---- 干运行 ----
    if args.dry_run:
        print("=" * 60)
        print("  干运行模式 (Dry Run)")
        print("=" * 60)
        print(f"  期刊:    {journal_name}")
        print(f"  ISSN:    {issn}")
        print(f"  日期:    {args.from_date} ~ {args.to_date}")
        print(f"  条件:    language=en, has_abstract=True, type=article")
        print(f"  API Key: {'已设置' if api_key else '未设置（礼貌池）'}")
        print()

        try:
            total = count_records(issn, args.from_date, args.to_date)
            print(f"  预计可获取: {total} 条记录")
            pages = (total + PER_PAGE - 1) // PER_PAGE
            print(f"  约需分页:   {pages} 页 (每页{PER_PAGE}条)")
            est = pages * REQUEST_DELAY
            print(f"  预计耗时:   {format_duration(est)}（不含网络延迟）")
        except Exception as e:
            print(f"  [错误] 查询失败: {e}")
            sys.exit(1)
        return

    # ---- 正式采集 ----
    if not args.output:
        print("[错误] 采集模式需要指定 -o/--output 输出文件路径")
        sys.exit(1)

    print("=" * 60)
    print("  单期刊 OpenAlex 数据采集工具")
    print("=" * 60)

    fetch_journal(
        issn=issn,
        journal_name=journal_name,
        from_date=args.from_date,
        to_date=args.to_date,
        output_path=args.output,
        api_key=api_key,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
