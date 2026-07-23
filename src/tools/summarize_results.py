#!/usr/bin/env python3
"""将评估结果汇总成 Markdown 报告，便于技术交付与汇报。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def build_report(metrics: Dict[str, Any], config: Dict[str, Any], csv_rows: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append("# 评估结果摘要")
    lines.append("")
    lines.append("## 1. 基本信息")
    lines.append("")
    lines.append(f"- 结果目录: {metrics.get('results_dir', 'N/A')}")
    lines.append(f"- 样本数: {metrics.get('samples', 'N/A')}")
    lines.append("")

    lines.append("## 2. 核心指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | ---: |")
    for key in ["miou", "oa", "mse", "rmse", "nmse"]:
        if key in metrics:
            lines.append(f"| {key} | {format_metric(metrics[key])} |")

    if "pck" in metrics and isinstance(metrics["pck"], dict):
        lines.append("")
        lines.append("### PCK")
        lines.append("")
        lines.append("| 阈值 | PCK |")
        lines.append("| --- | ---: |")
        for threshold, value in sorted(metrics["pck"].items()):
            lines.append(f"| {threshold} | {format_metric(value)} |")

    if csv_rows:
        lines.append("")
        lines.append("## 3. 样本级统计摘要")
        lines.append("")
        lines.append("| 指标 | 均值 | 最大值 | 最小值 |")
        lines.append("| --- | ---: | ---: | ---: |")
        numeric_columns = []
        for column in csv_rows[0].keys():
            try:
                values = [numeric_value(row[column]) for row in csv_rows]
                if all(not (value != value) for value in values):
                    numeric_columns.append((column, values))
            except Exception:
                continue
        for column, values in numeric_columns:
            if not values:
                continue
            finite_values = [value for value in values if value == value]
            if not finite_values:
                continue
            lines.append(
                f"| {column} | {mean(finite_values):.4f} | {max(finite_values):.4f} | {min(finite_values):.4f} |"
            )

    lines.append("")
    lines.append("## 4. 结论建议")
    lines.append("")
    miou_value = numeric_value(metrics.get("miou", float("nan")))
    if miou_value == miou_value and miou_value >= 0.6:
        lines.append("- 当前结果显示分割与定位任务具备较好的初步可行性，可继续扩大实验规模。")
    elif miou_value == miou_value and miou_value >= 0.4:
        lines.append("- 当前结果已具备一定基础，但仍建议增加数据增强、标注质量控制与更稳健的对比实验。")
    else:
        lines.append("- 当前结果仍偏弱，建议先检查数据质量、标注一致性与训练设置后再评估业务可行性。")

    if config:
        lines.append("")
        lines.append("## 5. 配置概览")
        lines.append("")
        lines.append("| 配置项 | 值 |")
        lines.append("| --- | --- |")
        for key, value in sorted(config.items()):
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f"| {key} | {value} |")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成评估结果 Markdown 报告")
    parser.add_argument("--results_dir", required=True, help="结果目录，包含 metrics_summary.json")
    parser.add_argument("--output", default="", help="输出 Markdown 文件路径，默认为结果目录下的 summary_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    metrics_path = results_dir / "metrics_summary.json"
    config_path = results_dir / "config.json"
    detail_path = results_dir / "evaluation_details.csv"
    output_path = Path(args.output).resolve() if args.output else results_dir / "summary_report.md"

    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到评估摘要文件: {metrics_path}")

    metrics = load_json(metrics_path)
    metrics["results_dir"] = str(results_dir)
    config = load_json(config_path) if config_path.exists() else {}
    csv_rows = load_csv_rows(detail_path)

    report = build_report(metrics, config, csv_rows)
    output_path.write_text(report, encoding="utf-8")
    print(f"报告已生成: {output_path}")


if __name__ == "__main__":
    main()
