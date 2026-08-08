"""
report.py — 生成最终一页表和热图

行 = 统计量, 列 = 任务族, 单元格 = p50 [p5, p99]
按 (task, layer) 分组（约束 15）。
"""

import json
import os
from collections import defaultdict

import numpy as np


def percentile_summary(values: list[float]) -> dict:
    """计算分布摘要。"""
    if not values:
        return {"p5": None, "p50": None, "p95": None, "p99": None,
                "mean": None, "min": None, "max": None, "n": 0}
    arr = np.array(values)
    return {
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(values),
    }


def format_cell(summary: dict) -> str:
    """格式化单元格：p50 [p5, p99]"""
    if summary["p50"] is None:
        return "N/A"
    return f"{summary['p50']:.4f} [{summary['p5']:.4f}, {summary['p99']:.4f}]"


def generate_report(results_dir: str, output_path: str):
    """
    从 results 目录生成汇总报告。

    预期输入：results_dir 下每个任务有一个子目录，
    包含 per-layer per-step 的 JSON 统计文件。
    """
    report = {
        "measurements": {},
        "per_task_layer": defaultdict(lambda: defaultdict(list)),
    }

    # 扫描所有任务目录
    if not os.path.exists(results_dir):
        return report

    for task_dir in sorted(os.listdir(results_dir)):
        task_path = os.path.join(results_dir, task_dir)
        if not os.path.isdir(task_path):
            continue

        stats_file = os.path.join(task_path, "stats.json")
        if not os.path.exists(stats_file):
            continue

        with open(stats_file) as f:
            stats = json.load(f)

        task_name = stats.get("task_name", task_dir)
        for layer_stats in stats.get("layers", []):
            layer = layer_stats["layer"]
            key = (task_name, layer)

            for metric in ["coverage", "F_oneshot", "F_bestfirst",
                           "churn_rate", "touched_blocks",
                           "churn_touched_blocks"]:
                if metric in layer_stats:
                    report["per_task_layer"][metric][key].append(
                        layer_stats[metric]
                    )

    # 汇总
    for metric, data in report["per_task_layer"].items():
        report["measurements"][metric] = {}
        for (task, layer), values in data.items():
            report["measurements"][metric][f"{task}_L{layer}"] = \
                percentile_summary(values)

    # 写出
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report["measurements"], f, indent=2)

    return report
