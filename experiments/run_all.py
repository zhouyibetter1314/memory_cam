"""一键复现实验：系统性能 + 消融实验 + 用户交互实验。

运行：
    cd code && python -m experiments.run_all

产出：
    results/metrics.json      机器可读的全部原始结果
    results/tables.md         论文用表格（Markdown）
    results/ablation.md       消融实验表
"""

import argparse
import json
import logging
import os
import sys
import time
from statistics import mean, pstdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osm.config import DEFAULT                     # noqa: E402
from osm.pipeline import run_sequence              # noqa: E402
from experiments.sim_dataset import generate_dataset  # noqa: E402
from osm.runtime import add_log_arguments, configure_logging

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

METRIC_KEYS = [
    "map50", "idf1", "reid_acc", "loc_error_median", "loc_error_mean",
    "memory_consistency", "pending_per_10min", "corrections_per_10min",
    "change_precision", "change_recall", "change_f1",
    "query_success_top3", "answer_pos_error_median", "n_nodes", "n_objects_gt", "n_auto_update",
    "n_user_confirmed", "n_user_rejected", "n_user_initiated",
    "user_initiated_per_10min", "tlx",
]

HIGHER_BETTER = {"map50", "idf1", "reid_acc", "memory_consistency",
                 "change_precision", "change_recall", "change_f1",
                 "query_success_top3"}


def aggregate(runs):
    out = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in runs if r[k] == r[k]]     # 过滤 nan
        out[k] = mean(vals) if vals else float("nan")
        out[k + "_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    return out


def fmt(v, nd=3, pct=False):
    if v != v:
        return "—"
    return f"{v * 100:.1f}" if pct else f"{v:.{nd}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="批量复现仿真实验（不限速）")
    add_log_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging(args)
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    cfg = DEFAULT
    scenes = generate_dataset(cfg, seed=cfg.seed)

    # ---------------------------------------------------------------- 主实验
    logger.info("[1/3] 系统性能实验：%s 个场景序列", len(scenes))
    main_runs = [run_sequence(s, cfg, seed=cfg.seed + s["scene_id"], user="oracle")
                 for s in scenes]
    main_agg = aggregate(main_runs)

    # ---------------------------------------------------------------- 消融实验
    ablations = {
        "Full（完整方法）": (dict(), "oracle"),
        "w/o 外观特征（仅几何）": (dict(w_app=0.0), "oracle"),
        "w/o 几何约束（仅外观）": (dict(w_geo=0.0), "oracle"),
        "w/o 人机协同（全自动更新）": (dict(auto_update_margin=1.0), "none"),
        "w/o 回环校正（位姿漂移）": (dict(pose_drift_std=0.0035), "oracle"),
    }
    # 中期基线：仅几何关联（无外观重识别）+ 无用户协同 = 中期阶段原型
    logger.info("[1.5/3] 中期基线（Baseline）")
    base_cfg = cfg.override(w_app=0.0, auto_update_margin=1.0)
    base_runs = [run_sequence(s, base_cfg, seed=cfg.seed + s["scene_id"], user="none")
                 for s in scenes]
    base_agg = aggregate(base_runs)

    logger.info("[2/3] 消融实验")
    abl_results = {}
    for name, (over, user) in ablations.items():
        c = cfg.override(**over)
        runs = [run_sequence(s, c, seed=cfg.seed + s["scene_id"], user=user) for s in scenes]
        abl_results[name] = aggregate(runs)
        logger.info("%s: reid=%.3f loc=%.3fm F1=%.3f", name,
                    abl_results[name]['reid_acc'], abl_results[name]['loc_error_median'],
                    abl_results[name]['change_f1'])

    # ---------------------------------------------------------------- 用户实验
    logger.info("[3/3] 用户交互实验（交互代价与主观负荷）")
    user_variants = {
        "常规场景 + 有人参与（本文方法）": (dict(), "oracle"),
        "常规场景 + 无用户参与": (dict(auto_update_margin=1.0), "none"),
        "困难场景（外观退化）+ 有人参与": (dict(feat_noise_std=0.032), "oracle"),
        "困难场景（外观退化）+ 无用户参与": (dict(feat_noise_std=0.032,
                                          auto_update_margin=1.0), "none"),
    }
    user_results = {}
    for name, (over, user) in user_variants.items():
        c = cfg.override(**over)
        runs = [run_sequence(s, c, seed=cfg.seed + s["scene_id"], user=user) for s in scenes]
        user_results[name] = aggregate(runs)

    # ---------------------------------------------------------------- 输出
    payload = {
        "config": cfg.__dict__ if hasattr(cfg, "__dict__") else {},
        "main": main_agg,
        "baseline": base_agg,
        "main_per_scene": main_runs,
        "ablation": abl_results,
        "user_study": user_results,
        "dataset": {
            "name": "Sim-Scene-12（合成动态室内场景）",
            "n_scenes": cfg.n_scenes,
            "frames_per_scene": cfg.frames_per_scene,
            "frame_interval_s": cfg.frame_interval_s,
            "n_objects": cfg.n_objects,
            "total_duration_min": cfg.n_scenes * cfg.frames_per_scene * cfg.frame_interval_s / 60,
        },
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---------------- 表 1：系统性能 ----------------
    rows = [
        ("检测 mAP@0.5", fmt(main_agg["map50"], 3, True) + "%", "≥ 75.0%"),
        ("多目标跟踪 IDF1", fmt(main_agg["idf1"], 3, True) + "%", "≥ 80.0%"),
        ("跨时段重识别准确率", fmt(main_agg["reid_acc"], 3, True) + "%", "≥ 85.0%"),
        ("物体定位中位误差 (m)", fmt(main_agg["loc_error_median"], 3), "≤ 0.20"),
        ("物体定位平均误差 (m)", fmt(main_agg["loc_error_mean"], 3), "—"),
        ("记忆更新一致率", fmt(main_agg["memory_consistency"], 3, True) + "%", "≥ 88.0%"),
        ("变化检测 精确率", fmt(main_agg["change_precision"], 3, True) + "%", "—"),
        ("变化检测 召回率", fmt(main_agg["change_recall"], 3, True) + "%", "—"),
        ("变化检测 F1", fmt(main_agg["change_f1"], 3, True) + "%", "≥ 80.0%"),
        ("待确认事件率 (次/10 min)", fmt(main_agg["pending_per_10min"], 2), "≤ 2.00"),
        ("用户纠正次数 (次/10 min)", fmt(main_agg["corrections_per_10min"], 2), "≤ 1.00"),
        ("查询成功率 Top-3", fmt(main_agg["query_success_top3"], 3, True) + "%", "≥ 85.0%"),
        ("NASA-TLX 主观负荷", fmt(main_agg["tlx"], 1), "≤ 40.0"),
        ("记忆库节点数 / 真值物体数",
         f"{main_agg['n_nodes']:.0f} / {main_agg['n_objects_gt']:.0f}", "接近 1:1"),
    ]
    t1 = ["| 指标 | 实验结果 | 目标值 |", "| --- | --- | --- |"]
    t1 += [f"| {a} | {b} | {c} |" for a, b, c in rows]

    # ---------------- 表 2：消融 ----------------
    keys = [("reid_acc", "重识别准确率"), ("loc_error_median", "定位中位误差 (m)"),
            ("memory_consistency", "记忆一致率"), ("change_f1", "变化检测 F1"),
            ("pending_per_10min", "待确认/10 min"), ("query_success_top3", "查询成功率 Top-3")]
    t2 = ["| 配置 | " + " | ".join(k[1] for k in keys) + " |",
          "| --- | " + " | ".join("---" for _ in keys) + " |"]
    for name, agg in abl_results.items():
        cells = []
        for k, _ in keys:
            v = agg[k]
            cells.append(fmt(v, 3, pct=(k in HIGHER_BETTER)) + ("%" if k in HIGHER_BETTER else ""))
        t2.append(f"| {name} | " + " | ".join(cells) + " |")

    # ---------------- 表 3：用户实验 ----------------
    ukeys = [("pending_per_10min", "待确认/裁决 (次/10 min)"),
             ("user_initiated_per_10min", "主动纠正 (次/10 min)"),
             ("memory_consistency", "记忆一致率"),
             ("query_success_top3", "查询成功率 Top-3"), ("tlx", "NASA-TLX")]
    t3 = ["| 交互配置 | " + " | ".join(k[1] for k in ukeys) + " |",
          "| --- | " + " | ".join("---" for _ in ukeys) + " |"]
    for name, agg in user_results.items():
        cells = []
        for k, _ in ukeys:
            v = agg[k]
            cells.append(fmt(v, 3, pct=(k in HIGHER_BETTER)) + ("%" if k in HIGHER_BETTER else ""))
        t3.append(f"| {name} | " + " | ".join(cells) + " |")

    # ---------------- 表 4：中期基线 vs 最终系统 ----------------
    evo_keys = [("map50", "检测 mAP@0.5", True), ("idf1", "跟踪 IDF1", True),
                ("reid_acc", "重识别准确率", True), ("loc_error_median", "定位中位误差 (m)", False),
                ("memory_consistency", "记忆一致率", True), ("change_f1", "变化检测 F1", True),
                ("pending_per_10min", "待确认/裁决 (次/10 min)", False),
                ("query_success_top3", "查询成功率 Top-3", True)]
    t4 = ["| 指标 | 中期基线 | 最终系统 | 变化 |", "| --- | --- | --- | --- |"]
    for k, name, pct in evo_keys:
        b, f = base_agg[k], main_agg[k]
        delta = f - b
        arrow = ("+" if delta >= 0 else "−")
        unit = "%" if pct else ""
        t4.append(f"| {name} | {fmt(b, 3, pct=pct)}{unit} | {fmt(f, 3, pct=pct)}{unit} "
                  f"| {arrow}{abs(delta) * (100 if pct else 1):.3f}{unit} |")

    samples = "\n".join(f"- {a}" for a in main_runs[0].get("sample_answers", []))
    md = f"""# 实验结果（自动生成，勿手工编辑）

数据集：{payload['dataset']['name']}；{payload['dataset']['n_scenes']} 个序列 × \
{payload['dataset']['frames_per_scene']} 帧，帧间隔 {payload['dataset']['frame_interval_s']} s，\
总时长 {payload['dataset']['total_duration_min']:.0f} 分钟；每场景 {payload['dataset']['n_objects']} 个物体。
随机种子 {cfg.seed}，命令 `python -m experiments.run_all`。

## 表 1 系统性能

{chr(10).join(t1)}

## 表 4 中期基线 → 最终系统（演进对比）

{chr(10).join(t4)}

## 表 2 消融实验

{chr(10).join(t2)}

## 表 3 用户交互实验

{chr(10).join(t3)}

## 问答示例（场景 0）

{samples}
"""
    with open(os.path.join(RESULTS, "tables.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(RESULTS, "ablation.md"), "w", encoding="utf-8") as f:
        f.write("# 消融实验\n\n" + "\n".join(t2))

    print("\n" + md)
    logger.info("结果已写入 %s（用时 %.1fs）", RESULTS, time.time() - t0)


if __name__ == "__main__":
    main()
