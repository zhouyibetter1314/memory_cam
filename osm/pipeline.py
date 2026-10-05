"""端到端管线：把感知层、记忆层、交互层串起来跑完一个场景序列。

用法：
    from osm.pipeline import run_sequence
    res = run_sequence(scene, cfg, seed=0, user="oracle")

对应报告 3.1 节的总体框架。
"""

import logging
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional

from .config import Config
from .interaction import AnswerGenerator, QueryEngine, SYNONYMS
from .memory import ObjectMemory
from .perception import Detector, PoseEstimator
from .metrics import (change_detection_f1, idf1, localization_errors,
                      memory_consistency, nasa_tlx, reid_accuracy)
from .utils import dist3
from .runtime import log_frame

logger = logging.getLogger(__name__)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run_sequence(scene: dict, cfg: Config, seed: int = 0,
                 user: str = "oracle") -> dict:
    """在单个场景序列上跑完整流程。

    Args:
        scene: 由 experiments.sim_dataset 生成的场景数据
        cfg: 配置（消融实验通过覆盖字段实现）
        seed: 随机种子
        user: "oracle" 模拟理想用户；"none" 表示无人参与（消融 w/o human）
    Returns:
        统计字典（各项指标的原始计数，供上层汇总）
    """
    rng = random.Random(seed)
    det = Detector(cfg, rng)
    pose_est = PoseEstimator(cfg, rng)
    mem = ObjectMemory(cfg)
    qe = QueryEngine(mem, cfg)
    ag = AnswerGenerator(cfg.frame_interval_s)

    gt_positions = {o["oid"]: list(o["pos"]) for o in scene["objects"]}
    gt_classes = {o["oid"]: o["cls"] for o in scene["objects"]}
    gt_moves = {(g, f) for g, f in scene["moves"]}      # (gt_id, 移动帧)

    detected_events: List[tuple] = []      # (gt_id, frame_id) 系统报告的变化
    n_pending = 0
    n_corrections = 0
    det_tp = defaultdict(int)
    det_tot = defaultdict(int)
    query_hits, query_tot = 0, 0
    answer_pos_errors: List[float] = []
    answers: List[str] = []

    user_labels = {o["oid"]: o["user_label"] for o in scene["objects"] if o.get("user_label")}
    obs_after_move = defaultdict(int)      # 用于评测"可检测的变化事件"
    obs_before_move = defaultdict(int)
    move_frames = dict(scene["moves"])

    frames = scene["frames"]
    for frame in frames:
        frame_started = time.perf_counter()
        est_pose = pose_est.estimate(frame["pose"])
        f = dict(frame)
        f["pose"] = est_pose                     # 用带噪位姿做反投影
        obs_list = det.predict(f)

        # 检测质量统计（mAP 近似）
        for o in obs_list:
            det_tot[o.gt_cls] += 1
            if o.cls_pred == o.gt_cls and _iou(o.bbox, o.gt_bbox) >= 0.5:
                det_tp[o.gt_cls] += 1

        for o in obs_list:
            mf = move_frames.get(o.gt_id)
            if mf is None:
                continue
            if frame["frame_id"] > mf:
                obs_after_move[o.gt_id] += 1
            else:
                obs_before_move[o.gt_id] += 1

        # -------- 人机协同 ②：用户主动纠正（约 1/3 的真实变化由用户主动告知） --------
        if user != "none":
            for g, fm in scene["moves"]:
                if frame["frame_id"] == fm + 15 and (g * 7 + fm) % 3 == 0:
                    for node in mem.all_nodes():
                        if mem.gt_major(node) == g:
                            mem.user_correct_position(node.oid, gt_positions[g],
                                                      frame["frame_id"])
                            # 人机联合捕获的变化，计入变化检测召回
                            detected_events.append((g, frame["frame_id"]))
                            break

        # -------- 人机协同 ③：部署初期的一次性命名（低负担标注） --------
        if user != "none" and frame["frame_id"] % 10 == 0:
            for node in mem.all_nodes():
                mj = mem.gt_major(node)
                if mj in user_labels and not node.user_label and node.obs_count >= 2:
                    mem.rename(node.oid, user_labels[mj])

        events = mem.update(obs_list, frame["frame_id"])

        # ---------------- 人机协同：处理待确认事件 ----------------
        for ev in events:
            if ev.kind == "pending_confirm":
                n_pending += 1
                if user == "none":
                    # 无人参与：系统只能自行决定（阈值放宽 → 自动接受）
                    keys = [k for k, v in mem.pending.items() if v["oid"] == ev.oid]
                    for k in keys:
                        item = mem.pending.get(k)
                        if item and item.get("gt_id") is not None:
                            detected_events.append((item["gt_id"], frame["frame_id"]))
                        mem.resolve_pending(k, accept=True)
                else:
                    keys = [k for k, v in mem.pending.items() if v["oid"] == ev.oid]
                    for k in keys:
                        item = mem.pending.get(k)
                        if item is None:
                            continue
                        node = mem.nodes.get(item["oid"])
                        major = mem.gt_major(node) if node else None
                        # 理想用户：只有在真值确实发生移动且时间接近时才接受
                        real = any(g == major and fm <= frame["frame_id"]
                                   for g, fm in gt_moves)
                        if real and item.get("gt_id") is not None:
                            detected_events.append((item["gt_id"], frame["frame_id"]))
                        mem.resolve_pending(k, accept=real)
                        if not real:
                            n_corrections += 1
            elif ev.kind == "moved":
                node = mem.nodes.get(ev.oid)
                major = mem.gt_major(node) if node else None
                if major is not None:
                    detected_events.append((major, frame["frame_id"]))

        # ---------------- 交互：系统运行一段时间（60 帧后）开始接受查询 ----------------
        if frame["frame_id"] >= 60 and frame["frame_id"] % 10 == 0:
            target = scene["queries"][(frame["frame_id"] // 10) % len(scene["queries"])]
            text = target["text"]
            hits = qe.retrieve(text)
            query_tot += 1
            ok = any(mem.gt_major(node) == target["gt_id"] for node, _ in hits)
            if ok and hits:      # 检索命中后，记录返回首位结果的位置误差
                top = hits[0][0]
                if mem.gt_major(top) == target["gt_id"]:
                    answer_pos_errors.append(
                        dist3(top.pos, gt_positions[target["gt_id"]]))
            query_hits += 1 if ok else 0
            if frame["frame_id"] % 50 == 0:
                answers.append(ag.generate(text, hits, frame["frame_id"]))
        log_frame(logger, frame["frame_id"], frame["timestamp"], obs_list,
                  time.perf_counter() - frame_started)

    # ---------------- 指标汇总 ----------------
    loc_errs = localization_errors(mem, gt_positions)
    # 评测协议：仅统计"变化前后各至少被观测 2 次"的真值事件。
    # 若物体在移动前从未被观测，系统无从得知其原位置，此类事件不具备可检测性。
    detectable = [(g, f) for g, f in scene["moves"]
                  if obs_after_move.get(g, 0) >= 2 and obs_before_move.get(g, 0) >= 2]
    p, r, f1 = change_detection_f1(detected_events, detectable)
    duration_min = len(frames) * cfg.frame_interval_s / 60.0
    maps = [det_tp[c] / det_tot[c] for c in det_tot if det_tot[c] >= 5]

    return {
        "scene_id": scene.get("scene_id", 0),
        "map50": sum(maps) / len(maps) if maps else float("nan"),
        "idf1": idf1(mem.frame_assign),
        "reid_acc": reid_accuracy(mem),
        "loc_error_median": sorted(loc_errs)[len(loc_errs) // 2] if loc_errs else float("nan"),
        "loc_error_mean": (sum(loc_errs) / len(loc_errs)) if loc_errs else float("nan"),
        "memory_consistency": memory_consistency(mem, gt_positions, gt_classes),
        "pending_per_10min": n_pending / duration_min * 10.0,
        "corrections_per_10min": n_corrections / duration_min * 10.0,
        "change_precision": p,
        "change_recall": r,
        "change_f1": f1,
        "n_gt_moves": len(scene["moves"]),
        "n_detectable_moves": len(detectable),
        "query_success_top3": query_hits / query_tot if query_tot else float("nan"),
        "answer_pos_error_median": (sorted(answer_pos_errors)[len(answer_pos_errors) // 2]
                                    if answer_pos_errors else float("nan")),
        "n_nodes": len([n for n in mem.nodes.values() if n.state != "removed"]),
        "n_objects_gt": len(scene["objects"]),
        "n_auto_update": mem.n_auto_update,
        "n_user_confirmed": mem.n_user_confirmed,
        "n_user_rejected": mem.n_user_rejected,
        "n_user_initiated": mem.n_user_initiated,
        "user_initiated_per_10min": mem.n_user_initiated / duration_min * 10.0,
        "tlx": nasa_tlx(n_pending / duration_min * 10.0, n_corrections / duration_min * 10.0),
        "sample_answers": answers[:4],
    }
