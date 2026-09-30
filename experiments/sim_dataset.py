"""合成动态场景数据集 Sim-Scene-12 / Sim-Dyn（仿真版）。

生成 12 个"日常室内场景"序列，每个序列 200 帧（帧间隔 3 s，即 10 分钟），
包含：物体移动事件、临时遮挡、视角变化、外观扰动、类别混淆。

用途：在缺少真实第一视角数据集时，为系统提供可复现的定量评测环境。
真实数据替换方式：保持 frames 的字段结构（pose / visible / frame_id /
timestamp），把 Detector 换成真实模型即可，上层管线无需改动。
"""

import math
import random
from typing import Dict, List

from osm.perception import DepthEstimator

CLASSES: Dict[str, float] = {          # 类别 -> 常见放置高度（m）
    "水杯": 0.90, "钥匙": 0.80, "背包": 0.60, "眼镜": 0.85, "手机": 0.80,
    "遥控器": 0.85, "书本": 0.85, "充电器": 0.50, "耳机": 0.80, "鼠标": 0.80,
    "剪刀": 0.85, "药盒": 0.90, "手表": 0.80, "雨伞": 0.70, "水壶": 0.90,
    "笔记本": 0.80, "台灯": 1.10, "拖鞋": 0.10,
}

# 高频类别（场景中会出现多个同类实例，用于检验"同类多实例"的区分能力）
CORE_CLASSES = ["水杯", "钥匙", "背包", "眼镜", "手机", "遥控器",
                "书本", "充电器", "耳机", "鼠标", "水壶", "笔记本"]

# 外观原型构成：类内共享原型权重 vs 个体差异权重（真实中同款物品外观高度相似）
CLS_PROTO_W = 0.97
INST_W = 0.24

CONFUSABLE = {                          # 易混淆类别（模拟分类器错误）
    "水杯": ["水壶", "药盒"], "水壶": ["水杯"], "钥匙": ["手表", "剪刀"],
    "背包": ["书本", "笔记本"], "眼镜": ["手表"], "手机": ["遥控器", "手表"],
    "遥控器": ["手机", "鼠标"], "书本": ["笔记本", "背包"],
    "充电器": ["鼠标", "遥控器"], "耳机": ["剪刀"], "鼠标": ["充电器"],
    "剪刀": ["钥匙", "耳机"], "药盒": ["水杯"], "手表": ["眼镜", "钥匙"],
    "雨伞": ["水壶"], "笔记本": ["书本"], "台灯": ["水壶"], "拖鞋": ["书本"],
}

QUERY_TEMPLATES = [
    "我的{w}放在哪了？",
    "{w}在哪里？",
    "帮我找一下{w}",
    "上次什么时候看到{w}的？",
]

# 用户自定义名称的前缀（模拟用户在部署初期为常用物品命名，属低负担一次性标注）
LABEL_PREFIX = ["常用", "备用", "随身", "办公", "睡前", "出门"]

CHANGE_QUERIES = ["最近有什么东西位置变了？", "房间里哪些物品被移动过？"]


def _traj(frame_id: int, n_frames: int, room: float):
    """相机沿之字形路径巡检，并左右摆头扫描（扩大视野覆盖）。"""
    t = frame_id / max(1, n_frames - 1)
    legs = 4
    u = (t * legs) % 1.0
    lane = int(t * legs)
    z = 0.6 + (room - 1.2) * (lane / max(1, legs - 1))
    x = 0.6 + (room - 1.2) * u if lane % 2 == 0 else (room - 0.6) - (room - 1.2) * u
    forward = 1.0 if lane % 2 == 0 else -1.0
    base = math.atan2(forward, 0.0)                 # 前进方向
    theta = base + 0.85 * math.sin(t * math.pi * 8)  # 摆头扫描
    return (x, z, theta)


def generate_scene(scene_id: int, cfg, seed: int) -> dict:
    rng = random.Random(seed * 1000 + scene_id)
    room = cfg.room_size
    names = list(CLASSES)

    # 类别级外观原型（同类实例共享，模拟"同款物品外观高度相似"）
    cls_proto = {c: [rng.gauss(0, 1) for _ in range(cfg.feat_dim)] for c in CORE_CLASSES}
    for c, v in cls_proto.items():
        n = math.sqrt(sum(x * x for x in v)) or 1e-12
        cls_proto[c] = [x / n for x in v]

    # 物体类别：从高频类别中随机分配，保证出现"同类多实例"
    cls_list = [rng.choice(CORE_CLASSES) for _ in range(cfg.n_objects)]
    for c in rng.sample(CORE_CLASSES, 5):        # 强制 5 个类别至少 2 个实例
        idx = rng.randrange(cfg.n_objects)
        cls_list[idx] = c
        cls_list[(idx + 3) % cfg.n_objects] = c

    objects = []
    for i in range(cfg.n_objects):
        cls = cls_list[i]
        inst = [rng.gauss(0, 1) for _ in range(cfg.feat_dim)]
        ni = math.sqrt(sum(x * x for x in inst)) or 1e-12
        inst = [x / ni for x in inst]
        proto = [CLS_PROTO_W * c + INST_W * e for c, e in zip(cls_proto[cls], inst)]
        n = math.sqrt(sum(x * x for x in proto)) or 1e-12
        objects.append({
            "oid": i + 1,
            "cls": cls,
            "pos": [rng.uniform(0.4, room - 0.4), CLASSES[cls] + rng.uniform(-0.06, 0.06),
                    rng.uniform(0.4, room - 0.4)],
            "proto_feat": [x / n for x in proto],
            "confusable": CONFUSABLE.get(cls, [CORE_CLASSES[0]]),
            "occlusion_phase": rng.uniform(0, 2 * math.pi),
        })

    # ---------------- 真实移动事件 ----------------
    moves = []
    chosen = rng.sample(objects, min(cfg.move_events_per_scene, len(objects)))
    for o in chosen:
        f_move = rng.randint(int(cfg.frames_per_scene * 0.25),
                             int(cfg.frames_per_scene * 0.8))
        moves.append((o["oid"], f_move))

    frames = []
    tan_h = math.tan(DepthEstimator.FOV_H)
    tan_v = math.tan(DepthEstimator.FOV_V)
    for fid in range(cfg.frames_per_scene):
        pose = _traj(fid, cfg.frames_per_scene, room)
        # 应用已发生的移动
        for oid, fm in moves:
            if fid == fm:
                o = next(x for x in objects if x["oid"] == oid)
                ang = rng.uniform(0, 2 * math.pi)
                d = rng.uniform(0.8, 2.0)
                o["pos"][0] = min(max(o["pos"][0] + d * math.cos(ang), 0.3), room - 0.3)
                o["pos"][2] = min(max(o["pos"][2] + d * math.sin(ang), 0.3), room - 0.3)
        cx_c, cz_c, th = pose
        visible = []
        for o in objects:
            dx, dz = o["pos"][0] - cx_c, o["pos"][2] - cz_c
            dist = math.hypot(dx, dz)
            if dist > 4.5 or dist < 0.8:       # 太远看不清 / 太近出画
                continue
            ang = math.atan2(dx, dz) - th
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            if abs(ang) > DepthEstimator.FOV_H:
                continue
            # 临时遮挡：随时间起伏 + 距离衰减
            occ = 0.35 * max(0.0, math.sin(fid * 0.05 + o["occlusion_phase"]))
            vis = max(0.15, 1.0 - occ - 0.05 * dist)
            u = math.tan(ang) / tan_h
            if abs(u) > 0.96:                  # 中心超出画面 → 本帧不参与观测
                continue
            cx = 0.5 + 0.5 * u
            hgt = o["pos"][1]
            cy = 0.5 - (hgt - DepthEstimator.CAM_H) / (2 * dist * tan_v)
            if not (0.05 < cy < 0.95):         # 高度方向出画
                continue
            w = min(0.6, max(0.06, 0.28 / dist))
            hh = min(0.6, max(0.06, 0.42 / dist))
            visible.append({
                "oid": o["oid"], "cls": o["cls"], "proto_feat": o["proto_feat"],
                "confusable": o["confusable"], "dist": dist, "visibility": vis,
                "bbox": (min(max(cx - w / 2, 0.0), 1.0), min(max(cy - hh / 2, 0.0), 1.0),
                         min(max(cx + w / 2, 0.0), 1.0), min(max(cy + hh / 2, 0.0), 1.0)),
            })
        frames.append({
            "frame_id": fid,
            "timestamp": fid * cfg.frame_interval_s,
            "pose": pose,
            "visible": visible,
        })

    # ---------------- 用户命名（一次性标注，约 40% 物品） ----------------
    for o in objects:
        o["user_label"] = None
    for o in rng.sample(objects, int(cfg.n_objects * 0.4)):
        o["user_label"] = rng.choice(LABEL_PREFIX) + o["cls"]

    # ---------------- 查询集合 ----------------
    from osm.interaction import SYNONYMS
    canon2words: Dict[str, List[str]] = {}
    for w, c in SYNONYMS.items():
        canon2words.setdefault(c, []).append(w)
    labeled = [o for o in objects if o["user_label"]]
    queries = []
    for i in range(cfg.queries_per_scene):
        if i % 7 == 6:
            queries.append({"text": rng.choice(CHANGE_QUERIES), "gt_id": None})
            continue
        # 70% 的提问针对用户命名过的物品（真实使用习惯）
        if labeled and rng.random() < 0.7:
            o = rng.choice(labeled)
            queries.append({"text": rng.choice(["我的{w}在哪？", "{w}放在哪了？", "帮我找{w}"])
                            .format(w=o["user_label"]), "gt_id": o["oid"]})
        else:
            o = rng.choice(objects)
            words = canon2words.get(o["cls"], [o["cls"]])
            queries.append({"text": rng.choice(QUERY_TEMPLATES).format(w=rng.choice(words)),
                            "gt_id": o["oid"]})

    return {"scene_id": scene_id, "objects": objects, "frames": frames,
            "moves": moves, "queries": queries}


def generate_dataset(cfg, seed: int = 0) -> List[dict]:
    return [generate_scene(i, cfg, seed) for i in range(cfg.n_scenes)]
