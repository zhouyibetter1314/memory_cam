"""感知层：物体与空间信息提取。

对应报告 3.2 节。四类能力：
    1) 物体检测与分类      Detector
    2) 单目深度估计 + 反投影 DepthEstimator
    3) 视觉定位（位姿）     PoseEstimator
    4) 外观编码/重识别      AppearanceEncoder

默认使用"仿真后端"（SimBackend），保证无 GPU 环境可复现；真实模型只需替换
对应类的 predict() 实现，接口签名保持不变（见各类注释中的 TODO）。
"""

import math
import random
from typing import List, Sequence, Tuple

from .types import Observation
from .utils import l2_normalize


class Detector:
    """物体检测与分类。

    仿真后端：对视场内可见物体按 det_score 采样，输出带噪声的 bbox 与类别。
    真实后端：TODO 替换为 YOLOv8 / Grounding-DINO，接口 predict(image) -> List[dict]
             每个 dict 至少包含 bbox / cls / score。
    """

    def __init__(self, cfg, rng: random.Random):
        self.cfg = cfg
        self.rng = rng

    def predict(self, frame: dict) -> List[Observation]:
        cfg = self.cfg
        obs_list: List[Observation] = []
        pose = frame["pose"]                      # (x, y, theta)
        for obj in frame["visible"]:
            # 检测召回：遮挡越严重召回越低；小物体/远距离更难
            p_det = 0.93 * obj["visibility"] * (1.0 - 0.05 * obj["dist"] / 4.0)
            if self.rng.random() > p_det:
                continue
            score = self.rng.uniform(0.55, 0.98)
            if score < cfg.det_score_thresh:
                continue
            # 类别预测：以 0.94 概率正确，否则预测为混淆类别
            cls_pred = obj["cls"]
            if self.rng.random() > 0.94:
                cls_pred = self.rng.choice(obj["confusable"])
            # bbox：真值框 + 与框尺寸成正比的高斯扰动，并裁剪到 [0,1]
            bx1, by1, bx2, by2 = obj["bbox"]
            jx, jy = 0.05 * (bx2 - bx1), 0.05 * (by2 - by1)
            bbox = (
                min(max(bx1 + self.rng.gauss(0, jx), 0.0), 1.0),
                min(max(by1 + self.rng.gauss(0, jy), 0.0), 1.0),
                min(max(bx2 + self.rng.gauss(0, jx), 0.0), 1.0),
                min(max(by2 + self.rng.gauss(0, jy), 0.0), 1.0),
            )
            # 单目深度：真值距离 + 相对噪声
            depth = max(0.3, obj["dist"] * (1.0 + self.rng.gauss(0, cfg.depth_noise_ratio)))
            pos_world = DepthEstimator.back_project(bbox, depth, pose)
            feat = AppearanceEncoder.observe(obj["proto_feat"], cfg, self.rng)
            obs_list.append(
                Observation(
                    frame_id=frame["frame_id"],
                    timestamp=frame["timestamp"],
                    cls_pred=cls_pred,
                    cls_score=score,
                    bbox=bbox,
                    depth=depth,
                    cam_pose=pose,
                    feat=feat,
                    pos_world=pos_world,
                    gt_id=obj["oid"],
                    gt_bbox=obj["bbox"],
                    gt_cls=obj["cls"],
                )
            )
        return obs_list


class DepthEstimator:
    """单目深度估计 + 像素→世界坐标反投影。

    真实后端：TODO 替换为 DepthAnything / Metric3D，输出每像素深度后按 bbox 取中位深度。
    """

    FOV_H = math.radians(45.0)   # 水平半视场角
    FOV_V = math.radians(30.0)   # 垂直半视场角
    CAM_H = 1.4                  # 相机高度（m）

    @classmethod
    def back_project(cls, bbox: Tuple[float, float, float, float], depth: float,
                     pose: Tuple[float, float, float]) -> List[float]:
        """把检测框中心按针孔模型反投影到世界坐标系。

        Args:
            bbox: 归一化框 (x1, y1, x2, y2)
            depth: 物体中心深度（水平距离，m）
            pose: 相机位姿 (x, z, theta)
        Returns:
            世界坐标 [x, y, z]（y 为高度）
        """
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        # 水平角：由归一化横坐标反解
        ang = math.atan((cx - 0.5) * 2.0 * math.tan(cls.FOV_H))
        # 相机→世界（绕 y 轴旋转 theta）
        x_c, z_c = pose[0], pose[1]
        th = pose[2]
        xw = x_c + depth * math.sin(th + ang)
        zw = z_c + depth * math.cos(th + ang)
        yw = cls.CAM_H - (cy - 0.5) * 2.0 * depth * math.tan(cls.FOV_V)
        return [xw, yw, zw]


class PoseEstimator:
    """视觉定位：输出相机位姿。

    真实后端：TODO 替换为 ORB-SLAM3 / VINS-Mono；pose_drift_std > 0 表示关闭回环校正。
    """

    def __init__(self, cfg, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self._drift = [0.0, 0.0]

    def estimate(self, gt_pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
        cfg = self.cfg
        if cfg.pose_drift_std > 0:                       # 累积漂移（无回环）
            self._drift[0] += self.rng.gauss(0, cfg.pose_drift_std)
            self._drift[1] += self.rng.gauss(0, cfg.pose_drift_std)
        return (
            gt_pose[0] + self.rng.gauss(0, cfg.pose_noise_std) + self._drift[0],
            gt_pose[1] + self.rng.gauss(0, cfg.pose_noise_std) + self._drift[1],
            gt_pose[2] + self.rng.gauss(0, 0.01),
        )


class AppearanceEncoder:
    """外观特征编码器（用于跨视角重识别）。

    真实后端：TODO 替换为 CLIP / DINOv2 图像编码器，输出 512/768 维特征后 L2 归一化。
    """

    @staticmethod
    def observe(proto: Sequence[float], cfg, rng: random.Random) -> List[float]:
        noise = [rng.gauss(0, cfg.feat_noise_std) for _ in proto]
        return l2_normalize([p + n for p, n in zip(proto, noise)])


class Reidentifier:
    """跨视角/跨时段的物体重识别：外观证据 + 几何证据的分级判定。

    融合相似度（用于排序与代价）：
        sim(o, m) = (w_app * cos(f_o, f_m) + w_geo * exp(-d^2 / (2 sigma^2))) / (w_app + w_geo)

    分级判定（解决"物体被移动后几何证据失效"这一核心难点）：
        match          外观强匹配（s_app >= app_hi） → 认定为同一物体，允许位置变化
        pending        外观中等 + 位置显著变化      → 转用户确认（低置信不自动改写记忆）
        match          外观中等 + 位置一致          → 认定为同一物体
        reject         两项证据均不足               → 视为未见过的新物体
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # ---------------- 基础相似度 ----------------
    def similarity(self, obs: Observation, node) -> float:
        from .utils import cosine, dist3
        s_app = max(0.0, cosine(obs.feat, node.feat))
        d = dist3(obs.pos_world, node.pos)
        s_geo = math.exp(-(d ** 2) / (2 * self.cfg.geo_sigma ** 2))
        w_sum = self.cfg.w_app + self.cfg.w_geo or 1.0
        sim = (self.cfg.w_app * s_app + self.cfg.w_geo * s_geo) / w_sum
        if obs.cls_pred != node.cls:      # 类别不一致 → 降权
            sim *= 0.55
        return sim

    def parts(self, obs: Observation, node):
        """返回 (s_app, s_geo, d, fused)。"""
        from .utils import cosine, dist3
        s_app = max(0.0, cosine(obs.feat, node.feat))
        d = dist3(obs.pos_world, node.pos)
        s_geo = math.exp(-(d ** 2) / (2 * self.cfg.geo_sigma ** 2))
        w_sum = self.cfg.w_app + self.cfg.w_geo or 1.0
        fused = (self.cfg.w_app * s_app + self.cfg.w_geo * s_geo) / w_sum
        if obs.cls_pred != node.cls:
            s_app *= 0.85          # 类别偶发错判：降权但不直接否定（配合几何证据与类别投票）
        return s_app, s_geo, d, fused

    # ---------------- 分级判定 ----------------
    def identify(self, obs: Observation, node):
        """返回 (verdict, score)。verdict ∈ {"match", "pending", "reject"}。"""
        cfg = self.cfg
        s_app, s_geo, d, fused = self.parts(obs, node)
        use_app = cfg.w_app > 0
        use_geo = cfg.w_geo > 0

        if use_app and s_app >= cfg.app_hi:
            return "match", max(fused, 0.95)                # 外观强证据：允许位置变化
        if use_app and d > cfg.move_thresh and s_app >= cfg.app_lo:
            return "pending", fused                          # 中等证据 + 位置突变 → 人工裁决
        if use_app and s_app >= cfg.app_mid and (not use_geo or s_geo >= cfg.geo_thresh):
            return "match", fused
        if not use_app and use_geo and s_geo >= cfg.geo_thresh:
            return "match", fused                            # w/o 外观：仅几何证据
        return "reject", fused
