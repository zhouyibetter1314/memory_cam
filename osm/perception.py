"""感知层：物体与空间信息提取。

对应报告 3.2 节。四类能力：
    1) 物体检测与分类      Detector
    2) 单目深度估计 + 反投影 DepthEstimator
    3) 视觉定位（位姿）     PoseEstimator
    4) 外观编码/重识别      AppearanceEncoder

Detector 用于仿真复现；ImagePerception 统一封装真实图像的 YOLO / motion 检测。
两者均输出 Observation，供同一条记忆管线消费。
"""

import logging
import math
import random
from typing import List, Sequence, Tuple

from .labels import canonical_class
from .types import Observation
from .utils import l2_normalize

logger = logging.getLogger(__name__)


class Detector:
    """物体检测与分类。

    仿真后端：对视场内可见物体按 det_score 采样，输出带噪声的 bbox 与类别。
    真实图像后端见 ImagePerception.predict(image, frame_id=..., timestamp=...)。
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


class YoloError(RuntimeError):
    """YOLO 配置、加载或推理错误。"""


class ImagePerception:
    """真实图像感知：YOLO / motion → HSV 特征、近似平面位置和 Observation。

    不读取摄像头或视频，也不创建窗口；输入为 OpenCV BGR 图像。
    YOLO 检测静止物体；motion 保留背景预热与 45 帧静止吸收。
    第三方依赖只在实例化真实图像后端时导入。
    """

    def __init__(self, cfg, *, detector="yolo", yolo_model="yolo26n.pt",
                 yolo_conf=0.35, yolo_imgsz=640, yolo_device=None, yolo_classes=None):
        if detector not in ("yolo", "motion"):
            raise ValueError(f"未知检测器：{detector}")
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("图像感知需要 pip install -r requirements.txt") from exc
        self.cv2, self.np = cv2, np
        self.cfg = cfg
        self.detector = detector
        self.model = None
        self.predict_options = dict(conf=yolo_conf, imgsz=yolo_imgsz,
                                    classes=yolo_classes, verbose=False, save=False)
        if yolo_device is not None:
            self.predict_options["device"] = yolo_device
        self.pixels_per_meter = 260.0  # 固定视角下的粗略比例，可由调用方调节。
        self.bg = None
        self.bg_alpha = 0.02
        self.min_area = 1800
        self.still_absorb = 45
        self._still = {}
        if detector == "yolo":
            self._load_yolo(yolo_model)

    def predict(self, image, *, frame_id: int, timestamp: float) -> List[Observation]:
        """处理单帧，保留调用方提供的帧编号与源时间戳。"""
        observations = (self._detect_yolo(image) if self.detector == "yolo"
                        else self._detect_motion(image))
        for obs in observations:
            obs.frame_id, obs.timestamp = frame_id, timestamp
        return observations

    def _load_yolo(self, weights):
        try:
            from ultralytics import YOLO
            conf, imgsz = self.predict_options["conf"], self.predict_options["imgsz"]
            if not math.isfinite(conf) or not 0 <= conf <= 1:
                raise ValueError("YOLO 置信度必须为 [0,1] 内的有限数值")
            if not isinstance(imgsz, int) or imgsz <= 0:
                raise ValueError("YOLO 图像尺寸必须为正整数")
            self.model = YOLO(weights)
            if self.model.task != "detect":
                raise ValueError("请使用目标检测权重（例如 yolo26n.pt）")
            classes = self.predict_options["classes"]
            names = self.model.names
            valid_ids = set(names) if isinstance(names, dict) else set(range(len(names)))
            if classes is not None and any(c not in valid_ids for c in classes):
                raise ValueError(f"类别编号必须属于模型类别：{sorted(valid_ids)}")
        except Exception as exc:
            raise YoloError(
                f"YOLO 初始化失败：{exc}。请安装 pip install -r requirements.txt，"
                "检查权重与参数；或使用 --detector motion") from exc
        logger.info("YOLO 检测器：model=%s options=%s", weights, self.predict_options)

    # ------------------------------------------------------------------
    def _signature(self, bbox) -> int:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 40) * 1000 + int((y1 + y2) / 40))

    def _hist(self, hsv, mask, bbox) -> list:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi_h = hsv[y1:y2, x1:x2]
        roi_m = mask[y1:y2, x1:x2] if mask is not None else None
        hist = self.cv2.calcHist([roi_h], [0, 1, 2], roi_m,
                                [12, 4, 2], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten().astype(float)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist.tolist()

    def _observation(self, hsv, bbox, cls_pred, score, mask=None):
        h, w = hsv.shape[:2]
        x1, y1, x2, y2 = bbox
        cx_m = ((x1 + x2) / 2 - w / 2) / self.pixels_per_meter
        cy_m = ((y1 + y2) / 2 - h / 2) / self.pixels_per_meter
        height_m = (y2 - y1) / self.pixels_per_meter
        return Observation(
            frame_id=-1, timestamp=0.0, cls_pred=cls_pred, cls_score=score,
            bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
            depth=1.0, cam_pose=(0.0, 0.0, 0.0),
            feat=self._hist(hsv, mask, bbox),
            pos_world=[round(cx_m, 3), round(height_m, 3), round(cy_m, 3)],
        )

    def _detect_yolo(self, frame):
        try:
            results = self.model.predict(source=frame, **self.predict_options)
            if not results or results[0].boxes is None:
                return []
            result = results[0]
            boxes = result.boxes.cpu()
            hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
            h, w = frame.shape[:2]
            observations = []
            for coords, score, cls_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                            boxes.cls.tolist()):
                if (not all(math.isfinite(v) for v in (*coords, score, cls_id))
                        or not 0 <= score <= 1 or score < self.predict_options["conf"]):
                    continue
                cls_id = int(cls_id)
                classes = self.predict_options["classes"]
                if classes is not None and cls_id not in classes:
                    continue
                x1, y1, x2, y2 = coords
                x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
                y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
                if x2 <= x1 or y2 <= y1 or int(x2) <= int(x1) or int(y2) <= int(y1):
                    continue
                name = canonical_class(result.names[cls_id])
                observations.append(self._observation(hsv, (x1, y1, x2, y2), name, score))
            return observations
        except Exception as exc:
            raise YoloError(
                f"YOLO 推理失败：{exc}。请检查权重与 --yolo-device；"
                "或使用 --detector motion") from exc

    def _detect_motion(self, frame):
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY).astype("float32")
        if self.bg is None:
            self.bg = gray
            return []
        diff = self.cv2.absdiff(gray, self.bg)
        _, mask = self.cv2.threshold(diff, 28, 255, self.cv2.THRESH_BINARY)
        mask = mask.astype("uint8")
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, self.np.ones((5, 5), self.np.uint8))
        mask = self.cv2.dilate(mask, self.np.ones((9, 9), self.np.uint8), iterations=2)

        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL,
                                           self.cv2.CHAIN_APPROX_SIMPLE)
        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        obs_list, new_still = [], {}
        for c in contours:
            area = self.cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, bw, bh = self.cv2.boundingRect(c)
            sig = self._signature((x, y, x + bw, y + bh))
            still = self._still.get(sig, 0) + 1
            new_still[sig] = still
            if still > self.still_absorb:
                continue                   # 静止物体已并入背景，不再重复上报
            obs_list.append(self._observation(
                hsv, (x, y, x + bw, y + bh), "未知物体",
                min(1.0, area / 12000.0), mask))
        self._still = new_still
        # 背景缓慢更新（被前景区域遮盖的部分由静止吸收机制处理）
        self.cv2.accumulateWeighted(gray, self.bg, self.bg_alpha)
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
            verdict, score = "match", max(fused, 0.95)
        elif use_app and d > cfg.move_thresh and s_app >= cfg.app_lo:
            verdict, score = "pending", fused
        elif use_app and s_app >= cfg.app_mid and (not use_geo or s_geo >= cfg.geo_thresh):
            verdict, score = "match", fused
        elif not use_app and use_geo and s_geo >= cfg.geo_thresh:
            verdict, score = "match", fused
        else:
            verdict, score = "reject", fused
        logger.debug("frame=%s oid=%s verdict=%s app=%.4f geo=%.4f distance=%.4f score=%.4f",
                     obs.frame_id, node.oid, verdict, s_app, s_geo, d, score)
        return verdict, score
