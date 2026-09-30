"""全局配置与超参数。

所有阈值集中在此，便于消融实验按名覆盖（见 experiments/run_all.py）。
"""

from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class Config:
    # ---------------- 随机性 ----------------
    seed: int = 20260924

    # ---------------- 感知层 ----------------
    det_score_thresh: float = 0.35      # 检测置信度阈值
    depth_noise_ratio: float = 0.06     # 单目深度相对噪声（占真值距离比例）
    pose_noise_std: float = 0.02        # 视觉定位平移噪声标准差（m）
    pose_drift_std: float = 0.0         # 无回环校正时的累积漂移（m/帧），消融项
    feat_dim: int = 64                  # 外观特征维度
    feat_noise_std: float = 0.015       # 外观特征观测噪声（相对归一化原型）

    # ---------------- 关联 / 重识别 ----------------
    w_app: float = 0.6                  # 外观相似度权重
    w_geo: float = 0.4                  # 几何（3D 位置）相似度权重
    match_thresh: float = 0.62          # 判定为同一物体的综合相似度阈值
    geo_sigma: float = 0.45             # 几何相似度的高斯尺度（m）
    max_gap_frames: int = 60            # 轨迹最长丢失帧数，超过则转"长期记忆"

    # 分级判定阈值（本文方法的关键：外观强证据可跨越位置变化，中等证据转人工确认）
    app_hi: float = 0.975               # 外观强匹配 → 直接认定为同一物体（允许位置变化）
    app_lo: float = 0.955               # 外观中等 + 位置显著变化 → 转待确认
    app_mid: float = 0.800              # 外观中等 + 位置一致 → 认定为同一物体
    geo_thresh: float = 0.70            # 几何相似度阈值（约对应 0.35 m 内）

    # ---------------- 记忆更新 ----------------
    pos_ema_alpha: float = 0.25         # 位置滑动平均更新系数
    move_thresh: float = 0.45           # 位置变化超过该值判定为"物体被移动"（m）
    move_confirm_frames: int = 2        # 位移需连续确认的帧数（抑制观测噪声引起的误报）
    auto_update_margin: float = 0.12    # 相似度高于 match_thresh + margin 才自动更新
    stale_frames: int = 240             # 超过该帧数未观测则标记为"可能已移除"

    # ---------------- 交互层 ----------------
    top_k: int = 3                      # 查询返回 Top-K
    query_pos_tol: float = 0.50         # 查询成功的位置容差（m）

    # ---------------- 仿真数据集 ----------------
    n_scenes: int = 12                  # 场景序列数
    frames_per_scene: int = 200         # 每序列帧数
    frame_interval_s: float = 3.0       # 帧间隔（s）→ 200 帧 = 10 min
    n_objects: int = 18                 # 每场景物体数
    room_size: float = 5.0              # 房间边长（m）
    move_events_per_scene: int = 3      # 每序列真实移动事件数
    queries_per_scene: int = 20         # 每序列查询数

    def override(self, **kwargs) -> "Config":
        c = Config(**asdict(self))
        for k, v in kwargs.items():
            if not hasattr(c, k):
                raise KeyError(f"未知配置项: {k}")
            setattr(c, k, v)
        return c

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


DEFAULT = Config()
