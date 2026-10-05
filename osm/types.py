"""核心数据结构。"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Observation:
    """感知层输出的一次物体观测（检测 + 深度 + 位姿反投影后的结果）。"""
    frame_id: int
    timestamp: float
    cls_pred: str            # 预测类别
    cls_score: float         # 类别置信度
    bbox: tuple              # (x1, y1, x2, y2)，归一化坐标
    depth: float             # 估计深度（m）
    cam_pose: tuple          # (x, y, theta)
    feat: List[float]        # 外观特征向量
    pos_world: List[float]   # 反投影得到的世界坐标 (x, y, z)
    gt_id: Optional[int] = None        # 仿真数据中的真值物体编号（仅评测使用）
    gt_bbox: Optional[tuple] = None    # 真值框（仅评测使用）
    gt_cls: Optional[str] = None       # 真值类别（仅评测使用）


@dataclass
class ObjectNode:
    """记忆中的一个物体级节点（持久物体实例）。"""
    oid: int
    cls: str
    pos: List[float]                       # 融合后的世界坐标
    feat: List[float]                      # 原型外观特征
    last_seen: int = -1
    first_seen: int = -1
    obs_count: int = 0
    conf: float = 0.0                      # 该节点的可信度（观测次数/相似度累积）
    history: List[List[float]] = field(default_factory=list)   # 位置历史
    state: str = "active"                  # active / missing / removed
    move_streak: int = 0                   # 连续检测到位移的帧数（变化确认用）
    user_label: Optional[str] = None       # 用户自定义名称（人机协同标注）
    notes: List[str] = field(default_factory=list)
    last_seen_timestamp: Optional[float] = None  # 输入源时间；与运行速度无关


@dataclass
class Event:
    """记忆层事件（用于人机协同与变化检测评测）。"""
    kind: str          # new_object / moved / disappeared / pending_confirm
    oid: int
    frame_id: int
    detail: Dict = field(default_factory=dict)
