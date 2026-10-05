"""记忆层：物体级持久记忆的构建、更新与人机协同修正。

对应报告 3.3 节。核心机制：
    1) 帧内关联：匈牙利匹配把观测分配给"活跃轨迹"
    2) 跨时段重识别：未匹配观测回查全量记忆库，实现"再次看到即认出"
    3) 变化检测 + 置信度分流：高置信自动更新，低置信转"待确认事件"交用户裁决
    4) 用户反馈闭环：确认 / 纠正 / 重命名，写入记忆并影响后续匹配
"""

import logging
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .config import Config
from .perception import Reidentifier
from .types import Event, ObjectNode, Observation
from .utils import dist3, ema, hungarian

logger = logging.getLogger(__name__)


class ObjectMemory:
    """持久物体记忆库。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reid = Reidentifier(cfg)
        self.nodes: Dict[int, ObjectNode] = {}
        self.next_oid = 1
        self.events: List[Event] = []
        self.frame_assign: List[Tuple[int, int]] = []     # (gt_id, pred_oid) 评测用
        self.pending: Dict[int, dict] = {}                # 待确认事件 -> 暂存的更新
        self.n_auto_update = 0
        self.n_user_confirmed = 0
        self.n_user_rejected = 0
        self.n_user_initiated = 0        # 用户主动纠正次数
        self.n_reid_calls = 0
        self.n_reid_hit = 0

    # ------------------------------------------------------------------ 主更新
    def update(self, observations: List[Observation], frame_id: int) -> List[Event]:
        """处理一帧的所有观测，返回本帧产生的事件。"""
        cfg = self.cfg
        frame_events: List[Event] = []
        logger.debug("frame=%s observations=%s nodes=%s pending=%s",
                     frame_id, len(observations), len(self.nodes), len(self.pending))
        if logger.isEnabledFor(logging.DEBUG):
            for oi, obs in enumerate(observations):
                logger.debug("frame=%s obs=%s input_time=%.6f cls=%s score=%.3f bbox=%s pos=%s",
                             frame_id, oi, obs.timestamp, obs.cls_pred, obs.cls_score,
                             obs.bbox, obs.pos_world)
        if not observations:
            self._mark_missing(frame_id)
            return frame_events

        active = [n for n in self.nodes.values()
                  if n.state != "removed" and frame_id - n.last_seen <= cfg.max_gap_frames]
        # ---------- 1) 帧内关联（观测 ↔ 活跃轨迹） ----------
        matches, unmatched_obs = self._associate(observations, active)

        for oi, node, verdict, sim in matches:
            obs = observations[oi]
            moved = self._check_move(node, obs.pos_world)
            if verdict == "pending":
                if not moved:
                    # 位移尚未被连续确认：只刷新外观与可见性，不改写位置（保守更新）
                    self._apply_update(node, obs, frame_id, update_pos=False)
                    continue
                # 中等证据 + 位置突变：挂起等待用户确认（同一节点同一时刻只问一次）
                if any(p["oid"] == node.oid for p in self.pending.values()):
                    continue
                key = len(self.pending) + 1
                self.pending[key] = {
                    "kind": "moved", "oid": node.oid, "frame_id": frame_id,
                    "new_pos": list(obs.pos_world), "new_feat": list(obs.feat),
                    "sim": sim, "gt_id": obs.gt_id,
                    "timestamp": obs.timestamp,
                }
                frame_events.append(Event("pending_confirm", node.oid, frame_id,
                                          {"reason": "position_jump", "sim": round(sim, 3)}))
                continue
            if moved:
                frame_events.append(Event("moved", node.oid, frame_id,
                                          {"from": [round(v, 3) for v in node.pos],
                                           "to": [round(v, 3) for v in obs.pos_world]}))
            self._apply_update(node, obs, frame_id)
            self._record(node, obs)

        # ---------- 2) 跨时段重识别（未匹配观测 ↔ 全量记忆） ----------
        still_unmatched = []
        for oi in unmatched_obs:
            obs = observations[oi]
            self.n_reid_calls += 1
            best, best_verdict, best_sim = None, "reject", 0.0
            for node in self.nodes.values():
                if node.state == "removed":
                    continue
                verdict, sim = self.reid.identify(obs, node)
                if verdict == "match" and sim > best_sim:
                    best, best_verdict, best_sim = node, verdict, sim
            if best is not None:
                self.n_reid_hit += 1
                if self._check_move(best, obs.pos_world):
                    frame_events.append(Event("moved", best.oid, frame_id,
                                              {"from": [round(v, 3) for v in best.pos],
                                               "to": [round(v, 3) for v in obs.pos_world]}))
                self._apply_update(best, obs, frame_id)
                self._record(best, obs)
            else:
                still_unmatched.append(oi)

        # ---------- 3) 新物体建库 ----------
        for oi in still_unmatched:
            obs = observations[oi]
            node = ObjectNode(
                oid=self.next_oid, cls=obs.cls_pred, pos=list(obs.pos_world),
                feat=list(obs.feat), last_seen=frame_id, first_seen=frame_id,
                obs_count=1, conf=obs.cls_score, history=[list(obs.pos_world)],
                last_seen_timestamp=obs.timestamp,
            )
            self.nodes[self.next_oid] = node
            self.next_oid += 1
            self._record(node, obs)
            frame_events.append(Event("new_object", node.oid, frame_id,
                                      {"score": round(obs.cls_score, 3)}))

        self._mark_missing(frame_id)
        self.events.extend(frame_events)
        for event in frame_events:
            logger.info("frame=%s %s oid=%s detail=%s",
                        event.frame_id, event.kind, event.oid, event.detail)
        return frame_events

    # ------------------------------------------------------------------ 关联
    def _associate(self, observations: List[Observation], nodes: List[ObjectNode]):
        """匈牙利最小代价匹配。

        代价由分级判定给出：被判为 reject 的候选额外惩罚（×0.5），
        使匈牙利优先把观测分配给"证据充分"的记忆节点。

        Returns:
            (pairs, unmatched_obs)；pairs 元素为 (obs_idx, node, verdict, score)
        """
        if not nodes:
            return [], list(range(len(observations)))
        verdicts = [[self.reid.identify(o, n) for n in nodes] for o in observations]
        cost = [[1.0 - (s if v != "reject" else s * 0.5) for v, s in row] for row in verdicts]

        if len(observations) <= len(nodes):
            assign = hungarian(cost)
            raw = list(enumerate(assign))
        else:
            t = hungarian([[cost[i][j] for i in range(len(observations))]
                           for j in range(len(nodes))])
            taken = [-1] * len(observations)
            for ni, oi in enumerate(t):
                if oi >= 0:
                    taken[oi] = ni
            raw = list(enumerate(taken))

        pairs, unmatched = [], []
        for oi, ni in raw:
            if ni < 0:
                unmatched.append(oi)
                continue
            verdict, score = verdicts[oi][ni]
            if verdict == "reject":
                unmatched.append(oi)
            else:
                pairs.append((oi, nodes[ni], verdict, score))
        return pairs, unmatched

    def _check_move(self, node: ObjectNode, pos: List[float]) -> bool:
        """带连续帧确认的位移判定：抑制单帧观测噪声引起的"假变化"。"""
        d = dist3(pos, node.pos)
        if d > self.cfg.move_thresh:
            node.move_streak = getattr(node, "move_streak", 0) + 1
        else:
            node.move_streak = 0
        return node.move_streak >= self.cfg.move_confirm_frames

    def _apply_update(self, node: ObjectNode, obs: Observation, frame_id: int,
                      update_pos: bool = True):
        cfg = self.cfg
        if update_pos:
            node.pos = ema(node.pos, obs.pos_world, cfg.pos_ema_alpha)
        node.feat = ema(node.feat, obs.feat, 0.2)
        node.feat = [x / (math.sqrt(sum(v * v for v in node.feat)) or 1e-12) for x in node.feat]
        node.last_seen = frame_id
        node.last_seen_timestamp = obs.timestamp
        node.obs_count += 1
        node.conf = min(1.0, node.conf + 0.05 * (1.0 - node.conf))
        node.history.append(list(node.pos))
        if len(node.history) > 50:
            node.history.pop(0)
        node.state = "active"
        # 类别投票修正：抑制单帧分类抖动（用户纠正后锁定类别，不再自动改写）
        if not getattr(node, "cls_locked", False):
            if not hasattr(node, "cls_votes"):
                node.cls_votes = Counter()
            node.cls_votes[obs.cls_pred] += obs.cls_score
            top, cnt = node.cls_votes.most_common(1)[0]
            if cnt >= 3 and cnt > sum(node.cls_votes.values()) * 0.5:
                node.cls = top      # 票数足够且占优才改写类别
        self.n_auto_update += 1

    def _record(self, node: ObjectNode, obs: Observation):
        """记录 gt_id 投票与帧级分配（仅评测使用）。"""
        if obs.gt_id is None:
            return
        if not hasattr(node, "gt_votes"):
            node.gt_votes = Counter()
        node.gt_votes[obs.gt_id] += 1
        self.frame_assign.append((obs.gt_id, node.oid))

    def _mark_missing(self, frame_id: int):
        for node in self.nodes.values():
            if node.state == "removed":
                continue
            gap = frame_id - node.last_seen
            if gap > self.cfg.stale_frames:
                node.state = "removed"
                self.events.append(Event("disappeared", node.oid, frame_id, {"gap": gap}))
                logger.info("frame=%s disappeared oid=%s gap=%s", frame_id, node.oid, gap)
            elif gap > self.cfg.max_gap_frames and node.state == "active":
                node.state = "missing"
                logger.info("frame=%s missing oid=%s gap=%s", frame_id, node.oid, gap)

    # ------------------------------------------------------------------ 人机协同
    def resolve_pending(self, key: int, accept: bool, correction: Optional[dict] = None):
        """用户对某条待确认事件的反馈。

        Args:
            key: 待确认事件编号
            accept: 是否接受系统提出的变更
            correction: 可选的用户纠正内容，如 {"pos": [...]} / {"cls": "..."} / {"name": "..."}
        """
        item = self.pending.pop(key, None)
        if item is None:
            return
        node = self.nodes.get(item["oid"])
        if node is None:
            return
        if accept:
            node.pos = correction.get("pos", item["new_pos"]) if correction else item["new_pos"]
            node.feat = item["new_feat"]
            node.last_seen = item["frame_id"]
            node.last_seen_timestamp = item.get("timestamp")
            node.history.append(list(node.pos))
            if item.get("gt_id") is not None:
                self.frame_assign.append((item["gt_id"], node.oid))
                if not hasattr(node, "gt_votes"):
                    node.gt_votes = Counter()
                node.gt_votes[item["gt_id"]] += 1
            self.n_user_confirmed += 1
        else:
            node.conf = max(0.1, node.conf - 0.15)      # 拒绝 → 降低该节点可信度
            self.n_user_rejected += 1
        logger.info("pending=%s oid=%s accepted=%s", key, node.oid, accept)
        if correction:
            if "cls" in correction:
                node.cls = correction["cls"]
                node.cls_locked = True      # 用户纠正优先，锁定类别
            if "name" in correction:
                node.user_label = correction["name"]
                node.notes.append(f"用户命名：{correction['name']}")

    def user_correct_position(self, oid: int, pos: List[float], frame_id: int = -1,
                              timestamp: Optional[float] = None):
        """用户主动纠正：直接改写物体位置（例如"我把杯子挪到书桌上了"）。"""
        node = self.nodes.get(oid)
        if node is None:
            return False
        node.pos = list(pos)
        node.history.append(list(node.pos))
        node.move_streak = 0
        if frame_id >= node.last_seen:
            node.last_seen_timestamp = timestamp
        node.last_seen = max(node.last_seen, frame_id)
        node.conf = min(1.0, node.conf + 0.10)
        node.notes.append(f"用户纠正位置 @frame {frame_id}")
        self.n_user_initiated += 1
        logger.info("frame=%s user_correct oid=%s pos=%s", frame_id, oid, pos)
        return True

    def rename(self, oid: int, name: str):
        node = self.nodes.get(oid)
        if node:
            node.user_label = name
            node.notes.append(f"用户命名：{name}")
            logger.info("rename oid=%s name=%s", oid, name)

    # ------------------------------------------------------------------ 查询接口
    def all_nodes(self) -> List[ObjectNode]:
        return [n for n in self.nodes.values() if n.state != "removed"]

    def snapshot_table(self) -> str:
        """以文本表格形式导出当前记忆库（用于演示与调试）。"""
        rows = sorted(self.all_nodes(), key=lambda n: n.oid)
        lines = ["oid | 名称/类别           | 位置 (x, z) m      | 置信度 | 最近观测帧 | 状态",
                 "----+---------------------+--------------------+--------+------------+------"]
        for n in rows:
            name = (n.user_label or n.cls)[:12]
            lines.append(f"{n.oid:>3} | {name:<19} | ({n.pos[0]:+.2f}, {n.pos[2]:+.2f})     "
                         f"| {n.conf:.2f}   | {n.last_seen:>10} | {n.state}")
        if not rows:
            lines.append("（记忆库为空）")
        return "\n".join(lines)

    def gt_major(self, node: ObjectNode) -> Optional[int]:
        votes = getattr(node, "gt_votes", None)
        return votes.most_common(1)[0][0] if votes else None
