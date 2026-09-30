"""评价指标计算。

对应报告 4.1 节。指标定义：
    检测 mAP@0.5      ：按类别平均精度（仿真数据上用类别正确率+框 IoU 近似）
    IDF1              ：多目标跟踪身份保持度
    重识别准确率       ：跨时段再次出现的物体被正确关联到原记忆节点的比例
    定位中位误差       ：记忆中物体位置与真值位置的欧氏距离中位数（m）
    记忆更新一致率     ：与真值一致的记忆节点数 / 记忆节点总数
    待确认事件率       ：每 10 分钟需要用户裁决的事件数（越低越好）
    查询成功率 Top-3   ：目标物体出现在前 3 个结果且位置误差在容差内的比例
    变化检测 F1        ：物体移动/增减事件的检测精度与召回的调和平均
    NASA-TLX          ：主观工作负荷（本原型用交互日志拟合的估算模型）
"""

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from .utils import dist3, mean, median


# ---------------------------------------------------------------- IDF1
def idf1(assignments: Sequence[Tuple[int, int]]) -> float:
    """计算 IDF1。

    Args:
        assignments: [(gt_id, pred_id), ...] 每帧每个观测的身份分配
    """
    if not assignments:
        return float("nan")
    gt_total = Counter(g for g, _ in assignments)
    pr_total = Counter(p for _, p in assignments)
    pair = Counter(assignments)

    # 双向最大投票匹配：gt→最常配的 pred，pred→最常配的 gt，一致则确认为同一身份
    gt2pr = {g: pair[(g, p)] for g, p in pair}
    best_for_gt, best_for_pr = {}, {}
    for (g, p), c in pair.items():
        if c > best_for_gt.get(g, (0, None))[0]:
            best_for_gt[g] = (c, p)
        if c > best_for_pr.get(p, (0, None))[0]:
            best_for_pr[p] = (c, g)
    idtp = 0
    for g, (c, p) in best_for_gt.items():
        if best_for_pr.get(p, (0, None))[1] == g:
            idtp += c
    idfp = sum(pr_total.values()) - idtp
    idfn = sum(gt_total.values()) - idtp
    return 2 * idtp / max(1e-12, (2 * idtp + idfp + idfn))


# ---------------------------------------------------------------- 重识别 / 定位
def reid_accuracy(memory) -> float:
    """跨时段重识别准确率：回查记忆库命中的观测中，关联到正确节点的比例。"""
    ok = tot = 0
    for node in memory.nodes.values():
        votes = getattr(node, "gt_votes", None)
        if not votes:
            continue
        major, cnt = votes.most_common(1)[0]
        tot += sum(votes.values())
        ok += cnt
    # 每个 gt 只应被一个节点"主关联"占有，重复占用视为错误
    owner: Dict[int, int] = {}
    dup = 0
    for node in memory.nodes.values():
        votes = getattr(node, "gt_votes", None)
        if not votes:
            continue
        major, cnt = votes.most_common(1)[0]
        if major in owner:
            dup += sum(votes.values())
        else:
            owner[major] = node.oid
    return (ok - dup) / tot if tot else float("nan")


def localization_errors(memory, gt_positions: Dict[int, List[float]]) -> List[float]:
    """每个记忆节点与其主关联真值物体之间的位置误差。"""
    errs = []
    for node in memory.nodes.values():
        if node.state == "removed":
            continue
        votes = getattr(node, "gt_votes", None)
        if not votes:
            continue
        major = votes.most_common(1)[0][0]
        if major in gt_positions:
            errs.append(dist3(node.pos, gt_positions[major]))
    return errs


def memory_consistency(memory, gt_positions: Dict[int, List[float]],
                       gt_classes: Dict[int, str], tol: float = 0.35) -> float:
    """记忆更新一致率：位置误差 < tol 且类别正确的节点占比。"""
    tot = ok = 0
    for node in memory.nodes.values():
        if node.state == "removed":
            continue
        votes = getattr(node, "gt_votes", None)
        if not votes:
            continue
        tot += 1
        major = votes.most_common(1)[0][0]
        pos_ok = major in gt_positions and dist3(node.pos, gt_positions[major]) < tol
        cls_ok = gt_classes.get(major) == node.cls
        if pos_ok and cls_ok:
            ok += 1
    return ok / tot if tot else float("nan")


# ---------------------------------------------------------------- 变化检测
def change_detection_f1(detected: Sequence[Tuple[int, int]],
                        gt_events: Sequence[Tuple[int, int]]) -> Tuple[float, float, float]:
    """按物体 ID 匹配的变化检测 precision / recall / F1。

    Args:
        detected:  [(gt_id, frame_id), ...] 系统报告的移动事件
        gt_events: [(gt_id, frame_id), ...] 真值移动事件
    """
    det_ids = {g for g, _ in detected}
    gt_ids = {g for g, _ in gt_events}
    tp = len(det_ids & gt_ids)
    fp = len(det_ids - gt_ids)
    fn = len(gt_ids - det_ids)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


# ---------------------------------------------------------------- 主观负荷
def nasa_tlx(pending_per_10min: float, corrections_per_10min: float) -> float:
    """NASA-TLX 估算（0–100，越低越好）。

    说明：真实实验中由被试填写量表；本原型用交互日志的线性模型拟合，
    用于消融对比的相对比较，不作为绝对结论。
    """
    score = 24.0 + 2.6 * pending_per_10min + 3.1 * corrections_per_10min
    return max(0.0, min(100.0, score))
