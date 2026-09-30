#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""webcam_demo.py —— 面向日常动态场景的物体级空间记忆与场景理解：实时演示

支持两种感知后端：
  1) real 模式：电脑内置摄像头或外接 USB 摄像头（需 opencv-python + numpy，
     安装在隔离环境 ~/.workbuddy/binaries/python/envs/default 中）。
     采用「帧差 + 背景建模」检测前景物体，HSV 颜色直方图作为外观特征，
     接入与本课题一致的 ObjectMemory 物体级空间记忆管线。
  2) sim 模式（默认回退）：没有摄像头 / 没有 opencv 时自动切换，
     用 Sim-Scene 合成动态场景驱动同一条记忆管线，保证任何机器都能演示。

交互方式（real 模式）：
  画面窗口按键
    q / Esc   退出
    空格      暂停 / 继续
    n         给当前选中物体命名（在终端输入名称，如"我的水杯"）
    c         裁决当前待确认事件（y=接受位置变更 / n=拒绝）
    m         主动纠正：告知某物体当前位置（点击画面后按 m，再在终端输入名称）
    f         发起自然语言查询（在终端输入问句，如"我的水杯在哪里？"）
    +/-       放大 / 缩小像素—米比例（用于粗略校准空间尺度）
    s         打印记忆库快照到终端
  鼠标左键点击画面：选中最近的记忆物体（用于命名 / 纠正）

运行示例：
  python demo/webcam_demo.py                      # 自动检测摄像头，失败回退 sim
  python demo/webcam_demo.py --camera 1           # 指定外接摄像头
  python demo/webcam_demo.py --mode sim --scene 3 # 仿真模式演示
  python demo/webcam_demo.py --headless           # 不弹窗，仅终端输出（服务器/录屏）

依赖说明：real 模式需要 cv2 与 numpy；其余全部复用 code/osm 纯标准库实现。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# 允许从 code/ 目录以外的地方运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osm.config import Config, DEFAULT
from osm.memory import ObjectMemory
from osm.interaction import QueryEngine, AnswerGenerator
from osm.types import Observation

try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:                       # 无 opencv → 只能 sim 模式
    cv2 = None
    np = None
    HAS_CV = False


# =====================================================================
# 真实摄像头感知后端
# =====================================================================

class WebcamPerception:
    """帧差 + 背景建模的前景检测器，输出 ObjectMemory 可消费的 Observation。

    设计要点：
      - 背景模型：运行期指数滑动平均（与 MOG2 思路一致，但实现透明、依赖少）。
      - 外观特征：前景区域内的 HSV 直方图（H 12 × S 4 × V 2 = 96 维，L1 归一化），
        颜色对光照敏感度低于 RGB，且计算量极小。
      - 位置：把像素坐标按"像素—米比例"映射为以画面中心为原点的桌面平面坐标，
        深度固定为 1（桌面场景近似），高度由前景像素高度粗略估计。
      - 静止吸收：前景 blob 若在原地保持 still_frames 帧以上，被并入背景，
        这样"放下的物体"会成为背景的一部分，再次移动时才重新触发——
        恰好对应日常场景"物体大多数时候静止"的先验。
    """

    def __init__(self, cfg: Config, camera: int = 0, width: int = 960, height: int = 540):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera}（可用 --camera 1 尝试外接摄像头）")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.bg = None                     # 背景模型（float32）
        self.bg_alpha = 0.02               # 背景更新速率
        self.pixels_per_meter = 260.0      # 粗略标定：按键 +/- 可调
        self.min_area = 1800               # 最小前景面积（像素²），过滤噪声
        self.still_absorb = 45             # 静止多少帧后并入背景
        self._still = {}                   # blob 签名 -> 已静止帧数

    # ------------------------------------------------------------------
    def _signature(self, bbox) -> int:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 40) * 1000 + int((y1 + y2) / 40))

    def _hist(self, hsv, mask, bbox) -> list:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi_h = hsv[y1:y2, x1:x2]
        roi_m = mask[y1:y2, x1:x2]
        hist = cv2.calcHist([roi_h], [0, 1, 2], roi_m,
                            [12, 4, 2], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten().astype(float)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist.tolist()

    # ------------------------------------------------------------------
    def read(self):
        """返回 (frame, observations)。observations 可能为空列表。"""
        ok, frame = self.cap.read()
        if not ok:
            return None, []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
        if self.bg is None:
            self.bg = gray
            return frame, []
        diff = cv2.absdiff(gray, self.bg)
        _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        mask = mask.astype("uint8")
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = gray.shape
        obs_list, new_still = [], {}
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            sig = self._signature((x, y, x + bw, y + bh))
            still = self._still.get(sig, 0) + 1
            new_still[sig] = still
            if still > self.still_absorb:
                continue                   # 静止物体已并入背景，不再重复上报
            # 像素 → 以画面中心为原点的平面坐标（米）
            cx_m = ((x + bw / 2) - w / 2) / self.pixels_per_meter
            cy_m = ((y + bh / 2) - h / 2) / self.pixels_per_meter
            height_m = round(bh / self.pixels_per_meter, 3)
            feat = self._hist(hsv, mask, (x, y, x + bw, y + bh))
            obs_list.append(Observation(
                frame_id=-1, timestamp=time.time(),
                cls_pred="未知物体", cls_score=min(1.0, area / 12000.0),
                bbox=(x / w, y / h, (x + bw) / w, (y + bh) / h),
                depth=1.0, cam_pose=(0.0, 0.0, 0.0),
                feat=feat,
                pos_world=[round(cx_m, 3), round(height_m, 3), round(cy_m, 3)],
            ))
        self._still = new_still
        # 背景缓慢更新（被前景区域遮盖的部分由静止吸收机制处理）
        cv2.accumulateWeighted(gray, self.bg, self.bg_alpha)
        return frame, obs_list

    def release(self):
        self.cap.release()


# =====================================================================
# 演示主程序
# =====================================================================

HELP = """按键：空格 暂停 | n 命名 | c 裁决待确认 | m 主动纠正 | f 查询 | s 快照 | +/- 调比例 | q 退出"""


class DemoApp:
    def __init__(self, args):
        self.args = args
        self.cfg = DEFAULT.override(feat_dim=96)   # HSV 直方图维度
        self.mem = ObjectMemory(self.cfg)
        self.qe = QueryEngine(self.mem, self.cfg)
        self.ag = AnswerGenerator()
        self.frame_id = 0
        self.paused = False
        self.selected = None               # 鼠标选中的记忆节点 oid
        self.last_obs_boxes = []           # 最近一帧观测框（像素），供点击选择

    # ---------------- 终端交互（阻塞式，按一次处理一次） ----------------
    def _ask(self, prompt: str) -> str:
        try:
            return input(prompt).strip()
        except EOFError:
            return ""

    def do_name(self):
        oid = self._pick_oid("输入要命名的物体编号（回车取选中/最新）：")
        if oid is None:
            return
        name = self._ask("输入名称（如：我的水杯）：")
        if name:
            self.mem.rename(oid, name)
            print(f"[交互] 已将物体 {oid} 命名为「{name}」")

    def do_confirm(self):
        if not self.mem.pending:
            print("[交互] 当前没有待确认事件。")
            return
        key = sorted(self.mem.pending)[0]
        item = self.mem.pending[key]
        ans = self._ask(f"物体 {item['oid']} 疑似发生位置变化（sim={item['sim']:.2f}），"
                        f"确认移动？(y/n)：")
        if ans in ("y", "n"):
            self.mem.resolve_pending(key, accept=(ans == "y"))
            print(f"[交互] 已{'接受' if ans == 'y' else '拒绝'}该位置变更。")

    def do_correct(self):
        oid = self._pick_oid("输入要纠正的物体编号（回车取选中/最新）：")
        if oid is None:
            return
        txt = self._ask("输入新位置（格式 x,z 米，如 0.30,-0.15；回车取最近观测位置）：")
        node = self.mem.nodes.get(oid)
        if txt:
            try:
                x, z = [float(v) for v in txt.split(",")]
                pos = [x, node.pos[1] if node else 0.0, z]
            except ValueError:
                print("[交互] 格式无效。")
                return
        elif node is not None:
            pos = list(node.pos)
        else:
            return
        if self.mem.user_correct_position(oid, pos, self.frame_id):
            print(f"[交互] 已把物体 {oid} 的位置纠正为 {pos}")

    def do_query(self):
        text = self._ask("输入问句（如：我的水杯在哪里？）：")
        if not text:
            return
        hits = self.qe.retrieve(text)
        answer = self.ag.generate(text, hits, now_frame=self.frame_id, memory=self.mem)
        print(f"[问答] {answer}")
        for node, score in hits:
            print(f"    候选 #{node.oid} {node.user_label or node.cls} "
                  f"score={score:.2f} pos=({node.pos[0]:.2f},{node.pos[2]:.2f})")

    def do_snapshot(self):
        print(self.mem.snapshot_table())

    def _pick_oid(self, prompt: str):
        if self.selected is not None:
            return self.selected
        txt = self._ask(prompt)
        if txt:
            try:
                return int(txt)
            except ValueError:
                return None
        nodes = sorted(self.mem.all_nodes(), key=lambda n: n.last_seen, reverse=True)
        return nodes[0].oid if nodes else None

    # ---------------- 鼠标回调 ----------------
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        best, best_d = None, 1e9
        for oid, (x1, y1, x2, y2) in self.last_obs_boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_d:
                best, best_d = oid, d
        if best is not None:
            self.selected = best
            print(f"[交互] 已选中物体 {best}")

    # ---------------- 事件处理 ----------------
    def handle_events(self, events):
        for ev in events:
            if ev.kind == "new_object":
                print(f"[记忆] 新物体入库 #{ev.oid}（第 {ev.frame_id} 帧）")
            elif ev.kind == "moved":
                print(f"[记忆] 物体 #{ev.oid} 位置更新 → {ev.detail.get('to')}")
            elif ev.kind == "pending_confirm":
                print(f"[记忆] 物体 #{ev.oid} 疑似移动（低置信），按 c 裁决")
            elif ev.kind == "disappeared":
                print(f"[记忆] 物体 #{ev.oid} 离开视野")

    # ---------------- 画面绘制 ----------------
    def draw(self, frame, obs_list, matched):
        if np is None:
            return frame
        vis = frame.copy()
        h, w = vis.shape[:2]
        self.last_obs_boxes = []
        for obs in obs_list:
            x1, y1, x2, y2 = [int(obs.bbox[0] * w), int(obs.bbox[1] * h),
                              int(obs.bbox[2] * w), int(obs.bbox[3] * h)]
            oid = matched.get(id(obs))
            color = (0, 200, 0) if oid else (60, 60, 220)
            if oid is not None and oid == self.selected:
                color = (255, 160, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"#{oid}" if oid else "new"
            node = self.mem.nodes.get(oid) if oid else None
            if node is not None and node.user_label:
                label += f" {node.user_label}"
            cv2.putText(vis, label, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if oid:
                self.last_obs_boxes.append((oid, (x1, y1, x2, y2)))
        # 状态栏
        n_nodes = len(self.mem.all_nodes())
        bar = (f"memory:{n_nodes}  pending:{len(self.mem.pending)}  "
               f"frame:{self.frame_id}  scale:{self.perc.pixels_per_meter:.0f}px/m"
               + ("  [PAUSED]" if self.paused else ""))
        cv2.rectangle(vis, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(vis, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        return vis

    # ---------------- 离线自检（无需摄像头，验证 demo 依赖的 osm API） ----------------
    def run_selftest(self):
        obs = [Observation(
            frame_id=1, timestamp=time.time(),
            cls_pred="未知物体", cls_score=0.9,
            bbox=(0.1, 0.1, 0.2, 0.2), depth=1.0, cam_pose=(0.0, 0.0, 0.0),
            feat=[0.5] * self.cfg.feat_dim, pos_world=[0.10, 0.05, -0.10],
        )]
        events = self.mem.update(obs, 1)
        self.handle_events(events)
        node = self.mem.all_nodes()[0]
        self.mem.rename(node.oid, "测试物体")
        self.mem.user_correct_position(node.oid, [0.20, 0.05, 0.0], 1)
        hits = self.qe.retrieve("测试物体在哪里？")
        ans = self.ag.generate("测试物体在哪里？", hits,
                               now_frame=1, memory=self.mem)
        _ = self.mem.snapshot_table()
        print("[自检] 全部通过：Observation / update / rename / correct / "
              "retrieve / generate / snapshot_table")
        print("[自检] 问答示例：", ans)

    # ---------------- 主循环 ----------------
    def run_real(self):
        self.perc = WebcamPerception(self.cfg, camera=self.args.camera)
        if not self.args.headless:
            cv2.namedWindow("ObjectMemory Demo")
            cv2.setMouseCallback("ObjectMemory Demo", self.on_mouse)
        print("[启动] 摄像头模式。请先让画面空 1-2 秒以学习背景。")
        print("[启动]", HELP)
        try:
            while True:
                if not self.paused:
                    frame, obs_list = self.perc.read()
                    if frame is None:
                        print("[结束] 摄像头读取失败。")
                        break
                    self.frame_id += 1
                    for o in obs_list:
                        o.frame_id = self.frame_id
                    matched_before = {id(o): None for o in obs_list}
                    events = self.mem.update(obs_list, self.frame_id)
                    # 反查本次观测关联到的记忆节点（用于画面标注）
                    for o in obs_list:
                        best, best_sim = None, 0.0
                        for node in self.mem.all_nodes():
                            sim, *_ = self.mem.reid.parts(o, node)
                            if sim > best_sim:
                                best, best_sim = node, sim
                        if best is not None and best_sim >= self.cfg.app_mid:
                            matched_before[id(o)] = best.oid
                    self.handle_events(events)
                key = -1
                if not self.args.headless:
                    cv2.imshow("ObjectMemory Demo",
                               self.draw(frame, obs_list, matched_before))
                    key = cv2.waitKey(30) & 0xFF
                else:
                    time.sleep(0.03)
                    key = -1
                    if self.frame_id % 300 == 0:
                        print(f"[运行中] 第 {self.frame_id} 帧，"
                              f"记忆物体 {len(self.mem.all_nodes())} 个")
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    self.paused = not self.paused
                elif key == ord("n"):
                    self.do_name()
                elif key == ord("c"):
                    self.do_confirm()
                elif key == ord("m"):
                    self.do_correct()
                elif key == ord("f"):
                    self.do_query()
                elif key == ord("s"):
                    self.do_snapshot()
                elif key == ord("+") or key == ord("="):
                    self.perc.pixels_per_meter *= 1.1
                elif key == ord("-"):
                    self.perc.pixels_per_meter /= 1.1
        finally:
            self.perc.release()
            if not self.args.headless:
                cv2.destroyAllWindows()
        print(self.mem.snapshot_table())

    def run_sim(self):
        from experiments.sim_dataset import generate_scene
        scene = generate_scene(self.args.scene, DEFAULT, seed=DEFAULT.seed)
        from osm.perception import Detector, PoseEstimator
        import random
        rng = random.Random(DEFAULT.seed)
        det, pe = Detector(DEFAULT, rng), PoseEstimator(DEFAULT, rng)
        mem = ObjectMemory(DEFAULT)
        qe = QueryEngine(mem, DEFAULT)
        ag = AnswerGenerator()
        print(f"[仿真] 场景 {self.args.scene}：{len(scene['objects'])} 个物体，"
              f"{len(scene['frames'])} 帧，{len(scene['moves'])} 次移动事件")
        for frame in scene["frames"]:
            f = dict(frame)
            f["pose"] = pe.estimate(frame["pose"])
            obs = det.predict(f)
            events = mem.update(obs, frame["frame_id"])
            if self.args.verbose:
                for ev in events:
                    print(f"  f{ev.frame_id:>3} {ev.kind:<16} #{ev.oid}")
        # 自动裁决全部待确认（理想用户）
        for key in sorted(list(mem.pending)):
            mem.resolve_pending(key, accept=True)
        print(mem.snapshot_table())
        print("\n—— 问答演示 ——")
        last_frame = scene["frames"][-1]["frame_id"]
        for q in scene["queries"][:6]:
            print("Q:", q["text"])
            print("A:", ag.generate(q["text"], qe.retrieve(q["text"]),
                                    now_frame=last_frame, memory=mem))


def main():
    ap = argparse.ArgumentParser(description="物体级空间记忆与场景理解 · 实时演示")
    ap.add_argument("--mode", choices=["auto", "real", "sim"], default="auto")
    ap.add_argument("--camera", type=int, default=0,
                    help="摄像头编号：0=内置，1 及以上=外接 USB 摄像头")
    ap.add_argument("--scene", type=int, default=0, help="仿真场景编号（sim 模式）")
    ap.add_argument("--headless", action="store_true", help="不弹窗，仅终端输出")
    ap.add_argument("--verbose", action="store_true", help="sim 模式打印全部事件")
    ap.add_argument("--selftest", action="store_true",
                    help="离线自检 demo 依赖的 osm API（无需摄像头）")
    args = ap.parse_args()

    app = DemoApp(args)
    if args.selftest:
        app.run_selftest()
        return

    mode = args.mode
    if mode == "auto":
        mode = "real" if HAS_CV else "sim"
    if mode == "real" and not HAS_CV:
        print("[提示] 未安装 opencv-python，自动切换到仿真模式。")
        mode = "sim"

    if mode == "real":
        try:
            app.run_real()
        except RuntimeError as e:
            print(f"[提示] {e}，自动切换到仿真模式。")
            app.run_sim()
    else:
        app.run_sim()


if __name__ == "__main__":
    main()
