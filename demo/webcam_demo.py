#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""webcam_demo.py —— 面向日常动态场景的物体级空间记忆与场景理解：实时演示

支持两种感知后端：
  1) real 模式：摄像头或 --video 指定的视频文件（需 opencv-python + numpy）。
     采用「帧差 + 背景建模」检测前景物体，HSV 颜色直方图作为外观特征，
     接入与本课题一致的 ObjectMemory 物体级空间记忆管线。
  2) sim 模式（默认回退）：没有摄像头 / 没有 opencv 时自动切换，
     用 Sim-Scene 合成动态场景驱动同一条记忆管线，保证任何机器都能演示。

交互方式（real 模式）：
  画面窗口按键
    q / Esc   退出
    .         推进一帧并暂停
    空格      暂停 / 继续
    n         给当前选中物体命名（在终端输入名称，如"我的水杯"）
    c         裁决当前待确认事件（y=接受位置变更 / n=拒绝）
    m         主动纠正：点击画面后按 m，在终端输入 x,z 坐标
    f         发起自然语言查询（在终端输入问句，如"我的水杯在哪里？"）
    +/-       放大 / 缩小像素—米比例（用于粗略校准空间尺度）
    s         打印记忆库快照到终端
  鼠标左键点击画面：选中最近的记忆物体（用于命名 / 纠正）

运行示例：
  python demo/webcam_demo.py                      # 自动检测摄像头，失败回退 sim
  python demo/webcam_demo.py --camera 1           # 指定外接摄像头
  python demo/webcam_demo.py --mode sim --scene 3 # 仿真模式演示
  python demo/webcam_demo.py --headless           # 不弹窗，仅终端输出（服务器/录屏）
  python demo/webcam_demo.py --video input.mp4 --step --log-level DEBUG
  python demo/webcam_demo.py --video input.mp4 --fps -1

依赖说明：real 模式需要 cv2 与 numpy；其余全部复用 code/osm 纯标准库实现。
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

# 允许从 code/ 目录以外的地方运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osm.config import Config, DEFAULT
from osm.memory import ObjectMemory
from osm.interaction import QueryEngine, AnswerGenerator
from osm.types import Observation
from osm.runtime import (FrameController, add_runtime_arguments,
                         configure_logging, log_frame)

logger = logging.getLogger(__name__)

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

    def __init__(self, cfg: Config, camera: int = 0, width: int = 960, height: int = 540,
                 video: str = None):
        self.cfg = cfg
        self.is_video = video is not None
        if not HAS_CV:
            raise RuntimeError("图像输入需要 opencv-python 与 numpy")
        if self.is_video and not Path(video).is_file():
            raise RuntimeError(f"视频文件不存在：{video}")
        self.cap = cv2.VideoCapture(video if self.is_video else camera)
        if not self.cap.isOpened():
            self.cap.release()
            if self.is_video:
                raise RuntimeError(f"无法打开视频：{video}")
            raise RuntimeError(f"无法打开摄像头 {camera}（可用 --camera 1 尝试外接摄像头）")
        if not self.is_video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.is_video else None
        self.source_index = -1
        self.timestamp = None
        self._warned_timestamp = False
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
    def _input_timestamp(self):
        if not self.is_video:
            return time.time()
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if (math.isfinite(timestamp) and timestamp >= 0
                and (self.timestamp is None or timestamp > self.timestamp)):
            return timestamp
        if self.source_fps and math.isfinite(self.source_fps) and self.source_fps > 0:
            timestamp = self.source_index / self.source_fps
            if self.timestamp is None or timestamp > self.timestamp:
                if not self._warned_timestamp:
                    logger.warning("视频时间戳不可用，按帧索引与源帧率 %.3f 换算", self.source_fps)
                    self._warned_timestamp = True
                return timestamp
        raise RuntimeError("视频缺少有效时间戳或源帧率，无法确定输入时间")

    def read(self):
        """返回 (frame, observations)。observations 可能为空列表。"""
        ok, frame = self.cap.read()
        if not ok:
            return None, []
        self.source_index += 1
        self.timestamp = self._input_timestamp()
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
                frame_id=-1, timestamp=self.timestamp,
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

HELP = """按键：. 下一帧 | 空格 播放/暂停 | n 命名 | c 裁决待确认 | m 主动纠正 | f 查询 | s 快照 | +/- 调比例 | q 退出"""


class DemoApp:
    def __init__(self, args):
        self.args = args
        self.cfg = DEFAULT.override(feat_dim=96)   # HSV 直方图维度
        self.mem = ObjectMemory(self.cfg)
        self.qe = QueryEngine(self.mem, self.cfg)
        self.ag = AnswerGenerator()
        self.frame_id = 0
        self.timestamp = None
        self.control = FrameController(args.fps, args.step)
        self.selected = None               # 鼠标选中的记忆节点 oid
        self.last_obs_boxes = []           # 最近一帧观测框（像素），供点击选择

    @property
    def paused(self):
        return self.control.paused

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

    def do_confirm(self):
        if not self.mem.pending:
            logger.info("当前没有待确认事件")
            return
        key = sorted(self.mem.pending)[0]
        item = self.mem.pending[key]
        ans = self._ask(f"物体 {item['oid']} 疑似发生位置变化（sim={item['sim']:.2f}），"
                        f"确认移动？(y/n)：")
        if ans in ("y", "n"):
            self.mem.resolve_pending(key, accept=(ans == "y"))

    def do_correct(self):
        oid = self._pick_oid("输入要纠正的物体编号（回车取选中/最新）：")
        if oid is None:
            return
        txt = self._ask("输入新位置（格式 x,z 米，如 0.30,-0.15；回车保留记忆位置）：")
        node = self.mem.nodes.get(oid)
        if txt:
            try:
                x, z = [float(v) for v in txt.split(",")]
                pos = [x, node.pos[1] if node else 0.0, z]
            except ValueError:
                logger.warning("位置格式无效")
                return
        elif node is not None:
            pos = list(node.pos)
        else:
            return
        self.mem.user_correct_position(oid, pos, self.frame_id, timestamp=self.timestamp)

    def do_query(self):
        text = self._ask("输入问句（如：我的水杯在哪里？）：")
        if not text:
            return
        hits = self.qe.retrieve(text)
        answer = self.ag.generate(text, hits, now_frame=self.frame_id, memory=self.mem,
                                  now_timestamp=self.timestamp)
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
            logger.info("已选中物体 %s", best)

    # ---------------- 事件处理 ----------------
    def handle_events(self, events):
        for ev in events:
            if ev.kind == "pending_confirm":
                logger.info("物体 #%s 待确认，按 c 裁决", ev.oid)

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
               f"  time:{self.timestamp:.3f}s  fps:{self.args.fps:g}"
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
        logger.info("自检通过：Observation / update / rename / correct / retrieve / generate / snapshot_table")
        print("[自检] 问答示例：", ans)

    # ---------------- 主循环 ----------------
    def _handle_key(self, key):
        if key in (ord("q"), 27):
            return False
        if key == ord("."):
            self.control.request_step()
        elif key == ord(" "):
            self.control.toggle_pause()
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
        elif key in (ord("+"), ord("=")):
            self.perc.pixels_per_meter *= 1.1
        elif key == ord("-"):
            self.perc.pixels_per_meter /= 1.1
        return True

    def _wait_gui(self, timing):
        """限速时持续泵送窗口事件；暂停只等待，不处理新帧。"""
        while True:
            remaining = self.control.remaining(*timing) if timing else 0.0
            if not self.control.ready:
                delay_ms = 30
            else:
                delay_ms = max(1, min(30, math.ceil(remaining * 1000)))
            key = cv2.waitKey(delay_ms) & 0xFF
            if not self._handle_key(key):
                return False
            if timing is None or self.paused or self.control.remaining(*timing) <= 0:
                return True

    def run_real(self):
        self.perc = WebcamPerception(self.cfg, camera=self.args.camera,
                                     video=self.args.video)
        try:
            if not self.args.headless:
                cv2.namedWindow("ObjectMemory Demo")
                cv2.setMouseCallback("ObjectMemory Demo", self.on_mouse)
            logger.info("启动%s模式；第一帧作为背景", "视频" if self.args.video else "摄像头")
            logger.info(HELP)
            frame, obs_list, matched = None, [], {}
            while True:
                if self.args.headless and not self.control.terminal_advance():
                    break
                timing = None
                if self.control.ready:
                    timing = self.control.begin_frame()
                    frame, obs_list = self.perc.read()
                    if frame is None:
                        if self.args.video:
                            logger.info("视频播放结束")
                        else:
                            logger.warning("摄像头读取失败")
                        break
                    self.frame_id += 1
                    self.timestamp = self.perc.timestamp
                    for obs in obs_list:
                        obs.frame_id = self.frame_id
                    events = self.mem.update(obs_list, self.frame_id)
                    # 保留现有画面标注逻辑，关联判定详情由核心 logger 输出。
                    matched = {id(obs): None for obs in obs_list}
                    for obs in obs_list:
                        best, best_sim = None, 0.0
                        for node in self.mem.all_nodes():
                            sim, *_ = self.mem.reid.parts(obs, node)
                            if sim > best_sim:
                                best, best_sim = node, sim
                        if best is not None and best_sim >= self.cfg.app_mid:
                            matched[id(obs)] = best.oid
                    self.handle_events(events)
                    log_frame(logger, self.frame_id, self.timestamp, obs_list,
                              self.control.clock() - timing[0])
                    if self.args.headless and self.paused:
                        print(f"[帧 {self.frame_id}] time={self.timestamp:.3f}s "
                              f"observations={len(obs_list)} memory={len(self.mem.all_nodes())}")
                if not self.args.headless:
                    cv2.imshow("ObjectMemory Demo", self.draw(frame, obs_list, matched))
                    if not self._wait_gui(timing):
                        break
                elif timing:
                    self.control.pace(*timing)
        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.perc.release()
            if not self.args.headless:
                cv2.destroyAllWindows()
            logger.info("结束，已处理 %s 帧", self.frame_id)
            print(self.mem.snapshot_table())

    def run_sim(self):
        from experiments.sim_dataset import generate_scene
        from osm.perception import Detector, PoseEstimator
        import random
        scene = generate_scene(self.args.scene, DEFAULT, seed=DEFAULT.seed)
        rng = random.Random(DEFAULT.seed)
        det, pe = Detector(DEFAULT, rng), PoseEstimator(DEFAULT, rng)
        self.mem = ObjectMemory(DEFAULT)
        self.qe = QueryEngine(self.mem, DEFAULT)
        self.ag = AnswerGenerator(DEFAULT.frame_interval_s)
        logger.info("仿真场景 %s：%s 个物体，%s 帧，%s 次移动",
                    self.args.scene, len(scene["objects"]), len(scene["frames"]), len(scene["moves"]))
        completed = True
        processed = 0
        try:
            for frame in scene["frames"]:
                if not self.control.terminal_advance():
                    completed = False
                    break
                timing = self.control.begin_frame()
                f = dict(frame)
                f["pose"] = pe.estimate(frame["pose"])
                obs = det.predict(f)
                self.frame_id, self.timestamp = frame["frame_id"], frame["timestamp"]
                self.mem.update(obs, self.frame_id)
                processed += 1
                log_frame(logger, self.frame_id, self.timestamp, obs,
                          self.control.clock() - timing[0])
                if self.paused:
                    print(f"[帧 {self.frame_id}] time={self.timestamp:.3f}s "
                          f"observations={len(obs)} memory={len(self.mem.all_nodes())}")
                self.control.pace(*timing)
        except KeyboardInterrupt:
            completed = False
            logger.info("用户中断")
        if completed:
            # 保留原有仿真演示策略：完成序列后自动接受待确认事件。
            for key in sorted(list(self.mem.pending)):
                self.mem.resolve_pending(key, accept=True)
        logger.info("结束，已处理 %s 帧", processed)
        print(self.mem.snapshot_table())
        if processed:
            print("\n—— 问答演示 ——")
            for q in scene["queries"][:6]:
                print("Q:", q["text"])
                print("A:", self.ag.generate(q["text"], self.qe.retrieve(q["text"]),
                                            now_frame=self.frame_id, memory=self.mem,
                                            now_timestamp=self.timestamp))


def main(argv=None):
    ap = argparse.ArgumentParser(description="物体级空间记忆与场景理解 · 实时演示")
    ap.add_argument("--mode", choices=["auto", "real", "sim"], default="auto")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--video", help="输入视频文件，保留源时间戳")
    source.add_argument("--camera", type=int, default=0,
                    help="摄像头编号：0=内置，1 及以上=外接 USB 摄像头")
    ap.add_argument("--scene", type=int, default=0, help="仿真场景编号（sim 模式）")
    ap.add_argument("--headless", action="store_true", help="不弹窗，仅终端输出")
    ap.add_argument("--verbose", action="store_true", help="兼容选项：默认使用 DEBUG 日志；显式 --log-level 优先")
    ap.add_argument("--selftest", action="store_true",
                    help="离线自检 demo 依赖的 osm API（无需摄像头）")
    add_runtime_arguments(ap)
    args = ap.parse_args(argv)
    if args.video and args.mode == "sim":
        ap.error("--video 不能与 --mode sim 组合")
    if args.video and not Path(args.video).is_file():
        ap.error(f"视频文件不存在：{args.video}")
    if args.video and not HAS_CV:
        ap.error("视频输入需要 opencv-python 与 numpy")
    configure_logging(args)

    app = DemoApp(args)
    if args.selftest:
        app.run_selftest()
        return

    mode = "real" if args.video else args.mode
    if mode == "auto":
        mode = "real" if HAS_CV else "sim"
    if mode == "real" and not HAS_CV:
        logger.warning("缺少 opencv-python / numpy，切换到仿真模式")
        mode = "sim"

    if mode == "real":
        try:
            app.run_real()
        except RuntimeError as e:
            if args.video:
                ap.error(str(e))
            logger.warning("%s，切换到仿真模式", e)
            app.run_sim()
    else:
        app.run_sim()


if __name__ == "__main__":
    main()
