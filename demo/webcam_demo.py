#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""webcam_demo.py —— 面向日常动态场景的物体级空间记忆与场景理解：实时演示

摄像头或 --video 指定的视频文件 → ImagePerception → ObjectMemory。
默认 YOLO26n；--detector motion 切回帧差＋背景建模。

交互方式：
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
  python demo/webcam_demo.py                      # 默认摄像头与 YOLO26n
  python demo/webcam_demo.py --camera 1           # 指定外接摄像头
  python demo/webcam_demo.py --headless           # 不弹窗，仅终端输出（服务器/录屏）
  python demo/webcam_demo.py --video input.mp4 --step --log-level DEBUG
  python demo/webcam_demo.py --video input.mp4 --fps -1

依赖安装：pip install -r requirements.txt。
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

# 允许从项目目录以外的地方运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osm.config import DEFAULT
from osm.memory import ObjectMemory
from osm.interaction import QueryEngine, AnswerGenerator
from osm.perception import ImagePerception
from osm.labels import display_class
from osm.runtime import (FrameController, add_runtime_arguments,
                         configure_logging, log_frame)

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV = True
except ImportError:
    cv2 = None
    HAS_CV = False


# =====================================================================
# 图像输入与命令行参数
# =====================================================================

def confidence_value(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("YOLO 置信度必须为 [0,1] 内的有限数值") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("YOLO 置信度必须为 [0,1] 内的有限数值")
    return value


def positive_int(text):
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("YOLO 图像尺寸必须为正整数") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("YOLO 图像尺寸必须为正整数")
    return value


class WebcamPerception:
    """摄像头 / 视频输入适配器；图像感知委托给 ImagePerception。"""

    def __init__(self, cfg, camera: int = 0, width: int = 960, height: int = 540,
                 video: str = None, *, detector: str = "yolo",
                 yolo_model: str = "yolo26n.pt", yolo_conf: float = 0.35,
                 yolo_imgsz: int = 640, yolo_device: str = None,
                 yolo_classes: list = None):
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
        try:
            if not self.is_video:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.is_video else None
            self.perception = ImagePerception(
                cfg, detector=detector, yolo_model=yolo_model, yolo_conf=yolo_conf,
                yolo_imgsz=yolo_imgsz, yolo_device=yolo_device, yolo_classes=yolo_classes)
        except BaseException:
            self.cap.release()
            raise
        self.source_index = -1
        self.timestamp = None
        self._warned_timestamp = False

    @property
    def pixels_per_meter(self):
        return self.perception.pixels_per_meter

    @pixels_per_meter.setter
    def pixels_per_meter(self, value):
        self.perception.pixels_per_meter = value

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
        observations = self.perception.predict(
            frame, frame_id=self.source_index + 1, timestamp=self.timestamp)
        return frame, observations

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
            label += f" {display_class(obs.cls_pred)} {obs.cls_score:.2f}"
            node = self.mem.nodes.get(oid) if oid else None
            if node is not None and node.user_label and node.user_label.isascii():
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

    def run(self):
        self.perc = WebcamPerception(
            self.cfg, camera=self.args.camera, video=self.args.video,
            detector=self.args.detector, yolo_model=self.args.yolo_model,
            yolo_conf=self.args.yolo_conf, yolo_imgsz=self.args.yolo_imgsz,
            yolo_device=self.args.yolo_device, yolo_classes=self.args.yolo_classes)
        try:
            if not self.args.headless:
                cv2.namedWindow("ObjectMemory Demo")
                cv2.setMouseCallback("ObjectMemory Demo", self.on_mouse)
            logger.info("启动%s模式；detector=%s%s", "视频" if self.args.video else "摄像头",
                        self.args.detector,
                        "；第一帧作为背景" if self.args.detector == "motion" else "")
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



def main(argv=None):
    ap = argparse.ArgumentParser(description="物体级空间记忆与场景理解 · 实时演示")
    ap.add_argument("--detector", choices=["yolo", "motion"], default="yolo",
                    help="真实输入检测器：yolo（默认 YOLO26n）或 motion（帧差＋背景建模）")
    ap.add_argument("--yolo-model", default="yolo26n.pt", help="YOLO 检测权重名称或本地路径")
    ap.add_argument("--yolo-conf", type=confidence_value, default=0.35, help="YOLO 置信度阈值（默认 0.35）")
    ap.add_argument("--yolo-imgsz", type=positive_int, default=640, help="YOLO 推理图像尺寸（默认 640）")
    ap.add_argument("--yolo-device", help="YOLO 设备：cpu、mps、0 等；默认由 Ultralytics 选择")
    ap.add_argument("--yolo-classes", nargs="+", type=int,
                    help="只检测指定模型类别编号，如 41 67；默认全部类别")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--video", help="输入视频文件，保留源时间戳")
    source.add_argument("--camera", type=int, default=0,
                    help="摄像头编号：0=内置，1 及以上=外接 USB 摄像头")
    ap.add_argument("--headless", action="store_true", help="不弹窗，仅终端输出")
    ap.add_argument("--verbose", action="store_true", help="兼容选项：默认使用 DEBUG 日志；显式 --log-level 优先")
    add_runtime_arguments(ap)
    args = ap.parse_args(argv)
    if args.video and not Path(args.video).is_file():
        ap.error(f"视频文件不存在：{args.video}")
    if not HAS_CV:
        ap.error("摄像头与视频输入需要 pip install -r requirements.txt")
    configure_logging(args)

    try:
        DemoApp(args).run()
    except RuntimeError as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
