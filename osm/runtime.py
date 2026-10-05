"""共享命令行日志与演示运行控制；输入时间不受播放速度影响。"""

import argparse
import logging
import math
import time


def fps_value(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fps 必须为正数或 -1") from exc
    if not math.isfinite(value) or (value != -1 and value <= 0):
        raise argparse.ArgumentTypeError("fps 必须为有限正数或 -1")
    return value


def add_log_arguments(parser):
    # None 用于区分默认 INFO 与用户显式选择 INFO。
    parser.add_argument("--log-level", choices=("INFO", "DEBUG"), default=None,
                        help="日志级别（默认 INFO），输出到 stderr")


def add_runtime_arguments(parser):
    add_log_arguments(parser)
    parser.add_argument("--step", action="store_true", help="第一帧后等待手动推进")
    parser.add_argument("--fps", type=fps_value, default=30.0,
                        help="处理帧率上限（默认 30）；-1 不主动限速")


def configure_logging(args):
    level = args.log_level or ("DEBUG" if getattr(args, "verbose", False) else "INFO")
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", force=True)


class FrameController:
    """门控读取与限速；首帧自动处理，单步后保持暂停。"""

    def __init__(self, fps=30.0, step=False, clock=None, sleep=None):
        self.fps = fps_value(str(fps))
        self.paused = step
        self._requested = True
        self.clock = clock or time.perf_counter
        self.sleep = sleep or time.sleep

    @property
    def ready(self):
        return not self.paused or self._requested

    def begin_frame(self):
        if not self.ready:
            raise RuntimeError("暂停期间不能处理新帧")
        self._requested = False
        return self.clock(), self.paused

    def request_step(self):
        self.paused = True
        self._requested = True

    def toggle_pause(self):
        self.paused = not self.paused
        self._requested = False

    def remaining(self, started, single_step=False):
        if single_step or self.fps == -1:
            return 0.0
        return max(0.0, 1.0 / self.fps - (self.clock() - started))

    def pace(self, started, single_step=False):
        delay = self.remaining(started, single_step)
        if delay > 0:
            self.sleep(delay)

    def terminal_advance(self):
        """终端单步；非法输入不推进，EOF 与 q 正常结束。"""
        if self.ready:
            return True
        while True:
            try:
                command = input("[单步] 回车处理下一帧，q 退出：").strip().lower()
            except EOFError:
                return False
            if command == "q":
                return False
            if not command:
                self.request_step()
                return True


def log_frame(logger, frame_id, timestamp, observations, elapsed):
    logger.debug("frame=%s input_time=%.6f observations=%s processing_ms=%.3f",
                 frame_id, timestamp, len(observations), elapsed * 1000)
