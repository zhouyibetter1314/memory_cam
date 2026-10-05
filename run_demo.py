"""单场景演示：跑完一个序列并打印记忆库状态与问答示例。

    cd code && python run_demo.py
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from osm.config import DEFAULT
from osm.interaction import AnswerGenerator, QueryEngine
from osm.memory import ObjectMemory
from osm.perception import Detector, PoseEstimator
from experiments.sim_dataset import generate_scene
from osm.runtime import (FrameController, add_runtime_arguments,
                         configure_logging, log_frame)

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="单场景物体记忆演示")
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging(args)
    control = FrameController(args.fps, args.step)
    cfg = DEFAULT
    scene = generate_scene(0, cfg, seed=cfg.seed)
    rng = __import__("random").Random(cfg.seed)
    det, pose_est = Detector(cfg, rng), PoseEstimator(cfg, rng)
    mem = ObjectMemory(cfg)

    logger.info("启动仿真演示：%s 帧", len(scene["frames"]))
    last_frame = None
    processed = 0
    try:
        for frame in scene["frames"]:
            if not control.terminal_advance():
                break
            timing = control.begin_frame()
            f = dict(frame)
            f["pose"] = pose_est.estimate(frame["pose"])
            obs = det.predict(f)
            mem.update(obs, frame["frame_id"])
            last_frame = frame
            processed += 1
            log_frame(logger, frame["frame_id"], frame["timestamp"], obs,
                      control.clock() - timing[0])
            if control.paused:
                print(f"[帧 {frame['frame_id']}] time={frame['timestamp']:.3f}s "
                      f"observations={len(obs)} memory={len(mem.all_nodes())}")
            control.pace(*timing)
    except KeyboardInterrupt:
        logger.info("用户中断")
    logger.info("结束，已处理 %s 帧", processed)

    print(f"真值物体数：{len(scene['objects'])}；记忆库节点数：{len(mem.all_nodes())}")
    print(f"自动更新 {mem.n_auto_update} 次；用户确认 {mem.n_user_confirmed} 次；"
          f"用户拒绝 {mem.n_user_rejected} 次\n")
    print("记忆库（前 8 个节点）：")
    for n in sorted(mem.all_nodes(), key=lambda x: -x.conf)[:8]:
        pos = "({:.2f}, {:.2f}, {:.2f})".format(*n.pos)
        print(f"  #{n.oid} {n.cls:<5s} 位置{pos} 观测{n.obs_count:3d}次 置信度{n.conf:.2f} {n.state}")

    qe, ag = QueryEngine(mem, cfg), AnswerGenerator(cfg.frame_interval_s)
    print("\n问答示例：")
    for q in ["我的钥匙放在哪了？", "水杯在哪里？", "上次什么时候看到背包的？",
              "最近有什么东西位置变了？"]:
        hits = qe.retrieve(q)
        if last_frame is not None:
            print(f"  Q: {q}\n  A: {ag.generate(q, hits, last_frame['frame_id'], now_timestamp=last_frame['timestamp'])}")


if __name__ == "__main__":
    main()
