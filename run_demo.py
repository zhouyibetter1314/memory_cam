"""单场景演示：跑完一个序列并打印记忆库状态与问答示例。

    cd code && python run_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from osm.config import DEFAULT
from osm.interaction import AnswerGenerator, QueryEngine
from osm.memory import ObjectMemory
from osm.perception import Detector, PoseEstimator
from experiments.sim_dataset import generate_scene


def main():
    cfg = DEFAULT
    scene = generate_scene(0, cfg, seed=cfg.seed)
    rng = __import__("random").Random(cfg.seed)
    det, pose_est = Detector(cfg, rng), PoseEstimator(cfg, rng)
    mem = ObjectMemory(cfg)

    for frame in scene["frames"]:
        f = dict(frame)
        f["pose"] = pose_est.estimate(frame["pose"])
        mem.update(det.predict(f), frame["frame_id"])

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
        print(f"  Q: {q}\n  A: {ag.generate(q, hits, scene['frames'][-1]['frame_id'])}")


if __name__ == "__main__":
    main()
