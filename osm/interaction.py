"""交互层：基于空间记忆的检索与自然语言问答。

对应报告 3.4 节。设计要点：
    * 查询解析：中文提问 → 意图（定位 / 时间 / 变化回顾）+ 目标物体
    * 混合检索：类别过滤 + 用户自定义名称匹配 + 记忆可信度加权
    * 答案生成：模板化生成器（可插拔替换为 VLM/LLM，接口见 AnswerGenerator.llm_answer）

真实系统中，VLM 负责"这张图里是什么/在哪"，LLM 负责组织自然语言答复；
本原型用确定性模板保证实验结果可复现。
"""

import re
from typing import Dict, List, Optional, Tuple

from .types import ObjectNode


# 中文常用物体名称 → 规范类别的同义词表（真实系统可由 LLM 做同义归一）
SYNONYMS: Dict[str, str] = {
    "杯子": "水杯", "水杯": "水杯", "马克杯": "水杯", "保温杯": "水壶",
    "钥匙": "钥匙", "门钥匙": "钥匙", "车钥匙": "钥匙",
    "包": "背包", "背包": "背包", "书包": "背包", "双肩包": "背包",
    "眼镜": "眼镜", "墨镜": "眼镜",
    "手机": "手机", "电话": "手机",
    "遥控器": "遥控器", "遥控": "遥控器", "电视遥控器": "遥控器",
    "书": "书本", "书本": "书本", "课本": "书本",
    "充电器": "充电器", "充电头": "充电器", "充电线": "充电器",
    "耳机": "耳机", "耳麦": "耳机",
    "鼠标": "鼠标", "无线鼠标": "鼠标",
    "剪刀": "剪刀",
    "药": "药盒", "药盒": "药盒",
    "手表": "手表", "手环": "手表",
    "伞": "雨伞", "雨伞": "雨伞",
    "水壶": "水壶",
    "电脑": "笔记本", "笔记本": "笔记本", "本子": "笔记本",
    "台灯": "台灯", "灯": "台灯",
    "拖鞋": "拖鞋", "鞋": "拖鞋",
}

INTENT_PATTERNS = [
    ("change", re.compile(r"(变化|变了|移动|动过|不一样|新增|不见)")),
    ("when", re.compile(r"(什么时候|上次|最近一次|何时|多久)")),
    ("locate", re.compile(r"(在哪|哪里|哪儿|位置|放在|找)")),
]


class QueryEngine:
    """空间记忆检索引擎。"""

    def __init__(self, memory, cfg):
        self.memory = memory
        self.cfg = cfg

    # ---------------------------------------------------------- 查询解析
    @staticmethod
    def parse(text: str) -> Tuple[str, Optional[str]]:
        """返回 (意图, 规范类别)。"""
        intent = "locate"
        for name, pat in INTENT_PATTERNS:
            if pat.search(text):
                intent = name
                break
        target = None
        for word, canon in SYNONYMS.items():
            if word in text:
                target = canon
                break
        return intent, target

    # ---------------------------------------------------------- 检索
    def retrieve(self, text: str, top_k: Optional[int] = None) -> List[Tuple[ObjectNode, float]]:
        intent, target = self.parse(text)
        k = top_k or self.cfg.top_k
        nodes = self.memory.all_nodes()

        if intent == "change":
            recent = [n for n in nodes if len(n.history) > 1]
            recent.sort(key=lambda n: n.last_seen, reverse=True)
            return [(n, n.conf) for n in recent[:k]]

        scored = []
        for n in nodes:
            s = 0.0
            if target:
                if n.cls == target:
                    s += 1.0
                if n.user_label and (n.user_label == target or target in n.user_label):
                    s += 0.6
                if s == 0:
                    continue
                s += 0.25 * n.conf
            else:
                # 类别未命中词表：退化为名称字符重叠匹配
                s = 0.3 * n.conf
            scored.append((n, s))
        scored.sort(key=lambda x: (-x[1], -x[0].conf, x[0].oid))
        return scored[:k]


class AnswerGenerator:
    """自然语言答案生成（模板化，可插拔 LLM）。"""

    def __init__(self, frame_interval_s: float = 3.0):
        self.frame_interval_s = frame_interval_s

    def generate(self, text: str, hits: List[Tuple[ObjectNode, float]],
                 now_frame: int, memory=None) -> str:
        intent, _ = QueryEngine.parse(text)
        if not hits:
            return "记忆中没有找到相关物品，是否需要我再扫一遍房间？"

        if intent == "change":
            names = []
            for n, _ in hits:
                if len(n.history) >= 2:
                    d = sum((a - b) ** 2 for a, b in zip(n.history[-1], n.history[0])) ** 0.5
                    names.append(f"{self._name(n)}（位移约 {d:.2f} m）")
            return "最近位置发生变化的物品：" + "；".join(names[:3]) + "。"

        if intent == "when":
            n = hits[0][0]
            delta_s = (now_frame - n.last_seen) * self.frame_interval_s
            if delta_s < 60:
                t = f"{delta_s:.0f} 秒前"
            else:
                t = f"{delta_s / 60:.1f} 分钟前"
            return f"{self._name(n)} 最后一次出现在 {t}，位置 {self._pos(n)}。"

        n = hits[0][0]
        ans = f"{self._name(n)} 在 {self._pos(n)}（置信度 {n.conf:.2f}）。"
        if len(hits) > 1:
            alt = "、".join(self._name(m) for m, _ in hits[1:])
            ans += f" 另外还找到：{alt}。"
        return ans

    @staticmethod
    def _name(n: ObjectNode) -> str:
        return n.user_label or n.cls

    @staticmethod
    def _pos(n: ObjectNode) -> str:
        x, y, z = n.pos
        return f"距原点 ({x:.2f}, {z:.2f}) m、高 {y:.2f} m 处"

    # ---------------------------------------------------------- 真实 LLM 接口
    def llm_answer(self, text: str, hits: List[Tuple[ObjectNode, float]],
                   now_frame: int) -> str:  # pragma: no cover - 需联网模型
        """TODO：接入 VLM/LLM。

        真实系统的做法：
          1) 用 VLM 对 Top-K 候选的裁剪图像做二次确认（视觉校验）
          2) 把候选的 {名称, 位置, 最后出现时间, 置信度} 组织为结构化 prompt
          3) 由 LLM 生成符合用户习惯的自然语言答复
        为保持实验可复现，本原型默认调用 generate()，不发起网络请求。
        """
        raise NotImplementedError("请在此接入 VLM/LLM 后端")
