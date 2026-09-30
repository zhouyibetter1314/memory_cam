"""
osm —— Object-level Spatial Memory（面向日常动态场景的物体级空间记忆原型）

模块划分（对应报告"感知层 / 记忆层 / 交互层"三层架构）：
    perception  ：物体与空间信息提取（检测、深度反投影、位姿、跟踪、重识别）
    memory      ：持久物体记忆的构建与更新（含人机协同确认机制）
    interaction ：空间记忆交互与应用（检索 + 自然语言问答）
    metrics     ：评价指标计算
    pipeline    ：端到端管线编排

全部实现仅依赖 Python 标准库，便于在无 GPU / 无第三方包的环境中复现实验。
真实模型（YOLO / CLIP / SLAM / VLM）通过"后端接口"接入，默认使用仿真后端。
"""

__version__ = "1.0.0"
