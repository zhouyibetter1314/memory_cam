"""通用工具：向量运算、相似度、匈牙利匹配、统计量。"""

import math
import random
from typing import List, Sequence


# ---------------------------- 向量运算 ----------------------------

def l2_normalize(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1e-12
    return [x / n for x in v]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def dist3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def ema(old: Sequence[float], new: Sequence[float], alpha: float) -> List[float]:
    return [o + alpha * (n - o) for o, n in zip(old, new)]


# ---------------------------- 匈牙利匹配 ----------------------------

def hungarian(cost: List[List[float]]) -> List[int]:
    """求解最小代价二分匹配（e-maxx 版本，O(n^2 m)）。

    Args:
        cost: n x m 代价矩阵，要求 n <= m
    Returns:
        长度为 n 的列表，res[i] = 与第 i 行匹配的行号（-1 表示未匹配）
    """
    n, m = len(cost), len(cost[0]) if cost else 0
    if n == 0 or m == 0:
        return []
    if n > m:  # 保证 n <= m
        t = hungarian([[cost[i][j] for i in range(n)] for j in range(m)])
        res = [-1] * n
        for j, i in enumerate(t):
            if i >= 0:
                res[i] = j
        return res

    INF = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    res = [-1] * n
    for j in range(1, m + 1):
        if p[j] > 0:
            res[p[j] - 1] = j - 1
    return res


# ---------------------------- 统计量 ----------------------------

def median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def gaussian(rng: random.Random, mu: float = 0.0, sigma: float = 1.0) -> float:
    return rng.gauss(mu, sigma)
