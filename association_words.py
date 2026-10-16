# -*- coding: utf-8 -*-
"""
钱谦益 × 白居易 关联词挖掘（Fisher 精确检验 + BH 校正）
思路：找出在钱谦益和白居易中【同时高频】、且相对其他唐宋诗人显著富集的词。
"""

import os
import math
from collections import Counter
from scipy.stats import fisher_exact

# 复用主脚本的切分与特征提取逻辑
from stylometry_compare import (
    QIAN_FILE,
    TANG_SONG_FILE,
    POET_ALIASES,
    split_poets_by_line,
    extract_1_2_grams,
    _is_cjk,
    FUNCTION_WORDS,
)

TOP_K = 20          # 最多输出 20 个关联词
MIN_COUNT = 3       # 词在钱谦益和白居易中各自至少出现 3 次
P_THRESHOLD = 0.05  # 校正后 p 值阈值


def main():
    print("=" * 64)
    print("钱谦益 × 白居易 关联词挖掘（Fisher 精确检验 + Benjamini-Hochberg 校正）")
    print("=" * 64)

    # ---- 读取钱谦益 ----
    with open(QIAN_FILE, "r", encoding="utf-8", errors="ignore") as f:
        qian_text = f.read()
    qian_counter = extract_1_2_grams(qian_text)

    # ---- 切分唐宋六家 ----
    poet_buffers = split_poets_by_line(TANG_SONG_FILE)
    poet_counters = {}
    for poet in POET_ALIASES:
        text = "\n".join(poet_buffers.get(poet, []))
        poet_counters[poet] = extract_1_2_grams(text)

    bai_counter = poet_counters["白居易"]

    # ---- 其他诗人（排除白居易）合并为"对照组" ----
    other_counter = Counter()
    for poet in POET_ALIASES:
        if poet == "白居易":
            continue
        other_counter += poet_counters[poet]

    # ---- 候选词：在钱谦益和白居易中均 >= MIN_COUNT ----
    candidates = [
        w for w in qian_counter
        if qian_counter[w] >= MIN_COUNT and bai_counter.get(w, 0) >= MIN_COUNT
    ]
    print(f"\n候选词总数（钱&白各≥{MIN_COUNT}次）: {len(candidates)}")

    # ---- 对每个候选词做 Fisher 精确检验 ----
    # 2×2 列联表（以钱谦益为例，对比"其他诗人"）:
    #            词W    非词W
    # 钱谦益      a       b
    # 其他诗人    c       d
    #
    # 同理对白居易也做一次。取两者 p 值的较大值（更保守），
    # 要求两者都显著，才算"关联词"。
    qian_total = sum(qian_counter.values())
    bai_total = sum(bai_counter.values())
    other_total = sum(other_counter.values())

    raw_results = []  # (word, p_qian, p_bai, p_max, cq, cb, co)
    for w in candidates:
        a_q = qian_counter[w]
        b_q = qian_total - a_q
        c_q = other_counter.get(w, 0)
        d_q = other_total - c_q
        _, p_q = fisher_exact([[a_q, b_q], [c_q, d_q]], alternative="greater")

        a_b = bai_counter[w]
        b_b = bai_total - a_b
        c_b = other_counter.get(w, 0)
        d_b = other_total - c_b
        _, p_b = fisher_exact([[a_b, b_b], [c_b, d_b]], alternative="greater")

        # 两者都必须显著（取较大 p 值作为代表，更保守）
        p_max = max(p_q, p_b)
        raw_results.append((w, p_q, p_b, p_max, a_q, a_b, c_b))

    # ---- Benjamini-Hochberg 校正 ----
    raw_results.sort(key=lambda x: x[3])  # 按 p_max 升序
    p_values = [r[3] for r in raw_results]

    # 手动 BH 校正（避免依赖 qhchina 内部 API 版本差异）
    n = len(p_values)
    adjusted = [0.0] * n
    # BH: 从大到小，adjusted_p_i = min(1, p_i * n / rank_i)，并取累积最小
    sorted_idx = sorted(range(n), key=lambda i: p_values[i], reverse=True)
    prev = 1.0
    for rank, idx in enumerate(sorted_idx, 1):
        bh = p_values[idx] * n / rank
        if bh > 1.0:
            bh = 1.0
        if bh < prev:
            prev = bh
        adjusted[idx] = prev

    # ---- 筛选显著关联词 ----
    significant = []
    for i, (word, p_q, p_b, p_max, cq, cb, co) in enumerate(raw_results):
        if adjusted[i] < P_THRESHOLD:
            significant.append((word, adjusted[i], p_q, p_b, cq, cb, co))

    print(f"经 BH 校正后显著关联词数（adj_p < {P_THRESHOLD}）: {len(significant)}")

    # ---- 输出排行榜 ----
    print("\n" + "=" * 64)
    print(f"[钱谦益 × 白居易 关联词排行榜]（Top-{min(TOP_K, len(significant))}）")
    print("=" * 64)
    print(f"{'排名':<4} {'关联词':<8} {'BH校p':<10} {'钱频':<6} {'白频':<6} {'他频':<6} {'钱p':<10} {'白p':<10}")
    print("-" * 64)

    for rank, (word, adj_p, p_q, p_b, cq, cb, co) in enumerate(significant[:TOP_K], 1):
        print(f"{rank:<4} {word:<8} {adj_p:<10.4f} {cq:<6} {cb:<6} {co:<6} {p_q:<10.4g} {p_b:<10.4g}")

    print("=" * 64)
    print("说明：钱频=钱谦益中该词频次，白频=白居易中频次，他频=其他唐宋诗人合计频次")
    print("      钱p/白p = Fisher 精确检验（单侧 greater）原始 p 值")
    print("      BH校p = Benjamini-Hochberg 校正后 p 值（< 0.05 视为显著关联）")


if __name__ == "__main__":
    main()
