# -*- coding: utf-8 -*-
"""
七位清代诗人 vs 唐宋六家 余弦相似度比对
复用 stylometry_final.py 的修正版切分逻辑。
"""

import os
import math
from collections import Counter

from stylometry_final import (
    TANG_SONG_FILE,
    POET_ALIASES,
    split_poets,
    extract_1_2_grams,
    build_vector,
    cosine,
    _is_cjk,
)

BASE = os.path.dirname(os.path.abspath(__file__))
TOP_N = 300

# 七位清代诗人文件
QING_POETS = {
    "钱谦益": "qian_qianyi.txt",
    "吴伟业": "wu_weiye.txt",
    "王士祯": "wang_shizhen.txt",
    "朱彝尊": "zhu_yizun.txt",
    "尤侗":   "you_tong.txt",
    "查慎行": "zha_shenxing.txt",
    "顾贞观": "gu_zhenguan.txt",
    "叶燮":   "ye_xie.txt",
}


def main():
    print("=" * 64)
    print("八位清代诗人 vs 唐宋六家 余弦相似度比对")
    print("=" * 64)

    # ---- 读取清代诗人 ----
    qing_cnts = {}
    qing_chars = {}
    for name, fname in QING_POETS.items():
        path = os.path.join(BASE, fname)
        text = open(path, encoding="utf-8", errors="ignore").read()
        qing_cnts[name] = extract_1_2_grams(text)
        qing_chars[name] = sum(1 for c in text if _is_cjk(c))

    # ---- 切分唐宋六家 ----
    buffers = split_poets(TANG_SONG_FILE)
    tang_cnts = {}
    tang_chars = {}
    for poet in POET_ALIASES:
        text = "\n".join(buffers[poet])
        tang_cnts[poet] = extract_1_2_grams(text)
        tang_chars[poet] = sum(1 for c in text if _is_cjk(c))

    # ---- 打印字数 ----
    print("\n--- 清代诗人 ---")
    for name in QING_POETS:
        print(f"  [{name}] {qing_chars[name]} 字")
    print("--- 唐宋诗人 ---")
    for poet in POET_ALIASES:
        print(f"  [{poet}] {tang_chars[poet]} 字")

    # ---- 共享词表 ----
    global_cnt = Counter()
    for c in qing_cnts.values():
        global_cnt += c
    for c in tang_cnts.values():
        global_cnt += c
    vocab = [w for w, _ in global_cnt.most_common(TOP_N)]

    # ---- 计算相似度矩阵 ----
    results = {}
    for qname in QING_POETS:
        qvec = build_vector(qing_cnts[qname], vocab)
        sims = []
        for tpoet in POET_ALIASES:
            tvec = build_vector(tang_cnts[tpoet], vocab)
            sims.append((tpoet, cosine(qvec, tvec)))
        sims.sort(key=lambda x: x[1], reverse=True)
        results[qname] = sims

    # ---- 输出排行榜 ----
    for qname in QING_POETS:
        print(f"\n{'='*64}")
        print(f"[{qname} 诗风唐宋归属度排行榜]")
        print(f"{'='*64}")
        for i, (tpoet, sim) in enumerate(results[qname], 1):
            tag = " (最接近)" if i == 1 else ""
            print(f"  {i}. 与 [{tpoet}] 的余弦相似度: {sim:.4f}{tag}")

    # ---- 保存结果供 HTML 使用 ----
    import json
    out_path = os.path.join(BASE, "similarity_results.json")
    # JSON 需要普通 dict，且 key 顺序固定
    json_data = {qname: results[qname] for qname in QING_POETS}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
