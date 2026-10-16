# -*- coding: utf-8 -*-
"""
吴伟业 vs 唐宋五家 计量文体学比对脚本
（去除辛弃疾，因《唐宋诗醇》未收录）
复用 stylometry_compare.py 的四级防御框架。
"""

import os
import math
from collections import Counter
from scipy.stats import fisher_exact
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
import csv

# 复用核心逻辑
from stylometry_compare import (
    TANG_SONG_FILE,
    split_poets_by_line,
    extract_1_2_grams,
    _is_cjk,
    build_feature_vector,
    cosine_similarity,
)

# ============================================================
# 配置
# ============================================================
WU_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wu_weiye.txt")

# 去除辛弃疾，仅保留《唐宋诗醇》实际收录的五家
TARGET_POETS = ["李白", "杜甫", "白居易", "苏轼", "陆游"]

TOP_N = 300
TOP_K = 20
MIN_COUNT = 3
P_THRESHOLD = 0.05

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(OUTPUT_DIR, "wu_vs_tangsong_results.xlsx")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv_output_wu")
os.makedirs(CSV_DIR, exist_ok=True)


def compute_results():
    """复算全部结果。"""
    # ---- 吴伟业 ----
    with open(WU_FILE, "r", encoding="utf-8", errors="ignore") as f:
        wu_text = f.read()
    wu_counter = extract_1_2_grams(wu_text)
    wu_char_count = sum(1 for c in wu_text if _is_cjk(c))

    # ---- 唐宋五家（复用切分，但只用 TARGET_POETS）----
    poet_buffers = split_poets_by_line(TANG_SONG_FILE)
    poet_counters = {}
    char_counts = {}
    for poet in TARGET_POETS:
        text = "\n".join(poet_buffers.get(poet, []))
        poet_counters[poet] = extract_1_2_grams(text)
        char_counts[poet] = sum(1 for c in text if _is_cjk(c))

    # ---- 共享词表 + 余弦相似度 ----
    global_counter = Counter()
    global_counter += wu_counter
    for poet in TARGET_POETS:
        global_counter += poet_counters[poet]
    vocab = [w for w, _ in global_counter.most_common(TOP_N)]

    wu_vec = build_feature_vector(wu_counter, vocab)
    sim_rows = []
    for poet in TARGET_POETS:
        vec = build_feature_vector(poet_counters[poet], vocab)
        sim = cosine_similarity(wu_vec, vec)
        sim_rows.append((poet, sim, char_counts[poet]))
    sim_rows.sort(key=lambda x: x[1], reverse=True)

    # ---- 找出与吴伟业最相似的诗人，做关联词 Fisher 检验 ----
    closest_poet = sim_rows[0][0]
    closest_counter = poet_counters[closest_poet]

    other_counter = Counter()
    for poet in TARGET_POETS:
        if poet == closest_poet:
            continue
        other_counter += poet_counters[poet]

    candidates = [
        w for w in wu_counter
        if wu_counter[w] >= MIN_COUNT and closest_counter.get(w, 0) >= MIN_COUNT
    ]

    wu_total = sum(wu_counter.values())
    closest_total = sum(closest_counter.values())
    other_total = sum(other_counter.values())

    raw_results = []
    for w in candidates:
        a_w = wu_counter[w]
        _, p_w = fisher_exact(
            [[a_w, wu_total - a_w], [other_counter.get(w, 0), other_total - other_counter.get(w, 0)]],
            alternative="greater",
        )
        a_c = closest_counter[w]
        _, p_c = fisher_exact(
            [[a_c, closest_total - a_c], [other_counter.get(w, 0), other_total - other_counter.get(w, 0)]],
            alternative="greater",
        )
        raw_results.append((w, p_w, p_c, max(p_w, p_c), a_w, a_c, other_counter.get(w, 0)))

    raw_results.sort(key=lambda x: x[3])
    p_values = [r[3] for r in raw_results]
    n = len(p_values)
    adjusted = [0.0] * n
    sorted_idx = sorted(range(n), key=lambda i: p_values[i], reverse=True)
    prev = 1.0
    for rank, idx in enumerate(sorted_idx, 1):
        bh = p_values[idx] * n / rank
        if bh > 1.0:
            bh = 1.0
        if bh < prev:
            prev = bh
        adjusted[idx] = prev

    assoc_rows = []
    for i, (word, p_w, p_c, p_max, cw, cc, co) in enumerate(raw_results):
        if adjusted[i] < P_THRESHOLD:
            assoc_rows.append((word, adjusted[i], p_w, p_c, cw, cc, co))
    assoc_rows = assoc_rows[:TOP_K]

    return sim_rows, assoc_rows, char_counts, wu_char_count, closest_poet


def write_csv(sim_rows, assoc_rows, char_counts, wu_char_count, closest_poet):
    """写出 CSV 文件。"""
    sim_csv = os.path.join(CSV_DIR, "cosine_similarity_ranking.csv")
    with open(sim_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["排名", "诗人", "余弦相似度", "切分字数"])
        for rank, (poet, sim, cc) in enumerate(sim_rows, 1):
            w.writerow([rank, poet, f"{sim:.4f}", cc])

    assoc_csv = os.path.join(CSV_DIR, f"association_words_fisher_{closest_poet}.csv")
    with open(assoc_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["排名", "关联词", "BH校正p值", "吴伟业频次", f"{closest_poet}频次",
                     "其他诗人频次", "吴伟业原始p值", f"{closest_poet}原始p值"])
        for rank, (word, adj_p, p_w, p_c, cw, cc, co) in enumerate(assoc_rows, 1):
            w.writerow([rank, word, f"{adj_p:.4f}", cw, cc, co, f"{p_w:.4g}", f"{p_c:.4g}"])

    char_csv = os.path.join(CSV_DIR, "char_counts.csv")
    with open(char_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["诗人", "切分字数"])
        w.writerow(["吴伟业", wu_char_count])
        for poet in TARGET_POETS:
            w.writerow([poet, char_counts[poet]])

    print(f"[CSV] 已生成于 {CSV_DIR}/")
    print(f"  - cosine_similarity_ranking.csv")
    print(f"  - association_words_fisher_{closest_poet}.csv")
    print(f"  - char_counts.csv")


def write_xlsx(sim_rows, assoc_rows, char_counts, wu_char_count, closest_poet):
    """写出 xlsx 文件。"""
    wb = Workbook()
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    center = Alignment(horizontal="center", vertical="center")

    def style_header(ws, ncols):
        for col in range(1, ncols + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center

    def autosize(ws):
        for col in ws.columns:
            max_len = 0
            letter = get_column_letter(col[0].column)
            for cell in col:
                try:
                    val = str(cell.value) if cell.value is not None else ""
                    length = sum(2 if ord(c) > 127 else 1 for c in val)
                    if length > max_len:
                        max_len = length
                except Exception:
                    pass
            ws.column_dimensions[letter].width = min(max_len + 4, 40)

    # Sheet 1: 余弦相似度排行榜
    ws1 = wb.active
    ws1.title = "余弦相似度排行榜"
    headers1 = ["排名", "诗人", "余弦相似度", "切分字数"]
    ws1.append(headers1)
    for rank, (poet, sim, cc) in enumerate(sim_rows, 1):
        ws1.append([rank, poet, round(sim, 4), cc])
    style_header(ws1, len(headers1))
    autosize(ws1)

    # Sheet 2: 关联词 Fisher 检验
    ws2 = wb.create_sheet(f"关联词Fisher检验({closest_poet})")
    headers2 = ["排名", "关联词", "BH校正p值", "吴伟业频次", f"{closest_poet}频次",
                "其他诗人频次", "吴伟业原始p值", f"{closest_poet}原始p值"]
    ws2.append(headers2)
    for rank, (word, adj_p, p_w, p_c, cw, cc, co) in enumerate(assoc_rows, 1):
        ws2.append([rank, word, round(adj_p, 4), cw, cc, co, f"{p_w:.4g}", f"{p_c:.4g}"])
    style_header(ws2, len(headers2))
    autosize(ws2)

    # Sheet 3: 原始字数统计
    ws3 = wb.create_sheet("原始字数统计")
    headers3 = ["诗人", "切分字数"]
    ws3.append(headers3)
    ws3.append(["吴伟业", wu_char_count])
    for poet in TARGET_POETS:
        ws3.append([poet, char_counts[poet]])
    style_header(ws3, len(headers3))
    autosize(ws3)

    wb.save(XLSX_PATH)
    print(f"[XLSX] 已生成: {XLSX_PATH}")


def main():
    print("=" * 64)
    print("吴伟业 诗风唐宋归属度比对（1-2 Gram 字频 + 余弦相似度）")
    print("（已去除辛弃疾，《唐宋诗醇》未收录）")
    print("=" * 64)

    sim_rows, assoc_rows, char_counts, wu_char_count, closest_poet = compute_results()

    # ---- 终端输出 ----
    print(f"\n[吴伟业] 切分字数: {wu_char_count} 字")
    for poet in TARGET_POETS:
        print(f"[{poet}] 切分字数: {char_counts[poet]} 字")

    print("\n" + "=" * 64)
    print("[吴伟业 诗风唐宋归属度排行榜]")
    print("=" * 64)
    for rank, (poet, sim, cc) in enumerate(sim_rows, 1):
        tag = " (最接近)" if rank == 1 else ""
        print(f"{rank}. 与 [{poet}] 的余弦相似度: {sim:.4f}{tag}")

    print(f"\n最相似诗人: {closest_poet}")
    print(f"候选关联词数: {len(assoc_rows)}（BH校正后 p < {P_THRESHOLD}）")

    print("\n" + "=" * 64)
    print(f"[吴伟业 × {closest_poet} 关联词排行榜]")
    print("=" * 64)
    print(f"{'排名':<4} {'关联词':<8} {'BH校p':<10} {'吴频':<6} {f'{closest_poet}频':<6} {'他频':<6}")
    print("-" * 64)
    for rank, (word, adj_p, p_w, p_c, cw, cc, co) in enumerate(assoc_rows, 1):
        print(f"{rank:<4} {word:<8} {adj_p:<10.4f} {cw:<6} {cc:<6} {co:<6}")
    print("=" * 64)

    # ---- 导出文件 ----
    print()
    write_csv(sim_rows, assoc_rows, char_counts, wu_char_count, closest_poet)
    write_xlsx(sim_rows, assoc_rows, char_counts, wu_char_count, closest_poet)
    print("\n导出完成。")


if __name__ == "__main__":
    main()
