# -*- coding: utf-8 -*-
"""
将钱谦益 vs 唐宋六家的比对结果（余弦相似度排行榜 + 关联词 Fisher 检验）
导出为 xlsx 和 csv 两个文件。
"""

import os
import csv
from collections import Counter
from scipy.stats import fisher_exact
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

# 复用主脚本逻辑
from stylometry_compare import (
    QIAN_FILE,
    TANG_SONG_FILE,
    POET_ALIASES,
    split_poets_by_line,
    extract_1_2_grams,
    _is_cjk,
    build_feature_vector,
    cosine_similarity,
    TOP_N,
)
from association_words import MIN_COUNT, P_THRESHOLD, TOP_K

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(OUTPUT_DIR, "qian_vs_tangsong_results.xlsx")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv_output")
os.makedirs(CSV_DIR, exist_ok=True)


def compute_results():
    """复算全部结果，返回 (similarity_rows, association_rows, char_counts)。"""
    # ---- 钱谦益 ----
    with open(QIAN_FILE, "r", encoding="utf-8", errors="ignore") as f:
        qian_text = f.read()
    qian_counter = extract_1_2_grams(qian_text)
    qian_char_count = sum(1 for c in qian_text if _is_cjk(c))

    # ---- 唐宋六家 ----
    poet_buffers = split_poets_by_line(TANG_SONG_FILE)
    poet_counters = {}
    char_counts = {}
    for poet in POET_ALIASES:
        text = "\n".join(poet_buffers.get(poet, []))
        poet_counters[poet] = extract_1_2_grams(text)
        char_counts[poet] = sum(1 for c in text if _is_cjk(c))

    # ---- 共享词表 + 余弦相似度 ----
    global_counter = Counter()
    global_counter += qian_counter
    for poet in POET_ALIASES:
        global_counter += poet_counters[poet]
    vocab = [w for w, _ in global_counter.most_common(TOP_N)]

    qian_vec = build_feature_vector(qian_counter, vocab)
    sim_rows = []
    for poet in POET_ALIASES:
        vec = build_feature_vector(poet_counters[poet], vocab)
        sim = cosine_similarity(qian_vec, vec)
        sim_rows.append((poet, sim, char_counts[poet]))
    sim_rows.sort(key=lambda x: x[1], reverse=True)

    # ---- 关联词 Fisher 检验 ----
    bai_counter = poet_counters["白居易"]
    other_counter = Counter()
    for poet in POET_ALIASES:
        if poet == "白居易":
            continue
        other_counter += poet_counters[poet]

    candidates = [
        w for w in qian_counter
        if qian_counter[w] >= MIN_COUNT and bai_counter.get(w, 0) >= MIN_COUNT
    ]

    qian_total = sum(qian_counter.values())
    bai_total = sum(bai_counter.values())
    other_total = sum(other_counter.values())

    raw_results = []
    for w in candidates:
        a_q = qian_counter[w]
        _, p_q = fisher_exact(
            [[a_q, qian_total - a_q], [other_counter.get(w, 0), other_total - other_counter.get(w, 0)]],
            alternative="greater",
        )
        a_b = bai_counter[w]
        _, p_b = fisher_exact(
            [[a_b, bai_total - a_b], [other_counter.get(w, 0), other_total - other_counter.get(w, 0)]],
            alternative="greater",
        )
        raw_results.append((w, p_q, p_b, max(p_q, p_b), a_q, a_b, other_counter.get(w, 0)))

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
    for i, (word, p_q, p_b, p_max, cq, cb, co) in enumerate(raw_results):
        if adjusted[i] < P_THRESHOLD:
            assoc_rows.append((word, adjusted[i], p_q, p_b, cq, cb, co))
    assoc_rows = assoc_rows[:TOP_K]

    return sim_rows, assoc_rows, char_counts, qian_char_count


def write_csv(sim_rows, assoc_rows, char_counts, qian_char_count):
    """写出 CSV 文件（每个表一个 csv）。"""
    # 1) 余弦相似度排行榜
    sim_csv = os.path.join(CSV_DIR, "cosine_similarity_ranking.csv")
    with open(sim_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["排名", "诗人", "余弦相似度", "切分字数"])
        for rank, (poet, sim, cc) in enumerate(sim_rows, 1):
            w.writerow([rank, poet, f"{sim:.4f}", cc])

    # 2) 关联词 Fisher 检验
    assoc_csv = os.path.join(CSV_DIR, "association_words_fisher.csv")
    with open(assoc_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["排名", "关联词", "BH校正p值", "钱谦益频次", "白居易频次",
                     "其他诗人频次", "钱谦益原始p值", "白居易原始p值"])
        for rank, (word, adj_p, p_q, p_b, cq, cb, co) in enumerate(assoc_rows, 1):
            w.writerow([rank, word, f"{adj_p:.4f}", cq, cb, co, f"{p_q:.4g}", f"{p_b:.4g}"])

    # 3) 原始字数统计
    char_csv = os.path.join(CSV_DIR, "char_counts.csv")
    with open(char_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["诗人", "切分字数", "特征种类"])
        w.writerow(["钱谦益", qian_char_count, "—"])
        for poet in POET_ALIASES:
            w.writerow([poet, char_counts[poet], "—"])

    print(f"[CSV] 已生成:")
    print(f"  - {sim_csv}")
    print(f"  - {assoc_csv}")
    print(f"  - {char_csv}")


def write_xlsx(sim_rows, assoc_rows, char_counts, qian_char_count):
    """写出 xlsx 文件（多 sheet）。"""
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
                    # 中文字符按 2 宽度估算
                    length = sum(2 if ord(c) > 127 else 1 for c in val)
                    if length > max_len:
                        max_len = length
                except Exception:
                    pass
            ws.column_dimensions[letter].width = min(max_len + 4, 40)

    # ---- Sheet 1: 余弦相似度排行榜 ----
    ws1 = wb.active
    ws1.title = "余弦相似度排行榜"
    headers1 = ["排名", "诗人", "余弦相似度", "切分字数"]
    ws1.append(headers1)
    for rank, (poet, sim, cc) in enumerate(sim_rows, 1):
        ws1.append([rank, poet, round(sim, 4), cc])
    style_header(ws1, len(headers1))
    autosize(ws1)

    # ---- Sheet 2: 关联词 Fisher 检验 ----
    ws2 = wb.create_sheet("关联词Fisher检验")
    headers2 = ["排名", "关联词", "BH校正p值", "钱谦益频次", "白居易频次",
                "其他诗人频次", "钱谦益原始p值", "白居易原始p值"]
    ws2.append(headers2)
    for rank, (word, adj_p, p_q, p_b, cq, cb, co) in enumerate(assoc_rows, 1):
        ws2.append([rank, word, round(adj_p, 4), cq, cb, co, f"{p_q:.4g}", f"{p_b:.4g}"])
    style_header(ws2, len(headers2))
    autosize(ws2)

    # ---- Sheet 3: 原始字数统计 ----
    ws3 = wb.create_sheet("原始字数统计")
    headers3 = ["诗人", "切分字数", "特征种类"]
    ws3.append(headers3)
    ws3.append(["钱谦益", qian_char_count, "—"])
    for poet in POET_ALIASES:
        ws3.append([poet, char_counts[poet], "—"])
    style_header(ws3, len(headers3))
    autosize(ws3)

    wb.save(XLSX_PATH)
    print(f"[XLSX] 已生成: {XLSX_PATH}")


def main():
    print("=" * 60)
    print("导出比对结果为 xlsx + csv")
    print("=" * 60)

    sim_rows, assoc_rows, char_counts, qian_char_count = compute_results()

    write_csv(sim_rows, assoc_rows, char_counts, qian_char_count)
    write_xlsx(sim_rows, assoc_rows, char_counts, qian_char_count)

    print("\n导出完成。")


if __name__ == "__main__":
    main()
