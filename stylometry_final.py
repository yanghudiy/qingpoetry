# -*- coding: utf-8 -*-
"""
钱谦益 & 吴伟业 vs 唐宋六家 计量文体学比对脚本（修正版）
核心修正：
  1. 仅当行匹配"[诗人名]诗[序号]"模式时才切换归属（杜绝诗题误触发）
  2. 补全异体字：蘓轼、睂山、山隂
  3. for line in file 有限迭代，errors='ignore'，绝不死锁
纯标准库，极简实现。
"""

import os
import math
from collections import Counter

# ============================================================
# 路径
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
QIAN_FILE = os.path.join(BASE, "qian_qianyi.txt")
WU_FILE = os.path.join(BASE, "wu_weiye.txt")
TANG_SONG_FILE = os.path.join(BASE, "御选唐宋诗醇.txt")

TOP_N = 300

# ============================================================
# 诗人白名单（含异体字/代称）
# ============================================================
POET_ALIASES = {
    "李白":   ["李白", "李太白", "太白"],
    "杜甫":   ["杜甫", "子美", "杜工部"],
    "白居易": ["白居易", "乐天", "香山居士"],
    "韩愈":   ["韩愈", "退之", "韩昌黎", "昌黎"],
    "苏轼":   ["苏轼", "蘇轼", "蘓轼", "东坡", "苏东坡", "蘇東坡"],
    "陆游":   ["陆游", "陸游", "务观", "放翁"],
}

# 中文数字（用于识别"诗一""诗二"等）
CN_NUMS = set("一二三四五六七八九十")


def detect_section_header(line):
    """
    仅当该行是真正的分卷标题时返回诗人名，否则返回 None。
    真正的分卷标题模式：包含诗人代称 + "诗" + 中文数字，且行长度 < 20。
    例如："陇西李白诗一"、"眉山蘓轼诗三"、"山隂陆游诗一"
    这样"赠李白""梦李白二首"等诗题不会误触发。
    """
    if len(line) >= 20:
        return None
    # 必须包含 "诗" + 紧跟一个中文数字
    for i in range(len(line) - 1):
        if line[i] == "诗" and line[i + 1] in CN_NUMS:
            # 再检查是否包含某诗人代称
            for alias, poet in _ALIAS_PAIRS:
                if alias in line:
                    return poet
            return None
    return None


# 构建 (alias -> poet) 索引，长别名优先
_ALIAS_PAIRS = []
for _poet, _aliases in POET_ALIASES.items():
    for _a in _aliases:
        _ALIAS_PAIRS.append((_a, _poet))
_ALIAS_PAIRS.sort(key=lambda x: len(x[0]), reverse=True)


# ============================================================
# 行扫描切分（for 循环，绝不死锁）
# ============================================================
def split_poets(filepath):
    """线性扫描，仅按分卷标题切换归属，正文行无条件累加。"""
    buffers = {p: [] for p in POET_ALIASES}
    current = None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                s = line.strip()
                if not s:
                    continue
                hit = detect_section_header(s)
                if hit is not None:
                    current = hit
                    continue
                if current is not None:
                    buffers[current].append(s)
            except Exception:
                pass
    return buffers


# ============================================================
# 1-2 Gram 字频
# ============================================================
FUNCTION_WORDS = set("之乎者也而以其矣焉哉于乃则所此皆为以")


def _is_cjk(ch):
    c = ord(ch)
    return 0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF or 0xF900 <= c <= 0xFAFF


def extract_1_2_grams(text):
    chars = [c for c in text if _is_cjk(c) and c not in FUNCTION_WORDS]
    counter = Counter()
    for c in chars:
        counter[c] += 1
    for i in range(len(chars) - 1):
        counter[chars[i] + chars[i + 1]] += 1
    return counter


def build_vector(counter, vocab):
    total = sum(counter.values())
    if total == 0:
        return [0.0] * len(vocab)
    return [counter.get(w, 0) / total for w in vocab]


def cosine(a, b):
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("钱谦益 & 吴伟业 vs 唐宋六家 余弦相似度比对（修正版）")
    print("=" * 60)

    # ---- 读取清代诗人 ----
    qian_text = open(QIAN_FILE, encoding="utf-8", errors="ignore").read()
    wu_text = open(WU_FILE, encoding="utf-8", errors="ignore").read()
    qian_cnt = extract_1_2_grams(qian_text)
    wu_cnt = extract_1_2_grams(wu_text)

    qian_chars = sum(1 for c in qian_text if _is_cjk(c))
    wu_chars = sum(1 for c in wu_text if _is_cjk(c))

    # ---- 切分唐宋六家 ----
    buffers = split_poets(TANG_SONG_FILE)
    poet_cnts = {}
    poet_chars = {}
    for poet in POET_ALIASES:
        text = "\n".join(buffers[poet])
        poet_cnts[poet] = extract_1_2_grams(text)
        poet_chars[poet] = sum(1 for c in text if _is_cjk(c))

    # ---- 打印字数核对 ----
    print(f"\n[钱谦益] {qian_chars} 字")
    print(f"[吴伟业] {wu_chars} 字")
    for poet in POET_ALIASES:
        print(f"[{poet}] {poet_chars[poet]} 字")

    # ---- 构建共享词表 ----
    global_cnt = Counter()
    global_cnt += qian_cnt
    global_cnt += wu_cnt
    for p in POET_ALIASES:
        global_cnt += poet_cnts[p]
    vocab = [w for w, _ in global_cnt.most_common(TOP_N)]

    qian_vec = build_vector(qian_cnt, vocab)
    wu_vec = build_vector(wu_cnt, vocab)

    # ---- 计算相似度 ----
    qian_sims = [(p, cosine(qian_vec, build_vector(poet_cnts[p], vocab))) for p in POET_ALIASES]
    wu_sims = [(p, cosine(wu_vec, build_vector(poet_cnts[p], vocab))) for p in POET_ALIASES]
    qian_sims.sort(key=lambda x: x[1], reverse=True)
    wu_sims.sort(key=lambda x: x[1], reverse=True)

    # ---- 输出 ----
    print("\n" + "=" * 60)
    print("[钱谦益 诗风唐宋归属度排行榜]")
    print("=" * 60)
    for i, (p, s) in enumerate(qian_sims, 1):
        print(f"{i}. 与 [{p}] 的余弦相似度: {s:.4f}{' (最接近)' if i == 1 else ''}")

    print("\n" + "=" * 60)
    print("[吴伟业 诗风唐宋归属度排行榜]")
    print("=" * 60)
    for i, (p, s) in enumerate(wu_sims, 1):
        print(f"{i}. 与 [{p}] 的余弦相似度: {s:.4f}{' (最接近)' if i == 1 else ''}")
    print("=" * 60)


if __name__ == "__main__":
    main()
