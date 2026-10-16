# -*- coding: utf-8 -*-
"""
钱谦益 vs 唐宋六家 计量文体学比对脚本
四级防御：异体字白名单 / 贪婪聚类 / 绝对安全阀 / 1-2Gram 余弦相似度
纯标准库实现，100% 鲁棒，绝不死锁、绝不崩溃。
"""

import os
import math
from collections import Counter

# ============================================================
# 路径配置（可按需修改）
# ============================================================
QIAN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qian_qianyi.txt")
TANG_SONG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "御选唐宋诗醇.txt")

TOP_N = 300  # 高频特征维度

# ============================================================
# 第一级：繁简 / 异体字 / 代称 白名单（Alias Mapping）
# ============================================================
POET_ALIASES = {
    "李白":   ["李白", "李太白", "太白"],
    "杜甫":   ["杜甫", "子美", "杜工部"],
    "白居易": ["白居易", "乐天", "香山居士"],
    "苏轼":   ["苏轼", "蘇軾", "东坡", "苏东坡", "蘇東坡"],
    "陆游":   ["陆游", "陸游", "务观", "放翁"],
    "辛弃疾": ["辛弃疾", "幼安", "稼轩"],
}

# 归属标签 -> 规范化诗人名（输出用）
# 注意：按"具体度"排序，长别名优先匹配，避免短别名误抢。
def _build_alias_index():
    """构建 (alias -> poet_name) 索引，按 alias 长度降序排列，长串优先命中。"""
    pairs = []
    for poet, aliases in POET_ALIASES.items():
        for a in aliases:
            pairs.append((a, poet))
    # 长别名优先，避免 "太白" 抢在 "李太白" 之前等边界问题
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


ALIAS_INDEX = _build_alias_index()


def detect_poet(line):
    """若该短行命中某诗人代称，返回诗人名；否则返回 None。"""
    for alias, poet in ALIAS_INDEX:
        if alias in line:
            return poet
    return None


# ============================================================
# 第二级 + 第三级：行扫描贪婪聚类 + 绝对安全阀
# ============================================================
def split_poets_by_line(filepath):
    """
    线性扫描归属法：
      - for line in file: 有限迭代，绝不死锁
      - errors='ignore'：容忍任何编码乱码
      - try/except：单行异常强行跳过
      - 短行(<15字)命中诗人代称 -> 切换活动归属标签
      - 其余行无条件累加到当前活动诗人的 buffer
    返回: dict[poet_name -> list[str]]
    """
    buffers = {poet: [] for poet in POET_ALIASES}
    active_poet = None

    # 第三级：绝对安全阀 —— for 循环，迭代次数固定
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    # 去除首尾空白后判断
                    stripped = line.strip()
                    if not stripped:
                        # 空行不切换标签，但也不写入（避免无意义空行膨胀）
                        continue

                    # 第二级：短行 + 命中代称 -> 切换归属
                    if len(stripped) < 15:
                        hit = detect_poet(stripped)
                        if hit is not None:
                            active_poet = hit
                            continue  # 标签行本身不计入正文

                    # 其余正文行：无条件累加到当前活动诗人
                    if active_poet is not None:
                        buffers[active_poet].append(stripped)
                except Exception:
                    # 单行致命乱码：强行跳过，绝不中断整体流程
                    pass
    except Exception as e:
        # 文件级异常也兜底（理论上不会触发，但防御到底）
        print(f"[警告] 读取文件出现异常: {e}")

    return buffers


# ============================================================
# 第四级：1-2 Gram 字频特征 + 余弦相似度
# ============================================================

# 需要过滤的文言虚词（单字）
FUNCTION_WORDS = set("之乎者也而以其矣焉哉于乃则所此皆为以")

# 中英文标点 + 数字 + 常见符号（保留汉字）
# 采用"白名单"思路：只保留 CJK 统一表意文字，其余一律剔除，最鲁棒。
def _is_cjk(ch):
    """判断是否为 CJK 统一表意文字（汉字），排除标点、数字、字母、符号。"""
    code = ord(ch)
    # CJK 统一表意文字基本区 + 扩展A + 兼容汉字
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def extract_1_2_grams(text):
    """
    从纯文本中提取 1-gram（单字）与 2-gram（相邻双字）。
    过滤：标点、数字、换行、空格、文言虚词。
    返回: Counter
    """
    # 1) 清洗：只保留汉字序列（自动剔除所有标点/数字/空白/乱码符号）
    chars = [c for c in text if _is_cjk(c)]
    # 2) 进一步剔除文言虚词
    chars = [c for c in chars if c not in FUNCTION_WORDS]

    counter = Counter()

    # 1-gram：单字
    for c in chars:
        counter[c] += 1

    # 2-gram：相邻双字（在已过滤的序列上取相邻对，保证"春风"类成词）
    for i in range(len(chars) - 1):
        bigram = chars[i] + chars[i + 1]
        counter[bigram] += 1

    return counter


def build_feature_vector(counter, vocab):
    """根据共享词表 vocab，构建相对词频向量（list[float]）。"""
    total = sum(counter.values())
    if total == 0:
        return [0.0] * len(vocab)
    return [counter.get(w, 0) / total for w in vocab]


def cosine_similarity(vec_a, vec_b):
    """
    标准余弦相似度: (A·B) / (||A|| * ||B||)
    向量已通过相对词频归一化，消除字数差异误差。
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("钱谦益 诗风唐宋归属度比对（1-2 Gram 字频 + 余弦相似度）")
    print("=" * 60)

    # ---- 读取钱谦益 ----
    qian_text = ""
    try:
        with open(QIAN_FILE, "r", encoding="utf-8", errors="ignore") as f:
            qian_text = f.read()
    except Exception as e:
        print(f"[错误] 无法读取钱谦益文件: {e}")
        return

    qian_counter = extract_1_2_grams(qian_text)
    qian_char_count = sum(1 for c in qian_text if _is_cjk(c))
    print(f"\n[钱谦益] 切分字数: {qian_char_count} 字，特征种类: {len(qian_counter)}")

    # ---- 切分唐宋六家 ----
    poet_buffers = split_poets_by_line(TANG_SONG_FILE)

    poet_counters = {}
    for poet in POET_ALIASES:
        text = "\n".join(poet_buffers.get(poet, []))
        cnt = extract_1_2_grams(text)
        poet_counters[poet] = cnt
        char_count = sum(1 for c in text if _is_cjk(c))
        print(f"[{poet}] 切分字数: {char_count} 字，特征种类: {len(cnt)}")

    # ---- 构建共享词表：所有诗人(含钱谦益)合并后的全局 top_n 高频特征 ----
    global_counter = Counter()
    global_counter += qian_counter
    for poet in POET_ALIASES:
        global_counter += poet_counters[poet]
    vocab = [w for w, _ in global_counter.most_common(TOP_N)]
    print(f"\n[共享特征空间] 全局 Top-{TOP_N} 特征维度已构建（共 {len(vocab)} 维）")

    # ---- 构建向量 ----
    qian_vec = build_feature_vector(qian_counter, vocab)
    poet_vectors = {
        poet: build_feature_vector(poet_counters[poet], vocab)
        for poet in POET_ALIASES
    }

    # ---- 计算余弦相似度 ----
    results = []
    for poet in POET_ALIASES:
        sim = cosine_similarity(qian_vec, poet_vectors[poet])
        results.append((poet, sim))

    # ---- 学术排行榜输出 ----
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("[钱谦益 诗风唐宋归属度排行榜]")
    print("=" * 60)
    for rank, (poet, sim) in enumerate(results, 1):
        tag = " (最接近)" if rank == 1 else ""
        print(f"{rank}. 与 [{poet}] 的余弦相似度: {sim:.4f}{tag}")
    print("=" * 60)


if __name__ == "__main__":
    main()
