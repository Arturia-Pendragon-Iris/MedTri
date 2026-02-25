import re
from typing import Dict, List, Tuple, Optional

def levenshtein_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    # 滚动数组优化空间
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,       # 删除
                curr[j-1] + 1,     # 插入
                prev[j-1] + cost   # 代价
            ))
        prev = curr
    return prev[-1]

def levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b: return 1.0
    d = levenshtein_distance(a.lower(), b.lower())
    return 1.0 - d / max(len(a), len(b))

def annotate_text_with_levenshtein(
    text: str,
    disease_descriptions: Dict[str, str],
    threshold: float = 0.85,
    once_per_disease: bool = False,
    max_ngram_words: Optional[int] = None
) -> str:

    disease_keys = list(disease_descriptions.keys())
    disease_keys_lower = [k.lower() for k in disease_keys]
    disease_word_lens = [len(k.split()) for k in disease_keys_lower]
    if max_ngram_words is None:
        max_ngram_words = max(disease_word_lens)

    word_iter = list(re.finditer(r"\b[\w\-]+\b", text))
    words = [m.group(0) for m in word_iter]
    spans: List[Tuple[int, int]] = [m.span() for m in word_iter]

    matches: List[Tuple[int, int, str]] = []  # (start_idx, end_idx, disease_key)
    used_keys = set()

    W = len(words)
    for i in range(W):
        for n in range(1, max_ngram_words + 1):
            j = i + n
            if j > W: break
            candidate = " ".join(words[i:j]).lower()

            best_key = None
            best_sim = 0.0
            for k_lower, k_orig in zip(disease_keys_lower, disease_keys):
                if abs(len(candidate) - len(k_lower)) > max(6, int(0.6*len(k_lower))):
                    continue
                sim = levenshtein_ratio(candidate, k_lower)
                if sim > best_sim:
                    best_sim = sim
                    best_key = k_orig

            if best_key and best_sim >= threshold:
                if once_per_disease and best_key in used_keys:
                    continue
                used_keys.add(best_key)
                start_char = spans[i][0]
                end_char = spans[j-1][1]
                matches.append((start_char, end_char, best_key))

    matches.sort(key=lambda x: (x[0], -x[1]))
    filtered = []
    last_end = -1
    for s, e, k in matches:
        if s >= last_end:
            filtered.append((s, e, k))
            last_end = e
        else:
            if filtered and (e - s) > (filtered[-1][1] - filtered[-1][0]):
                filtered[-1] = (s, e, k)
                last_end = e

    out = text
    for s, e, k in sorted(filtered, key=lambda x: x[1], reverse=True):
        desc = disease_descriptions.get(k, "").strip()
        if not desc:
            continue
        out = out[:e] + f" [{desc}]" + out[e:]

    return out


if __name__ == "__main__":
    import json
    with open("/data_backup/Project/Text_transfer/diseases_description.json", "r", encoding="utf-8") as f:
        disease_descriptions = json.load(f)

    text = "Findings: cardiomegaly and mild edema. Possible atelectasis in right base; no pneumothorax."
    print(annotate_text_with_levenshtein(text, disease_descriptions, threshold=0.86, once_per_disease=False))
