import json
import re
import random
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional


def parse_anatomy_block(text: str) -> OrderedDict:
    pattern = re.compile(
        r'\s*([A-Za-z][A-Za-z _\-()]*)\s*:\s*(.*?)(?=(?:\n|\s)+[A-Za-z][A-Za-z _\-()]*\s*:|\Z)',
        flags=re.DOTALL
    )
    out = OrderedDict()
    for m in pattern.finditer(text.strip()):
        key = m.group(1).strip()
        val = m.group(2).strip()
        out[key] = val
    return out


def reconstruct_text(od: OrderedDict) -> str:
    parts = [f"{k}: {v}" for k, v in od.items()]
    return " ".join(parts)


def normalize_key(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def choose_alternative(current: str, pool: List[str]) -> Optional[str]:
    candidates = [p for p in pool if p.strip() and p.strip() != current.strip()]
    if not candidates:
        return None
    return random.choice(candidates)


def remix_two_anatomies(
        text: str,
        options_by_anatomy: Dict[str, List[str]],
        n_replace: int = 2,
        seed: Optional[int] = None
) -> Tuple[str, List[Tuple[str, str, str]]]:
    if seed is not None:
        random.seed(seed)

    od = parse_anatomy_block(text)

    norm_to_original = {normalize_key(k): k for k in od.keys()}
    options_norm = {normalize_key(k): v for k, v in options_by_anatomy.items()}

    common_norm_keys = [k for k in norm_to_original.keys() if k in options_norm]
    if not common_norm_keys:
        return text, []

    random.shuffle(common_norm_keys)
    to_change = common_norm_keys[:min(n_replace, len(common_norm_keys))]

    changes: List[Tuple[str, str, str]] = []
    for nk in to_change:
        orig_key = norm_to_original[nk]
        current_desc = od[orig_key]
        alt = choose_alternative(current_desc, options_norm[nk])
        if alt is None:
            continue
        od[orig_key] = alt
        changes.append((orig_key, current_desc, alt))

    return reconstruct_text(od), changes


if __name__ == "__main__":
    with open("/data_backup/Project/Text_transfer/anatomy.json", "r", encoding="utf-8") as f:
        options = json.load(f)

    text = ("trachea: Patent trachea; no obstruction. bronchi: Patent main bronchi; no obstruction. mediastinum: Triangle-shaped density in anterior mediastinum; thymic remnant. heart: Normal cardiac silhouette; no significant pathology. lung, parenchyma: No mass, nodule, or infiltration detected in both lungs. pleura: No pleural effusion or thickening observed in both hemithorax. adrenal glands: No significant pathology in bilateral adrenal glands. abdomen: No significant pathology detected in abdominal sections. bones: No lytic destructive lesions observed.")

    new_text, changed = remix_two_anatomies(text, options, n_replace=2, seed=42)

    print(">>> New text:\n", new_text)
    print("\n>>> Changes:")
    for a, old, new in changed:
        print(f"- {a}\n  OLD: {old}\n  NEW: {new}\n")
