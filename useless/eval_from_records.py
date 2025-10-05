# eval_from_records.py
# 评测：用 seeds 作为模型预测来对比 gold（忽略 ranking/predictions）
# 指标：P@10, P@100, R@100, Avg Rank（未命中=seed_count+1）
# 输出：宏平均 + 每疾病详情（含完整 seeds）

import os
import re
import json
import csv
import argparse
from typing import List, Dict, Any, Tuple, Set

# ========= 归一化 & 同义词 =========

SALT_OR_FORM_WORDS = {
    "sodium","potassium","hydrochloride","hydrobromide","sulfate","phosphate",
    "acetate","tartrate","mesylate","citrate","maleate","succinate","nitrate",
    "gel","pill","capsule","capsules","infusion","dpi","ointment","solution",
    "drops","eye","oral","product","placebo","tablet","tablets","injection",
    "spray","cream","patch","extended","release","er","xr","sr","mg","ug","µg"
}
PUNCT_TO_SPACE = re.compile(r"[®™\-\_/\\,;:()\[\]]+")
MULTISPACE = re.compile(r"\s+")

def normalize_name(s: str) -> str:
    if not s:
        return ""
    x = s.lower()
    x = x.replace("+", " + ")
    x = PUNCT_TO_SPACE.sub(" ", x)
    toks = [t for t in MULTISPACE.sub(" ", x).strip().split(" ") if t]
    toks = [t for t in toks if t not in SALT_OR_FORM_WORDS]
    x = " ".join(toks)
    x = MULTISPACE.sub(" ", x).strip()
    return x

# 同义组（可扩）：互相等价
ALIAS_GROUPS = [
    {"nuedexta", "dextromethorphan + quinidine", "dextromethorphan quinidine"},
    {"growth hormone", "somatropin"},
    {"omega 3 fatty acids", "epa dha", "epanova"},
    {"ursodiol", "actigall"},
    {"aztreonam lysine", "aztreonam for inhalation", "azli"},
    {"ciprofloxacin dpi", "ciprofloxacin inhalation powder", "ciprofloxacin dry powder inhaler"},
]

def _alias_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for group in ALIAS_GROUPS:
        rep = sorted(group, key=lambda z: (len(z), z))[0]
        for g in group:
            m[normalize_name(g)] = rep
    return m

ALIAS_MAP = _alias_map()

def canonical(s: str) -> str:
    n = normalize_name(s)
    return ALIAS_MAP.get(n, n)

def is_match(a: str, b: str) -> bool:
    ca, cb = canonical(a), canonical(b)
    if ca == cb:
        return True
    if ca in cb or cb in ca:
        return True
    ta, tb = set(ca.split()), set(cb.split())
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    jacc = inter / union if union else 0.0
    return jacc >= 0.6

# ========= 指标 =========

def precision_at_k(preds: List[str], golds: List[str], K: int) -> float:
    use = preds[:K]
    matched_gold: Set[int] = set()
    for p in use:
        for gi, g in enumerate(golds):
            if gi in matched_gold:
                continue
            if is_match(p, g):
                matched_gold.add(gi)
                break
    return len(matched_gold) / K if K > 0 else 0.0

def recall_at_k(preds: List[str], golds: List[str], K: int) -> float:
    if not golds:
        return 0.0
    use = preds[:K]
    matched_gold: Set[int] = set()
    for p in use:
        for gi, g in enumerate(golds):
            if gi in matched_gold:
                continue
            if is_match(p, g):
                matched_gold.add(gi)
                break
    return len(matched_gold) / len(golds)

def avg_rank_with_seed_cap(preds: List[str], golds: List[str], seed_count: int) -> float:
    """
    Avg Rank：对每个 gold，找到首个命中的预测名次（1-based）。
    未命中 -> 记为 seed_count + 1（比如 seeds=12，未命中=13）。
    """
    if not golds:
        return 0.0
    not_found_rank = max(0, seed_count) + 1
    ranks = []
    for g in golds:
        r = not_found_rank
        for i, p in enumerate(preds):
            if is_match(p, g):
                r = i + 1
                break
        ranks.append(r)
    return sum(ranks) / len(ranks)

# ========= 读取与评估（只用 seeds） =========

def load_records(jsonl_path: str) -> List[Dict[str, Any]]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items

def evaluate_records(records: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    per_dis = []
    p10s, p100s, r10s ,r100s, avgrs = [], [],[], [], []

    for rec in records:
        disease = rec.get("disease","")
        golds: List[str] = rec.get("gold_drugs", []) or []
        seeds_raw: List[str] = rec.get("seeds", []) or []

        # 评测用的“预测”= seeds（保持原始顺序）
        preds = [s for s in seeds_raw if isinstance(s, str)]
        seed_count = len(preds)

        p10  = precision_at_k(preds, golds, 10)
        p100 = precision_at_k(preds, golds, 100)
        r10 = recall_at_k(preds, golds, 10)
        r100 = recall_at_k(preds, golds, 100)
        ar   = avg_rank_with_seed_cap(preds, golds, seed_count)

        per_dis.append({
            "disease": disease,
            "num_gold": len(golds),
            "num_preds(seeds)": seed_count,
            "P@10": round(p10, 4),
            "P@100": round(p100, 4),
            "R@10": round(r10, 4),
            "R@100": round(r100, 4),
            "AvgRank": round(ar, 3),
            "seeds": seeds_raw,  # 全部种子
            "gold": golds
        })

        p10s.append(p10)
        p100s.append(p100)
        r10s.append(r10)
        r100s.append(r100)
        avgrs.append(ar)

    macro = {
        "macro_P@10": round(sum(p10s)/len(p10s), 4) if p10s else 0.0,
        "macro_P@100": round(sum(p100s)/len(p100s), 4) if p100s else 0.0,
        "macro_R@10": round(sum(r10s)/len(r10s), 4) if r10s else 0.0,
        "macro_R@100": round(sum(r100s)/len(r100s), 4) if r100s else 0.0,
        "macro_AvgRank": round(sum(avgrs)/len(avgrs), 3) if avgrs else 0.0,
        "num_diseases": len(per_dis),
    }
    return macro, per_dis

def save_per_disease_csv(per_dis: List[Dict[str,Any]], path: str):
    keys = ["disease","num_gold","num_preds(seeds)","P@10","P@100","R@10","R@100","AvgRank","gold","seeds"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_dis:
            r = dict(row)
            r["gold"]  = "; ".join(r.get("gold", []))
            r["seeds"] = "; ".join(r.get("seeds", []))
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=str, default=r"runs_ablation\Debate-single\indication\_records.jsonl", help="path to indication/_records.jsonl")
    ap.add_argument("--out_dir", type=str, default="eval_out_Debate-single_indication")
    ap.add_argument("--save_csv", action="store_true",default=True)
    ap.add_argument("--save_json", action="store_true",default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_records(args.records)
    macro, per_dis = evaluate_records(records)

    print("=== Macro (seeds as predictions) ===")
    for k,v in macro.items():
        print(f"{k}: {v}")

    if args.save_csv:
        save_per_disease_csv(per_dis, os.path.join(args.out_dir, "per_disease.csv"))
        print(f"[saved] {os.path.join(args.out_dir, 'per_disease.csv')}")

    if args.save_json:
        with open(os.path.join(args.out_dir, "macro.json"), "w", encoding="utf-8") as f:
            json.dump(macro, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out_dir, "per_disease.json"), "w", encoding="utf-8") as f:
            json.dump(per_dis, f, ensure_ascii=False, indent=2)
        print(f"[saved] macro/per_disease JSON in {args.out_dir}")

if __name__ == "__main__":
    main()
