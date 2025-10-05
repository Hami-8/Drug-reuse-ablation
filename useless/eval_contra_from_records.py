# eval_contra_from_records.py
# 评测：contraindication —— 用 _records.jsonl 的 seeds 作为预测，
# 与 filtered_contraindication.csv 的 normalized_drug 作为金标准比对。
# 指标：P@10 / P@100 / R@100 / AvgRank（未命中=seed_count+1）
# 输出：宏平均 + 每疾病详情（含完整 seeds）

import os
import re
import csv
import json
import argparse
from typing import List, Dict, Any, Tuple, Set

# ================== 归一化与同义处理 ==================

SALT_OR_FORM_WORDS = {
    "sodium","potassium","hydrochloride","hydrobromide","sulfate","phosphate",
    "acetate","tartrate","mesylate","citrate","maleate","succinate","nitrate",
    "gel","pill","capsule","capsules","infusion","dpi","ointment","solution",
    "drops","eye","oral","product","placebo","tablet","tablets","injection",
    "spray","cream","patch","extended","release","er","xr","sr","mg","ug","µg",
    "soft","capsules","softgel","intravenous","intranasal","nasal","topical"
}
PUNCT_TO_SPACE = re.compile(r"[®™\-\_/\\,;:()\[\]%]+")
MULTISPACE = re.compile(r"\s+")

def normalize_name(s: str) -> str:
    if not s: return ""
    x = s.lower()
    x = x.replace("+", " + ")
    x = PUNCT_TO_SPACE.sub(" ", x)
    toks = [t for t in MULTISPACE.sub(" ", x).strip().split() if t]
    toks = [t for t in toks if t not in SALT_OR_FORM_WORDS]
    x = " ".join(toks)
    x = MULTISPACE.sub(" ", x).strip()
    return x

# 可按需扩展
ALIAS_GROUPS = [
    # 通用等价
    {"sodium valproate","valproate","valproic acid"},
    {"ursodiol","actigall"},
    {"aztreonam lysine","aztreonam for inhalation","azli"},
    {"omega 3 fatty acids","omega 3 acid ethyl esters","epa dha","fish oil","lovaza","ethyl icosapentate"},
]

def _alias_map() -> Dict[str,str]:
    m: Dict[str,str] = {}
    for g in ALIAS_GROUPS:
        rep = sorted(g, key=lambda z:(len(z), z))[0]
        for item in g:
            m[normalize_name(item)] = rep
    return m

ALIAS_MAP = _alias_map()

def canonical(s: str) -> str:
    n = normalize_name(s)
    return ALIAS_MAP.get(n, n)

def is_match(a: str, b: str) -> bool:
    ca, cb = canonical(a), canonical(b)
    if ca == cb: return True
    if not ca or not cb: return False
    if ca in cb or cb in ca:  # 子串宽松
        return True
    ta, tb = set(ca.split()), set(cb.split())
    if not ta or not tb: return False
    jacc = len(ta & tb) / len(ta | tb)
    return jacc >= 0.6

# ================== 指标 ==================

def precision_at_k(preds: List[str], golds: List[str], K: int) -> float:
    use = preds[:K]
    matched: Set[int] = set()
    for p in use:
        for gi, g in enumerate(golds):
            if gi in matched: continue
            if is_match(p, g):
                matched.add(gi)
                break
    return len(matched) / K if K>0 else 0.0

def recall_at_k(preds: List[str], golds: List[str], K: int) -> float:
    if not golds: return 0.0
    use = preds[:K]
    matched: Set[int] = set()
    for p in use:
        for gi, g in enumerate(golds):
            if gi in matched: continue
            if is_match(p, g):
                matched.add(gi)
                break
    return len(matched) / len(golds) if golds else 0.0

def avg_rank_with_cap(preds: List[str], golds: List[str], seed_count: int) -> float:
    if not golds: return 0.0
    not_found = seed_count + 1
    ranks = []
    for g in golds:
        r = not_found
        for i, p in enumerate(preds):
            if is_match(p, g):
                r = i+1
                break
        ranks.append(r)
    return sum(ranks)/len(ranks)

# ================== 数据读取 ==================

def load_contra_gold(csv_path: str) -> Dict[str, List[str]]:
    """
    从 filtered_contraindication.csv 读取金标准：
    - 按 disease 聚合所有 normalized_drug（去重）
    - 返回键为规范化后的 disease 名
    """
    d2g: Dict[str, List[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dis = row.get("disease","").strip()
            gold = row.get("normalized_drug","").strip()
            if not dis or not gold: continue
            key = normalize_name(dis)
            d2g.setdefault(key, [])
            if gold not in d2g[key]:
                d2g[key].append(gold)
    return d2g

def load_records(jsonl_path: str) -> List[Dict[str,Any]]:
    out = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

# 简单疾病名兜底匹配（若找不到 exact key）
def find_gold_for_disease(d2g: Dict[str,List[str]], disease_raw: str) -> List[str]:
    key = normalize_name(disease_raw)
    if key in d2g:
        return d2g[key]
    # 兜底：token Jaccard>0.8
    toks = set(key.split())
    best_key, best_sim = None, 0.0
    for k in d2g.keys():
        tk = set(k.split())
        if not toks or not tk: continue
        sim = len(toks & tk)/len(toks | tk)
        if sim > best_sim:
            best_sim = sim
            best_key = k
    if best_key and best_sim >= 0.8:
        return d2g[best_key]
    return []

# ================== 评测（用 seeds 当预测） ==================

def evaluate(records_path: str, gold_csv: str, out_dir: str, save_json: bool=True, save_csv: bool=True):
    os.makedirs(out_dir, exist_ok=True)
    d2g = load_contra_gold(gold_csv)
    recs = load_records(records_path)

    per_dis = []
    p10s, p100s, r100s, avgrs = [], [], [], []

    for rec in recs:
        disease = rec.get("disease","")
        seeds: List[str] = [s for s in rec.get("seeds", []) if isinstance(s, str)]
        preds = seeds[:]  # 评测用的预测 = seeds 顺序
        seed_count = len(preds)

        golds = find_gold_for_disease(d2g, disease)
        # gold 使用 CSV 的 normalized_drug；宽松匹配里会再做归一
        p10  = precision_at_k(preds, golds, 10)
        p100 = precision_at_k(preds, golds, 100)
        r100 = recall_at_k(preds, golds, 100)
        ar   = avg_rank_with_cap(preds, golds, seed_count)

        per_dis.append({
            "disease": disease,
            "num_gold": len(golds),
            "seed_count": seed_count,
            "P@10": round(p10,4),
            "P@100": round(p100,4),
            "R@100": round(r100,4),
            "AvgRank": round(ar,3),
            "gold_from_csv": golds,     # 只看 CSV 的金标
            "seeds": seeds              # 完整 seeds
        })

        p10s.append(p10); p100s.append(p100); r100s.append(r100); avgrs.append(ar)

    macro = {
        "macro_P@10": round(sum(p10s)/len(p10s), 4) if p10s else 0.0,
        "macro_P@100": round(sum(p100s)/len(p100s), 4) if p100s else 0.0,
        "macro_R@100": round(sum(r100s)/len(r100s), 4) if r100s else 0.0,
        "macro_AvgRank": round(sum(avgrs)/len(avgrs), 3) if avgrs else 0.0,
        "num_diseases": len(per_dis)
    }

    print("=== Contraindication (seeds vs CSV gold) ===")
    for k,v in macro.items():
        print(f"{k}: {v}")

    if save_json:
        with open(os.path.join(out_dir, "macro.json"), "w", encoding="utf-8") as f:
            json.dump(macro, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "per_disease.json"), "w", encoding="utf-8") as f:
            json.dump(per_dis, f, ensure_ascii=False, indent=2)
        print(f"[saved] JSON -> {out_dir}")

    if save_csv:
        import csv as _csv
        keys = ["disease","num_gold","seed_count","P@10","P@100","R@100","AvgRank","gold_from_csv","seeds"]
        with open(os.path.join(out_dir, "per_disease.csv"), "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in per_dis:
                r = dict(row)
                r["gold_from_csv"] = "; ".join(r.get("gold_from_csv", []))
                r["seeds"] = "; ".join(r.get("seeds", []))
                w.writerow(r)
        print(f"[saved] CSV -> {os.path.join(out_dir, 'per_disease.csv')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="contraindication_records.jsonl", help="path to contraindication/_records.jsonl")
    ap.add_argument("--gold_csv", default="filtered_contraindication.csv", help="path to filtered_contraindication.csv")
    ap.add_argument("--out_dir", default="eval_out_contraindication")
    ap.add_argument("--no_json", action="store_true")
    ap.add_argument("--no_csv", action="store_true")
    args = ap.parse_args()

    evaluate(
        records_path=args.records,
        gold_csv=args.gold_csv,
        out_dir=args.out_dir,
        save_json=(not args.no_json),
        save_csv=(not args.no_csv)
    )

if __name__ == "__main__":
    main()
