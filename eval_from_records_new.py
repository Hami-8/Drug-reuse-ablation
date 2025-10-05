# eval_from_records.py
# 评测：用 seeds 作为模型预测来对比 gold（忽略 ranking/predictions）
# 指标：AUPRC, AUROC, P@K, P@100, R@K, R@100, Avg Rank（未命中=seed_count+1）
# 输出：宏平均 + 每疾病详情（含完整 seeds）
# 额外：自动遍历 runs_ablation/<ConfigTag>/<subset>/_records.jsonl 逐一评测并汇总

import os
import re
import json
import csv
import argparse
from typing import List, Dict, Any, Tuple, Set
import numpy as np

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

# ========= 排名到标签（消除重复命中） =========
def greedy_labels_for_seeds(preds: List[str], golds: List[str]) -> List[int]:
    """
    给 seeds 生成 y_true 标签（1/0），使用“贪心去重匹配”：
      - 从前到后扫描每个 seed
      - 若能命中某个尚未匹配过的 gold，则该 seed 记为 1，并占用该 gold
      - 否则记为 0
    这样避免 “risedronate / risedronate sodium” 等重复命中同一标准答案导致 AUPRC 被重复抬高。
    """
    matched_gold: Set[int] = set()
    y_true = []
    for p in preds:
        label = 0
        for gi, g in enumerate(golds):
            if gi in matched_gold:
                continue
            if is_match(p, g):
                matched_gold.add(gi)
                label = 1
                break
        y_true.append(label)
    return y_true

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

# ======= AUPRC / AUROC（用你给的实现；基于 seeds 闭集）=======

def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUPRC 的 AP(平均精度)“排名定义”：对所有正例在其排名处的 precision@k 取均值。
    """
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    cum_tp = 0
    precisions = []
    for i, label in enumerate(y_true_sorted, start=1):
        if label == 1:
            cum_tp += 1
            precisions.append(cum_tp / i)
    if len(precisions) == 0:
        return float('nan')
    return float(np.mean(precisions))

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUROC 用 Mann–Whitney U / 秩和实现；对并列分数使用平均秩。
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # 稳定排序求秩
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # 并列平均秩
    uniq, inv, cnts = np.unique(y_score, return_inverse=True, return_counts=True)
    start_rank = {}
    cur = 1
    for u, c in zip(uniq, cnts):
        start_rank[u] = cur
        cur += c
    for u, c in zip(uniq, cnts):
        if c > 1:
            sr = start_rank[u]
            er = sr + c - 1
            avg = (sr + er) / 2.0
            ranks[inv == np.where(uniq == u)[0][0]] = avg
    R_pos = float(np.sum(ranks[y_true == 1]))
    U = R_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))

def ap_auroc_from_seeds(preds: List[str], golds: List[str]) -> Tuple[float, float]:
    """
    在 seeds 闭集上计算 AUPRC / AUROC：
    - y_true：来自 seeds 对 gold 的贪心匹配（避免一个 gold 被多个近义 seed 重复计数）
    - y_score：按排名生成分数，前排更高（线性缩放到 (0,1]）
    """
    if not preds:
        return float('nan'), float('nan')
    y_true = np.array(greedy_labels_for_seeds(preds, golds), dtype=int)
    n = len(preds)
    # 越靠前分数越高；避免0：用 1..n 线性映射到 (0,1]
    y_score = np.array([(n - i) / n for i in range(n)], dtype=float)
    return average_precision(y_true, y_score), auroc(y_true, y_score)

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

def evaluate_records(records: List[Dict[str, Any]], K: int = 10) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    per_dis = []
    pks, p100s, rks ,r100s, avgrs = [], [],[], [], []
    aps, rocs = [], []

    for rec in records:
        disease = rec.get("disease","")
        golds: List[str] = rec.get("gold_drugs", []) or []
        seeds_raw: List[str] = rec.get("seeds", []) or []

        # 评测用的“预测”= seeds（保持原始顺序）
        preds = [s for s in seeds_raw if isinstance(s, str)]
        seed_count = len(preds)

        p_at_k  = precision_at_k(preds, golds, K)
        p100 = precision_at_k(preds, golds, 100)
        r_at_k = recall_at_k(preds, golds, K)
        r100 = recall_at_k(preds, golds, 100)
        ar   = avg_rank_with_seed_cap(preds, golds, seed_count)
        ap, roc = ap_auroc_from_seeds(preds, golds)

        per_dis.append({
            "disease": disease,
            "num_gold": len(golds),
            "num_preds(seeds)": seed_count,
            "AUPRC": None if np.isnan(ap) else round(ap, 4),
            "AUROC": None if np.isnan(roc) else round(roc, 4),
            f"P@{K}": round(p_at_k, 4),
            "P@100": round(p100, 4),
            f"R@{K}": round(r_at_k, 4),
            "R@100": round(r100, 4),
            "AvgRank": round(ar, 3),
            "seeds": seeds_raw,  # 全部种子
            "gold": golds
        })

        if not np.isnan(ap): aps.append(ap)
        if not np.isnan(roc): rocs.append(roc)
        pks.append(p_at_k)
        p100s.append(p100)
        rks.append(r_at_k)
        r100s.append(r100)
        avgrs.append(ar)

    macro = {
        "macro_AUPRC": round(float(np.mean(aps)), 4) if aps else None,
        "macro_AUROC": round(float(np.mean(rocs)), 4) if rocs else None,
        f"macro_P@{K}": round(sum(pks)/len(pks), 4) if pks else 0.0,
        "macro_P@100": round(sum(p100s)/len(p100s), 4) if p100s else 0.0,
        f"macro_R@{K}": round(sum(rks)/len(rks), 4) if rks else 0.0,
        "macro_R@100": round(sum(r100s)/len(r100s), 4) if r100s else 0.0,
        "macro_AvgRank": round(sum(avgrs)/len(avgrs), 3) if avgrs else 0.0,
        "num_diseases": len(per_dis),
    }
    return macro, per_dis

def save_per_disease_csv(per_dis: List[Dict[str,Any]], path: str, K: int):
    keys = ["disease","num_gold","num_preds(seeds)","AUPRC","AUROC",f"P@{K}","P@100",f"R@{K}","R@100","AvgRank","gold","seeds"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_dis:
            r = dict(row)
            r["gold"]  = "; ".join(r.get("gold", []))
            r["seeds"] = "; ".join(r.get("seeds", []))
            w.writerow(r)

# ========= 自动遍历 runs_ablation =========

def find_records(root: str) -> List[Tuple[str, str, str]]:
    """
    返回 (config_tag, subset, records_path) 列表
    期望目录结构：runs_ablation/<ConfigTag>/<subset>/_records.jsonl
    """
    out = []
    if not os.path.isdir(root):
        return out
    for cfg in sorted(os.listdir(root)):
        cfg_dir = os.path.join(root, cfg)
        if not os.path.isdir(cfg_dir):
            continue
        for subset in ("indication", "contraindication"):
            rec_path = os.path.join(cfg_dir, subset, "_records.jsonl")
            if os.path.exists(rec_path):
                out.append((cfg, subset, rec_path))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs_ablation_without_sharing",
                    help="runs_ablation 根目录（包含多个 ConfigTag 子目录）")
    ap.add_argument("--out_root", type=str, default="eval_runs_ablation_without_sharing666",
                    help="评测输出根目录")
    ap.add_argument("--k", type=int, default=10, help="用于 P@K / R@K 的 K")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    triples = find_records(args.root)
    if not triples:
        print(f"[WARN] 未在 {args.root} 下找到任何 _records.jsonl")
        return

    summary_rows = []
    for cfg, subset, rec_path in triples:
        print(f"[EVAL] {cfg} / {subset}  <-  {rec_path}")
        try:
            records = load_records(rec_path)
            macro, per_dis = evaluate_records(records, K=args.k)

            out_dir = os.path.join(args.out_root, cfg, subset)
            os.makedirs(out_dir, exist_ok=True)

            # 保存 per-disease
            save_per_disease_csv(per_dis, os.path.join(out_dir, "per_disease.csv"), K=args.k)
            with open(os.path.join(out_dir, "per_disease.json"), "w", encoding="utf-8") as f:
                json.dump(per_dis, f, ensure_ascii=False, indent=2)

            # 保存 macro
            with open(os.path.join(out_dir, "macro.json"), "w", encoding="utf-8") as f:
                json.dump(macro, f, ensure_ascii=False, indent=2)

            # 汇总到 summary
            row = {
                "ConfigTag": cfg,
                "Subset": subset,
                "num_diseases": macro.get("num_diseases", 0),
                "macro_AUPRC": macro.get("macro_AUPRC"),
                "macro_AUROC": macro.get("macro_AUROC"),
                f"macro_P@{args.k}": macro.get(f"macro_P@{args.k}"),
                "macro_P@100": macro.get("macro_P@100"),
                f"macro_R@{args.k}": macro.get(f"macro_R@{args.k}"),
                "macro_R@100": macro.get("macro_R@100"),
                "macro_AvgRank": macro.get("macro_AvgRank"),
            }
            summary_rows.append(row)
        except Exception as e:
            print(f"[ERR] 评测失败：{cfg}/{subset} -> {e}")

    # 写 summary CSV/JSON
    if summary_rows:
        summ_csv = os.path.join(args.out_root, "_summary.csv")
        summ_json = os.path.join(args.out_root, "_summary.json")
        keys = ["ConfigTag","Subset","num_diseases","macro_AUPRC","macro_AUROC",
                f"macro_P@{args.k}","macro_P@100",f"macro_R@{args.k}","macro_R@100","macro_AvgRank"]
        with open(summ_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        with open(summ_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        print(f"[DONE] Summary saved to: {summ_csv}  /  {summ_json}")
    else:
        print("[WARN] 无可写入的汇总结果。")

if __name__ == "__main__":
    main()
