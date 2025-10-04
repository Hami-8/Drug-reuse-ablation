# rare_ablation_eval.py
import os, re, csv, json, argparse, glob
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from collections import defaultdict, OrderedDict
from datetime import datetime

# ---------------- 近似归一化 & 别名 ----------------
SALT_OR_FORM_WORDS = {
    "sodium","potassium","hydrochloride","hydrobromide","sulfate","phosphate",
    "acetate","tartrate","mesylate","citrate","maleate","succinate","nitrate",
    "gel","pill","capsule","capsules","infusion","dpi","ointment","solution",
    "drops","eye","oral","product","placebo","tablet","tablets","injection",
    "spray","cream","patch","extended","release","er","xr","sr","mg","ug","µg"
}
PUNCT_TO_SPACE = re.compile(r"[®™\-\_/\\,;:()\[\]]+")
MULTISPACE      = re.compile(r"\s+")
DIGITS          = re.compile(r"\b\d+(\.\d+)?\b")

ALIAS_GROUPS = [
    {"nuedexta", "dextromethorphan + quinidine", "dextromethorphan quinidine"},
    {"growth hormone", "somatropin"},
    {"omega 3 fatty acids", "epa dha", "epanova"},
    {"ursodiol", "actigall"},
    {"aztreonam lysine", "aztreonam for inhalation", "azli"},
    {"ciprofloxacin dpi", "ciprofloxacin inhalation powder", "ciprofloxacin dry powder inhaler"},
]

def _normalize_for_key(s: str) -> str:
    if not s:
        return ""
    x = s.lower().strip()
    x = PUNCT_TO_SPACE.sub(" ", x)
    x = x.replace("+", " ")
    x = DIGITS.sub(" ", x)  # 去掉剂量数字
    toks = [t for t in MULTISPACE.sub(" ", x).split(" ") if t]
    toks = [t for t in toks if t not in SALT_OR_FORM_WORDS]
    return " ".join(toks).strip()

# alias map（用归一化后的键）
_ALIAS_MAP: Dict[str, str] = {}
for group in ALIAS_GROUPS:
    norm_group = {_normalize_for_key(g) for g in group}
    rep = sorted(norm_group)[0] if norm_group else None
    if rep:
        for a in norm_group:
            _ALIAS_MAP[a] = rep

def _alias_key(norm: str) -> str:
    return _ALIAS_MAP.get(norm, norm)

def canon(s: str) -> str:
    return _alias_key(_normalize_for_key(s))

def is_match(a: str, b: str) -> bool:
    return canon(a) == canon(b)

def uniq_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        k = canon(s)
        if k and k not in seen:
            seen.add(k); out.append(s)
    return out

# ----------------- P@K / R@K（按用户指定） -----------------
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

def avg_rank(preds: List[str], golds: List[str]) -> float:
    """对每个 gold 求其在 preds 的首命中rank；未命中则 rank=len(preds)+1；对所有 gold 取均值。"""
    if not golds:
        return float("nan")
    ranks = []
    n = len(preds)
    for g in golds:
        r = n + 1
        for i, p in enumerate(preds):
            if is_match(p, g):
                r = i + 1
                break
        ranks.append(r)
    return float(np.mean(ranks)) if ranks else float("nan")

# ----------------- AUPRC(AP) / AUROC -----------------
def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
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
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
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

# ----------------- 载入 gold（来自 *_all.csv） -----------------
def load_gold_and_pool(csv_path: str, label_col: str) -> Dict[str, Dict[str, List[str]]]:
    """
    返回:
      disease -> {
         "gold": [drug1, ...],            # 由 is_gold_* == 1 决定
         "pool": [drug1, drug2, ...],     # 该病在 CSV 中出现过的全部 normalized_drug（去重）
      }
    """
    by_dis: Dict[str, Dict[str, List[str]]] = {}
    tmp_gold: Dict[str, Set[str]] = defaultdict(set)
    tmp_pool: Dict[str, Set[str]] = defaultdict(set)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dis = (row.get("disease") or "").strip()
            drug = (row.get("normalized_drug") or row.get("drug") or "").strip()
            if not dis or not drug:
                continue
            cdis = dis  # 保持原串作键（records 里一般同样字符串）
            tmp_pool[cdis].add(drug)
            try:
                is_pos = int(row.get(label_col, "0")) == 1
            except Exception:
                is_pos = False
            if is_pos:
                tmp_gold[cdis].add(drug)

    for dis in set(tmp_pool.keys()) | set(tmp_gold.keys()):
        pool_list = list(tmp_pool.get(dis, set()))
        gold_list = list(tmp_gold.get(dis, set()))
        # 去重保序（以 canon 为键）
        pool_list = uniq_preserve(pool_list)
        gold_list = uniq_preserve(gold_list)
        by_dis[dis] = {"gold": gold_list, "pool": pool_list}
    return by_dis

# ----------------- 读取 _records.jsonl（用 seeds 作为预测） -----------------
def load_records_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("seeds"), list):
                out.append(obj)
    return out

def seeds_by_disease(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    聚合到 disease 粒度；若同病多条，取第一条非空 seeds；并做近似归一化下的去重保序
    """
    group: Dict[str, List[str]] = {}
    for r in records:
        dis = r.get("disease") or ""
        if not dis: 
            continue
        if dis in group:
            continue
        seeds = r.get("seeds") or []
        group[dis] = uniq_preserve(seeds)
    return group

# ----------------- 评分（每个 disease） -----------------
def scores_from_seeds(seeds: List[str]) -> Dict[str, float]:
    """将 seeds 排名映射为分数（1.0 -> 1/n 线性递减）；键为 canon"""
    n = len(seeds)
    if n == 0:
        return {}
    return {canon(s): float(n - i) / n for i, s in enumerate(seeds)}

def per_disease_eval(seeds: List[str], golds: List[str], pool: List[str], K: int) -> Dict[str, Any]:
    # P@K / R@K / AvgRank 在 seeds vs golds 上算
    Pk = precision_at_k(seeds, golds, K)
    Rk = recall_at_k(seeds, golds, K)
    AR = avg_rank(seeds, golds)

    # AUPRC / AUROC：用 *_all.csv 的候选池（pool）构造 0/1 标签；
    # 对于不在 seeds 的候选，score=0
    gold_canon = {canon(x) for x in golds}
    score_map = scores_from_seeds(seeds)

    y_true, y_score = [], []
    for cand in pool:
        cc = canon(cand)
        y_true.append(1 if cc in gold_canon else 0)
        y_score.append(score_map.get(cc, 0.0))
    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    AP = average_precision(y_true, y_score) if len(y_true) else float("nan")
    ROC = auroc(y_true, y_score) if len(y_true) else float("nan")

    return {
        "num_gold": len(golds),
        "num_pool": len(pool),
        "num_seeds": len(seeds),
        "P@K": Pk, "R@K": Rk, "AvgRank": AR,
        "AUPRC": AP, "AUROC": ROC
    }

def macro_average(dicts: List[Dict[str, Any]], keys: List[str]) -> Dict[str, float]:
    res = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d and not (isinstance(d[k], float) and np.isnan(d[k]))]
        res[k] = float(np.mean(vals)) if vals else float("nan")
    return res

# ----------------- 主流程（统一 CLI） -----------------
def evaluate_config(config_dir: str, ind_all: str, contra_all: str, K: int) -> Dict[str, Any]:
    """
    针对单个 ConfigTag 目录，分别评测 indication 与 contraindication，然后给出 combined 宏平均
    """
    out = {"config_tag": os.path.basename(config_dir), "subsets": {}}

    # 载入标准集
    gold_ind = load_gold_and_pool(ind_all, label_col="is_gold_indication")
    gold_con = load_gold_and_pool(contra_all, label_col="is_gold_contraindication")

    for subset, goldmap in [("indication", gold_ind), ("contraindication", gold_con)]:
        rec_path = os.path.join(config_dir, subset, "_records.jsonl")
        if not os.path.exists(rec_path):
            continue
        records = load_records_jsonl(rec_path)
        pred_map = seeds_by_disease(records)

        per_dis_list = []
        for dis, seeds in pred_map.items():
            gm = goldmap.get(dis)
            if not gm:
                # 找不到该疾病的标准集条目则跳过
                continue
            golds = gm["gold"]
            pool  = gm["pool"]
            res = per_disease_eval(seeds, golds, pool, K)
            per_dis_list.append({
                "disease": dis,
                "num_gold": res["num_gold"], "num_pool": res["num_pool"], "num_seeds": res["num_seeds"],
                "P@K": res["P@K"], "R@K": res["R@K"], "AvgRank": res["AvgRank"],
                "AUPRC": res["AUPRC"], "AUROC": res["AUROC"]
            })

        macro = macro_average(per_dis_list, ["P@K","R@K","AvgRank","AUPRC","AUROC"])
        out["subsets"][subset] = {
            "count_diseases": len(per_dis_list),
            "macro": macro,
            "per_disease": per_dis_list
        }

    # combined（把两个子集的疾病结果合在一起再宏平均）
    combined_list = []
    for subset in ("indication","contraindication"):
        if subset in out["subsets"]:
            combined_list.extend(out["subsets"][subset]["per_disease"])
    out["combined"] = {
        "count_diseases": len(combined_list),
        "macro": macro_average(combined_list, ["P@K","R@K","AvgRank","AUPRC","AUROC"])
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs_ablation",
                    help="ablation 输出根目录（包含多个 ConfigTag 子目录）")
    ap.add_argument("--ind_all", type=str, default="filtered_indication_all.csv",
                    help="filtered_indication_all.csv")
    ap.add_argument("--contra_all", type=str, default="filtered_contraindication_all.csv",
                    help="filtered_contraindication_all.csv")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    if not os.path.isdir(args.out_dir):
        raise SystemExit(f"--out_dir 不存在：{args.out_dir}")

    config_dirs = [os.path.join(args.out_dir, d) for d in os.listdir(args.out_dir)
                   if os.path.isdir(os.path.join(args.out_dir, d))]

    results = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "args": vars(args), "configs": []}

    for cfg in sorted(config_dirs):
        res = evaluate_config(cfg, args.ind_all, args.contra_all, args.k)
        results["configs"].append(res)

    # 打印概览
    print("\n===== Ablation Summary (macro) =====")
    for res in results["configs"]:
        tag = res["config_tag"]
        comb = res["combined"]["macro"]
        print(f"[{tag}]  N={res['combined']['count_diseases']}"
              f" | P@{args.k}={comb['P@K']:.3f}  R@{args.k}={comb['R@K']:.3f}"
              f" | AvgRank={comb['AvgRank']:.3f}  AUPRC={comb['AUPRC']:.3f}  AUROC={comb['AUROC']:.3f}")

        for subset in ("indication","contraindication"):
            if subset in res["subsets"]:
                m = res["subsets"][subset]["macro"]
                n = res["subsets"][subset]["count_diseases"]
                print(f"  └─ {subset:<16} N={n}"
                      f" | P@{args.k}={m['P@K']:.3f}  R@{args.k}={m['R@K']:.3f}"
                      f" | AvgRank={m['AvgRank']:.3f}  AUPRC={m['AUPRC']:.3f}  AUROC={m['AUROC']:.3f}")
    # 保存 JSON
    out_json = os.path.join(args.out_dir, "ablation_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to: {out_json}")

if __name__ == "__main__":
    main()
