# rare_ablation_eval.py
import os, json, argparse, re, math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


# ========================= 近似归一化 =========================
SALT_OR_FORM_WORDS = {
    "sodium","potassium","hydrochloride","hydrobromide","sulfate","phosphate",
    "acetate","tartrate","mesylate","citrate","maleate","succinate","nitrate",
    "gel","pill","capsule","capsules","infusion","dpi","ointment","solution",
    "drops","eye","oral","product","placebo","tablet","tablets","injection",
    "spray","cream","patch","extended","release","er","xr","sr","mg","ug","µg"
}
PUNCT_TO_SPACE = re.compile(r"[®™\-\_/\\,;:()\[\]\+]+")   # + 也归一为空格
MULTISPACE = re.compile(r"\s+")
DIGITS = re.compile(r"^\d+(\.\d+)?$")

ALIAS_GROUPS = [
    {"nuedexta", "dextromethorphan + quinidine", "dextromethorphan quinidine"},
    {"growth hormone", "somatropin"},
    {"omega 3 fatty acids", "epa dha", "epanova"},
    {"ursodiol", "actigall"},
    {"aztreonam lysine", "aztreonam for inhalation", "azli"},
    {"ciprofloxacin dpi", "ciprofloxacin inhalation powder", "ciprofloxacin dry powder inhaler"},
]

# 预处理别名（先规范再建映射）
def _norm_base(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = PUNCT_TO_SPACE.sub(" ", s)
    s = s.replace(" and ", " ")
    s = MULTISPACE.sub(" ", s).strip()
    toks = [t for t in s.split() if t and (t not in SALT_OR_FORM_WORDS) and (not DIGITS.match(t))]
    s = " ".join(toks)
    # 进一步规整若干常见写法
    s = s.replace("omega-3", "omega 3")
    s = MULTISPACE.sub(" ", s).strip()
    return s

_ALIAS_CANON: Dict[str, str] = {}
def _build_alias():
    for grp in ALIAS_GROUPS:
        normed = sorted({_norm_base(x) for x in grp if _norm_base(x)})
        if not normed: continue
        canon = normed[0]  # 选词典序最小作代表
        for n in normed:
            _ALIAS_CANON[n] = canon
_build_alias()

def canon_name(s: str) -> str:
    b = _norm_base(s)
    return _ALIAS_CANON.get(b, b)


# ========================= 指标 =========================
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


# ========================= 评测主逻辑 =========================
def load_all_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 只需要 disease / normalized_drug
    if "disease" not in df.columns or "normalized_drug" not in df.columns:
        raise RuntimeError(f"{path} 缺少 disease 或 normalized_drug 列")
    df = df[["disease", "normalized_drug"]].copy()
    df["disease"] = df["disease"].astype(str).str.strip()
    df["normalized_drug"] = df["normalized_drug"].astype(str).str.strip()
    # 近似归一化
    df["disease_norm"] = df["disease"].astype(str)
    df["drug_canon"] = df["normalized_drug"].apply(canon_name)
    # 同病种去重
    df = df.groupby(["disease_norm", "drug_canon"], as_index=False).size()
    return df[["disease_norm", "drug_canon"]]

def read_records(jsonl_path: str) -> List[Dict[str, Any]]:
    out = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict): continue
            out.append(obj)
    return out

def eval_one_disease(
    disease: str,
    seeds: List[str],
    gold: List[str],
    universe_from_allcsv: List[str],
    k: int = 10
) -> Dict[str, Any]:
    # 近似归一化
    seeds_c = []
    seen = set()
    for s in seeds:
        c = canon_name(s)
        if c and c not in seen:
            seeds_c.append(c); seen.add(c)

    gold_set = {canon_name(x) for x in gold if canon_name(x)}
    universe = set(universe_from_allcsv)
    # 把 seeds 也并入全集（模型可能给出 all.csv 之外的说法）
    universe |= set(seeds_c)
    # 从全集里删掉空串
    universe.discard("")

    # ---------- Top-K 指标 ----------
    topk = seeds_c[:k]
    tp_topk = sum(1 for x in topk if x in gold_set)
    denom = max(1, min(k, len(seeds_c)))  # 防 0
    p_at_k = 100.0 * tp_topk / denom
    r_at_k = 100.0 * (tp_topk / max(1, len(gold_set)))

    # ---------- AvgRank ----------
    m = len(seeds_c)
    rank_map = {d: i+1 for i, d in enumerate(seeds_c)}  # 1-based
    if len(gold_set) == 0:
        avg_rank = float('nan')
    else:
        ranks = [rank_map.get(g, m + 1) for g in gold_set]
        avg_rank = float(np.mean(ranks))

    # ---------- AUPRC / AUROC ----------
    # 构造评分（seeds 按排名线性递减；其他均为 0）
    # 候选顺序只影响 AP；AUROC 用秩和对并列做平均
    m = len(seeds_c)
    scores = {}
    for i, d in enumerate(seeds_c, start=1):
        scores[d] = (m - i + 1) / m if m > 0 else 0.0
    for d in universe:
        scores.setdefault(d, 0.0)

    cand_list = list(scores.keys())
    y_score = np.array([scores[d] for d in cand_list], dtype=float)
    y_true = np.array([1 if d in gold_set else 0 for d in cand_list], dtype=int)

    ap = average_precision(y_true, y_score)
    roc = auroc(y_true, y_score)

    return {
        "disease": disease,
        "num_gold": len(gold_set),
        "seed_count": len(seeds_c),
        "P@10": p_at_k,
        "R@10": r_at_k,
        "AvgRank": avg_rank,
        "AUPRC": ap,
        "AUROC": roc,
        "seeds": seeds_c,
        "gold": sorted(list(gold_set)),
    }


def evaluate_records_one_subset(
    records_path: str,
    all_csv_path: str,
    k: int = 10,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    if not os.path.exists(records_path):
        raise FileNotFoundError(records_path)
    if not os.path.exists(all_csv_path):
        raise FileNotFoundError(all_csv_path)

    df_all = load_all_csv(all_csv_path)
    # disease -> 全集药物
    uni_map: Dict[str, List[str]] = {}
    for dis, subdf in df_all.groupby("disease_norm"):
        uni_map[dis] = subdf["drug_canon"].tolist()

    recs = read_records(records_path)
    per_disease = []
    for r in recs:
        disease = str(r.get("disease","")).strip()
        seeds = [s for s in (r.get("seeds") or []) if isinstance(s, str)]
        gold = [g for g in (r.get("gold_drugs") or []) if isinstance(g, str)]
        if not disease:
            continue
        uni = uni_map.get(disease, [])
        res = eval_one_disease(disease, seeds, gold, uni, k=k)
        per_disease.append(res)

    # 宏平均（忽略 NaN）
    def _nanmean(key):
        arr = np.array([x[key] for x in per_disease], dtype=float)
        return float(np.nanmean(arr)) if len(arr) else float('nan')

    summary = {
        "num_diseases": len(per_disease),
        "K": k,
        "AUPRC": _nanmean("AUPRC"),
        "AUROC": _nanmean("AUROC"),
        "P@10": _nanmean("P@10"),
        "R@10": _nanmean("R@10"),
        "AvgRank": _nanmean("AvgRank")
    }

    # 保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "per_disease.json"), "w", encoding="utf-8") as f:
            json.dump(per_disease, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def find_configs(out_dir: str) -> List[str]:
    if not os.path.isdir(out_dir):
        return []
    items = []
    for name in os.listdir(out_dir):
        p = os.path.join(out_dir, name)
        if os.path.isdir(p):
            items.append(name)
    items.sort()
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs_ablation", help="ablation 输出根目录（包含多个 ConfigTag 子目录）")
    ap.add_argument("--ind_all", type=str, required=True, help="filtered_indication_all.csv")
    ap.add_argument("--contra_all", type=str, required=True, help="filtered_contraindication_all.csv")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    configs = find_configs(args.out_dir)
    if not configs:
        raise RuntimeError(f"未在 {args.out_dir} 下发现消融子目录")

    table_rows = []
    for cfg in configs:
        base_cfg = os.path.join(args.out_dir, cfg)

        # indication
        ind_dir = os.path.join(base_cfg, "indication")
        ind_rec = os.path.join(ind_dir, "_records.jsonl")
        if os.path.exists(ind_rec):
            s_ind = evaluate_records_one_subset(
                records_path=ind_rec,
                all_csv_path=args.ind_all,
                k=args.k,
                save_dir=ind_dir
            )
        else:
            s_ind = {"AUPRC": float('nan'), "AUROC": float('nan'), "P@10": float('nan'),
                     "R@10": float('nan'), "AvgRank": float('nan'), "num_diseases": 0}

        # contraindication
        con_dir = os.path.join(base_cfg, "contraindication")
        con_rec = os.path.join(con_dir, "_records.jsonl")
        if os.path.exists(con_rec):
            s_con = evaluate_records_one_subset(
                records_path=con_rec,
                all_csv_path=args.contra_all,
                k=args.k,
                save_dir=con_dir
            )
        else:
            s_con = {"AUPRC": float('nan'), "AUROC": float('nan'), "P@10": float('nan'),
                     "R@10": float('nan'), "AvgRank": float('nan'), "num_diseases": 0}

        table_rows.append((cfg, s_ind, s_con))

    # 打印一个简表
    def _fmt(x): 
        if x != x or x is None:  # NaN
            return "—"
        return f"{x:.3f}"

    print("\n==== Ablation Evaluation (macro-averaged) ====")
    print("Config".ljust(26),
          "| IND  AUPRC  AUROC  P@10%  R@10%  AvgRank |",
          "CONTRA  AUPRC  AUROC  P@10%  R@10%  AvgRank")
    for cfg, si, sc in table_rows:
        print(cfg.ljust(26),
              f"| {_fmt(si['AUPRC'])}  {_fmt(si['AUROC'])}  {_fmt(si['P@10'])}  {_fmt(si['R@10'])}  {_fmt(si['AvgRank'])} |",
              f"{_fmt(sc['AUPRC'])}  {_fmt(sc['AUROC'])}  {_fmt(sc['P@10'])}  {_fmt(sc['R@10'])}  {_fmt(sc['AvgRank'])}")

if __name__ == "__main__":
    main()
