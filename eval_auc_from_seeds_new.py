# eval_auc_from_seeds_custom.py
# 仅用 seeds 顺序生成分数；AUPRC/ AUROC 使用自定义 average_precision / auroc。
# 用法一（单文件模式，保持兼容）：
# python eval_auc_from_seeds_custom.py \
#   --subset indication \
#   --records runs_eval/indication/_records.jsonl \
#   --pos_csv filtered_indication.csv \
#   --all_csv filtered_indication_all.csv \
#   --out_dir eval_out_ind_auc
#
# 用法二（自动模式：遍历 runs_ablation 下所有消融配置）：
# python eval_auc_from_seeds_custom.py \
#   --root runs_ablation \
#   --out_root eval_auc_runs_ablation \
#   --pos_ind filtered_indication.csv \
#   --all_ind filtered_indication_all.csv \
#   --pos_contra filtered_contraindication.csv \
#   --all_contra filtered_contraindication_all.csv

import os
import re
import csv
import json
import argparse
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

import numpy as np

# ================== 你要求的 AUPRC / AUROC 实现 ==================

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

# ================== 名称规范化 & 同义宽松匹配（与前一致） ==================

SALT_OR_FORM_WORDS = {
    "sodium","potassium","hydrochloride","hydrobromide","sulfate","phosphate",
    "acetate","tartrate","mesylate","citrate","maleate","succinate","nitrate",
    "gel","pill","capsule","capsules","infusion","dpi","ointment","solution",
    "drops","eye","oral","product","placebo","tablet","tablets","injection",
    "spray","cream","patch","extended","release","er","xr","sr","mg","ug","µg",
    "soft","softgel","intravenous","intranasal","nasal","topical"
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

ALIAS_GROUPS = [
    {"sodium valproate","valproate","valproic acid"},
    {"ursodiol","actigall"},
    {"aztreonam lysine","aztreonam for inhalation","azli"},
    {"omega 3 fatty acids","omega 3 acid ethyl esters","epa dha","fish oil","lovaza","ethyl icosapentate"},
    {"somatropin","growth hormone"},
    {"risedronate","risedronate sodium"},
    {"nuedexta","dextromethorphan + quinidine","dextromethorphan quinidine"},
    {"ciprofloxacin dpi","ciprofloxacin inhalation powder","ciprofloxacin dry powder inhaler"},
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

# ================== 读取 gold（正例）与 all（候选全集） ==================

POS_CONCLUSIONS = {"适应症线索","可能适应症且需注意安全"}  # indication 用

def load_pos_by_disease(pos_csv: str, subset: str) -> Dict[str, Set[str]]:
    """
    从 filtered_indication.csv / filtered_contraindication.csv 读取正例集合
    - key: 规范化 disease
    - val: 规范化 normalized_drug 的集合
    对 indication：若有 conclusion 列，仅保留 POS_CONCLUSIONS；否则全收。
    对 contraindication：整表均视为正例。
    """
    d2pos: Dict[str, Set[str]] = defaultdict(set)
    with open(pos_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_conc = "conclusion" in reader.fieldnames if reader.fieldnames else False
        for row in reader:
            dis = row.get("disease","").strip()
            drug = row.get("normalized_drug","").strip()
            if not dis or not drug: continue
            if subset == "indication" and has_conc:
                if row.get("conclusion","").strip() not in POS_CONCLUSIONS:
                    continue
            kd = normalize_name(dis)
            d2pos[kd].add(canonical(drug))
    return d2pos

def load_all_candidates(all_csv: str) -> Dict[str, List[str]]:
    """
    从 filtered_*_all.csv 读取候选全集（每病种所有 normalized_drug，去重保序）
    """
    d2all: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, Set[str]] = defaultdict(set)
    with open(all_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "disease" in reader.fieldnames and "normalized_drug" in reader.fieldnames:
            for row in reader:
                dis = row.get("disease","").strip()
                drug = row.get("normalized_drug","").strip()
                if not dis or not drug: continue
                kd, kd_drug = normalize_name(dis), canonical(drug)
                if kd_drug not in seen[kd]:
                    d2all[kd].append(kd_drug)
                    seen[kd].add(kd_drug)
            return d2all
    # 兜底：尝试列位置
    with open(all_csv, newline="", encoding="utf-8") as f2:
        reader = csv.reader(f2)
        next(reader, None)
        for row in reader:
            if not row: continue
            dis = row[0].strip() if len(row)>0 else ""
            drug = row[-3].strip() if len(row)>=3 else ""  # 尽量贴近示例结构
            if not dis or not drug: continue
            kd, kd_drug = normalize_name(dis), canonical(drug)
            if kd_drug not in seen[kd]:
                d2all[kd].append(kd_drug)
                seen[kd].add(kd_drug)
    return d2all

# ================== 读取 records（seeds 作为分数来源） ==================

def load_seeds_by_disease(records_path: str) -> Dict[str, List[str]]:
    """
    从 _records.jsonl 读取 seeds 列表（作为预测排名来源）
    - 返回：key=规范化 disease, val=原顺序 seeds（未规范化的原文，匹配时做 canonical）
    """
    d2seeds: Dict[str, List[str]] = {}
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            dis = obj.get("disease","") or obj.get("query_disease","")
            seeds = obj.get("seeds", [])
            if not dis or not isinstance(seeds, list): continue
            kd = normalize_name(dis)
            # 去重保序（按 canonical 去重）
            uniq, seen = [], set()
            for s in seeds:
                if not isinstance(s, str): continue
                cs = canonical(s)
                if cs and cs not in seen:
                    uniq.append(s)
                    seen.add(cs)
            d2seeds[kd] = uniq
    return d2seeds

# ================== 主评测逻辑（自定义 AUPRC / AUROC） ==================

def evaluate_auc(subset: str, records_path: str, pos_csv: str, all_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    d2pos = load_pos_by_disease(pos_csv, subset)
    d2all = load_all_candidates(all_csv)
    d2seeds = load_seeds_by_disease(records_path)

    per = []
    ap_list, roc_list = [], []
    skipped = []

    for kd, all_drugs in d2all.items():
        pos_set = d2pos.get(kd, set())
        seeds_raw = d2seeds.get(kd)
        if not all_drugs or seeds_raw is None:
            skipped.append({"disease": kd, "reason": "no_all_or_no_records"})
            continue

        # seeds 生成分数：rank 1 -> 1.0, rank N -> 1 - (N-1)/N = 1/N
        N = max(1, len(seeds_raw))
        seeds_scores = {}
        for rank, s in enumerate(seeds_raw, start=1):
            seeds_scores[canonical(s)] = 1.0 - (rank-1)/N

        y_true, y_score = [], []
        for drug in all_drugs:
            # 真值：正例=1（来自 pos_csv）；负例=0（all - pos）
            is_pos = any(is_match(drug, g) for g in pos_set)
            y_true.append(1 if is_pos else 0)
            # 预测分数：若在 seeds 中，取对应分；否则 0
            sc = 0.0
            for k_can, val in seeds_scores.items():
                if is_match(drug, k_can):
                    sc = max(sc, val)
            y_score.append(sc)

        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        num_pos = int(y_true.sum())
        num_neg = int(len(y_true) - num_pos)
        if num_pos == 0 or num_neg == 0:
            skipped.append({"disease": kd, "reason": f"invalid_class_dist pos={num_pos}, neg={num_neg}"})
            continue

        ap = average_precision(y_true, y_score)
        auc = auroc(y_true, y_score)

        per.append({
            "disease": kd,
            "num_candidates": int(len(all_drugs)),
            "num_pos": num_pos,
            "num_neg": num_neg,
            "AUPRC": round(ap, 6) if ap==ap else None,
            "AUROC": round(auc, 6) if auc==auc else None,
        })
        if ap==ap: ap_list.append(ap)
        if auc==auc: roc_list.append(auc)

    macro = {
        "subset": subset,
        "macro_AUPRC": round(sum(ap_list)/len(ap_list), 6) if ap_list else None,
        "macro_AUROC": round(sum(roc_list)/len(roc_list), 6) if roc_list else None,
        "num_valid_diseases": len(per),
        "num_skipped": len(skipped),
    }

    with open(os.path.join(out_dir, "per_disease.json"), "w", encoding="utf-8") as f:
        json.dump(per, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "macro.json"), "w", encoding="utf-8") as f:
        json.dump(macro, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "skipped.json"), "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)

    print("=== AUC Summary ===")
    print(json.dumps(macro, ensure_ascii=False, indent=2))
    print(f"[saved] {out_dir}/macro.json , per_disease.json , skipped.json")

# ================== 自动遍历 runs_ablation ==================

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

def auto_eval(root: str,
              out_root: str,
              pos_ind: str,
              all_ind: str,
              pos_contra: str,
              all_contra: str):
    os.makedirs(out_root, exist_ok=True)
    triples = find_records(root)
    if not triples:
        print(f"[WARN] 未在 {root} 下找到任何 _records.jsonl")
        return

    summary_rows = []
    for cfg, subset, rec_path in triples:
        print(f"[EVAL] {cfg} / {subset}  <-  {rec_path}")
        try:
            if subset == "indication":
                pos_csv, all_csv = pos_ind, all_ind
            else:
                pos_csv, all_csv = pos_contra, all_contra

            out_dir = os.path.join(out_root, cfg, subset)
            evaluate_auc(
                subset=subset,
                records_path=rec_path,
                pos_csv=pos_csv,
                all_csv=all_csv,
                out_dir=out_dir
            )

            # 读取 macro 以汇总
            with open(os.path.join(out_dir, "macro.json"), "r", encoding="utf-8") as f:
                macro = json.load(f)
            summary_rows.append({
                "ConfigTag": cfg,
                "Subset": subset,
                "macro_AUPRC": macro.get("macro_AUPRC"),
                "macro_AUROC": macro.get("macro_AUROC"),
                "num_valid_diseases": macro.get("num_valid_diseases"),
                "num_skipped": macro.get("num_skipped"),
            })
        except Exception as e:
            print(f"[ERR] 评测失败：{cfg}/{subset} -> {e}")

    # 写 summary CSV/JSON
    if summary_rows:
        summ_csv = os.path.join(out_root, "_summary_auc.csv")
        summ_json = os.path.join(out_root, "_summary_auc.json")
        keys = ["ConfigTag","Subset","macro_AUPRC","macro_AUROC","num_valid_diseases","num_skipped"]
        with open(summ_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        with open(summ_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        print(f"[DONE] AUC Summary saved to: {summ_csv}  /  {summ_json}")
    else:
        print("[WARN] 无可写入的汇总结果。")

# ================== CLI ==================

def main():
    ap = argparse.ArgumentParser()
    # 单文件模式（兼容）
    ap.add_argument("--subset", choices=["indication","contraindication"])
    ap.add_argument("--records", help="path to *_records.jsonl")
    ap.add_argument("--pos_csv", help="filtered_indication.csv 或 filtered_contraindication.csv")
    ap.add_argument("--all_csv", help="filtered_indication_all.csv 或 filtered_contraindication_all.csv")
    ap.add_argument("--out_dir", help="单文件模式输出目录")

    # 自动模式
    ap.add_argument("--root", type=str, default="runs_ablation_without_sharing",
                    help="runs_ablation 根目录（包含多个 ConfigTag 子目录）")
    ap.add_argument("--out_root", type=str, default="eval_auc_runs_ablation_without_sharing2",
                    help="自动模式评测输出根目录")
    ap.add_argument("--pos_ind", type=str, default="filtered_indication.csv")
    ap.add_argument("--all_ind", type=str, default="filtered_indication_all.csv")
    ap.add_argument("--pos_contra", type=str, default="filtered_contraindication.csv")
    ap.add_argument("--all_contra", type=str, default="filtered_contraindication_all.csv")

    args = ap.parse_args()

    # 自动模式优先（只要提供了 --root 就会走自动模式）
    if args.root:
        auto_eval(
            root=args.root,
            out_root=args.out_root,
            pos_ind=args.pos_ind,
            all_ind=args.all_ind,
            pos_contra=args.pos_contra,
            all_contra=args.all_contra
        )
        return

    # 否则走单文件模式（兼容旧用法）
    if not (args.subset and args.records and args.pos_csv and args.all_csv and args.out_dir):
        print("[ERROR] 单文件模式缺少必要参数。请提供 --subset --records --pos_csv --all_csv --out_dir，或使用 --root 自动模式。")
        return

    evaluate_auc(
        subset=args.subset,
        records_path=args.records,
        pos_csv=args.pos_csv,
        all_csv=args.all_csv,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()
