# run_ablation_configs.py
# 说明：
# - 依赖同目录下的 run_pipeline2.py（已包含 Orchestrator/Explorer/LLM 调用与并发安全I/O）
# - 针对 *_all.csv：
#     * indication: 使用列 is_gold_indication==1 作为正例；若列缺失，则 conclusion ∈ {"适应症线索","可能适应症且需注意安全"} 视为正例
#     * contraindication: 使用列 is_gold_contraindication==1 作为正例；若列缺失，则 conclusion == "禁忌信号" 视为正例
# - 为每个配置分别产出 <out_dir>/<ConfigTag>/<subset>/_records.jsonl

import os, re, json, argparse, threading
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_pipeline2 as rp2


# -------------------- 基础工具 --------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

_file_locks: Dict[str, threading.Lock] = {}
_guard = threading.Lock()
def file_lock(path: str) -> threading.Lock:
    with _guard:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]

def append_jsonl(path: str, rec: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with file_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_done_indices(jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if "sample_index" in obj:
                    done.add(int(obj["sample_index"]))
            except Exception:
                continue
    return done

def nname(s: str) -> str:
    try:
        return rp2.normalize_name(s)
    except Exception:
        s = (s or "").lower()
        s = re.sub(r"[®™]", "", s)
        s = re.sub(r"[^a-z0-9\+\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s


# -------------------- 数据载入（含新打标规则） --------------------
def load_all_csv_with_labels(path: str, subset: str) -> pd.DataFrame:
    """
    返回列：disease, normalized_drug, label(0/1)
    subset: "indication" 或 "contraindication"
    打标优先用 is_gold_indication / is_gold_contraindication；缺失则回退到 conclusion 文本。
    并对 (disease, normalized_drug) 去重（label 取 max）。
    """
    df = pd.read_csv(path)
    # 列名鲁棒获取
    disease_col = "disease" if "disease" in df.columns else df.columns[0]
    nd_col = "normalized_drug" if "normalized_drug" in df.columns else df.columns[-3]
    concl_col = "conclusion" if "conclusion" in df.columns else None

    # 规范字符串
    df[disease_col] = df[disease_col].astype(str).str.strip()
    df[nd_col] = df[nd_col].astype(str).str.strip()
    if concl_col:
        df[concl_col] = df[concl_col].astype(str).str.strip()

    if subset == "indication":
        if "is_gold_indication" in df.columns:
            lab = df["is_gold_indication"].fillna(0).astype(int)
        else:
            # 结论文本映射
            pos_set = {"适应症线索", "可能适应症且需注意安全"}
            lab = df[concl_col].apply(lambda x: 1 if str(x) in pos_set else 0).astype(int)
    else:  # contraindication
        if "is_gold_contraindication" in df.columns:
            lab = df["is_gold_contraindication"].fillna(0).astype(int)
        else:
            lab = df[concl_col].apply(lambda x: 1 if str(x) == "禁忌信号" else 0).astype(int)

    out = pd.DataFrame({
        "disease": df[disease_col],
        "normalized_drug": df[nd_col],
        "label": lab
    })
    # 丢掉空值
    out = out[(out["disease"].astype(str).str.len() > 0) & (out["normalized_drug"].astype(str).str.len() > 0)]
    # 对 (disease, normalized_drug) 去重：label 取 max（任何一条是正例就算正例）
    out = out.groupby(["disease", "normalized_drug"], as_index=False)["label"].max()
    return out


# -------------------- Explorer prompt --------------------
def make_bio_prompt(subset: str, disease: str, n: int) -> str:
    if subset == "indication":
        return (
            f"You are an expert biomedical researcher.\n"
            f"List {n} drugs that could be indicated or repurposed to treat the rare disease \"{disease}\".\n"
            f"Respond with drug names only."
        )
    else:
        return (
            f"You are an expert biomedical researcher.\n"
            f"List {n} drugs that are contraindicated or should be avoided in patients with \"{disease}\".\n"
            f"Respond with drug names only."
        )


# -------------------- 种子生成/缓存（共享以保证公平） --------------------
def gen_or_get_seeds(disease: str, subset: str, n_seeds: int,
                     cache: Dict[str, List[str]],
                     textual_feedback: bool,
                     df_all: pd.DataFrame) -> List[str]:
    """
    - textual_feedback=True：给 Explorer 完整 bio_prompt；否则传空（去文本反馈消融）。
    - 失败时兜底：从该病的 CSV 候选中抽取，优先正例，再补其它，去重到 n_seeds。
    """
    key = f"{subset}::{disease}"
    if key in cache:
        return cache[key]

    bio_prompt = make_bio_prompt(subset, disease, n_seeds) if textual_feedback else ""
    try:
        seeds = rp2.run_explorer(
            bio_prompt=bio_prompt,
            query={"entity": disease, "relation": ("indication" if subset=="indication" else "contraindication")},
            target_type="drug",
            n_seeds=n_seeds
        )
    except Exception:
        seeds = []

    # 兜底：用 CSV 候选
    if not seeds:
        cand_all = df_all[df_all["disease"] == disease]
        pos = cand_all[cand_all["label"] == 1]["normalized_drug"].astype(str).tolist()
        neg = cand_all[cand_all["label"] == 0]["normalized_drug"].astype(str).tolist()
        uniq, seen = [], set()
        for s in pos + neg:
            ss = s.strip()
            if ss and ss.lower() not in seen:
                uniq.append(ss); seen.add(ss.lower())
            if len(uniq) >= n_seeds:
                break
        seeds = uniq[:n_seeds]

    cache[key] = seeds
    return seeds


# -------------------- 消融补丁（monkeypatch agents） --------------------
class AgentPatches:
    def __init__(self, tag: str):
        self.tag = tag
        self.orig_run_pi = rp2.run_pi
        self.orig_run_skeptic = rp2.run_skeptic
        self.orig_run_proponent = rp2.run_proponent

    def __enter__(self):
        tag = self.tag

        # 禁用 Skeptic（single-agent / no-critique）
        def _skeptic_noop(mode, payload):
            if mode in ("build_counterchain", "execute_actions"):
                return {"graph_updates_per_hypothesis": []}
            return {}
        if tag in ("Debate-single", "Skeptic-no-critique"):
            rp2.run_skeptic = _skeptic_noop

        # PI 仅最终阶段（score/revise 返回空，不中断，不触发再生）
        def _pi_final_only(mode: str, payload: Dict[str, Any]):
            if mode == "init":
                out = self.orig_run_pi(mode, payload)
                try:
                    if isinstance(out, dict):
                        plan = out.get("plan", {}) or {}
                        plan["rounds"] = 1
                        out["plan"] = plan
                except Exception:
                    pass
                return out
            if mode == "score":
                return {
                    "scoring_summary": [],
                    "ranking": [],
                    "delta_since_last_round": 0.1,
                    "stop_decision": {"should_stop": False, "reason": "final-only ablation"}
                }
            if mode == "revise":
                return {"revisions": [], "notes": "final-only ablation; no interrupts."}
            return self.orig_run_pi(mode, payload)
        if tag == "PI-final-only":
            rp2.run_pi = _pi_final_only

        # 去启发式迁移：init 阶段移除 heuristic_priors
        def _pi_no_heuristics(mode: str, payload: Dict[str, Any]):
            if mode == "init":
                p = dict(payload)
                p.pop("heuristic_priors", None)
                return self.orig_run_pi(mode, p)
            return self.orig_run_pi(mode, payload)
        if tag == "No-heuristic-transfer":
            rp2.run_pi = _pi_no_heuristics

        return self

    def __exit__(self, exc_type, exc, tb):
        rp2.run_pi = self.orig_run_pi
        rp2.run_skeptic = self.orig_run_skeptic
        rp2.run_proponent = self.orig_run_proponent
        return False


# -------------------- 跑一个子任务（indication / contraindication） --------------------
def run_subset(subset: str,
               df_all: pd.DataFrame,
               out_dir: str,
               config_tag: str,
               textual_feedback: bool,
               n_seeds: int,
               workers: int,
               resume: bool,
               shared_seed_cache: Dict[str, List[str]]):
    ensure_dir(out_dir)
    jsonl_path = os.path.join(out_dir, "_records.jsonl")
    done = read_done_indices(jsonl_path) if resume else set()

    # 唯一病种
    diseases = sorted(df_all["disease"].astype(str).unique().tolist())
    jobs = []
    for i, dis in enumerate(diseases):
        if i in done:
            continue
        gold = df_all[(df_all["disease"] == dis) & (df_all["label"] == 1)]["normalized_drug"].astype(str).tolist()
        jobs.append((i, dis, gold))

    print(f"[{config_tag}/{subset}] planned={len(jobs)} (resume={resume}, done={len(done)})")

    def _work(job):
        idx, disease, gold_drugs = job
        relation = "indication" if subset == "indication" else "contraindication"
        try:
            # 生成/复用 seeds（共享以保证各配置公平，默认 True）
            seeds = gen_or_get_seeds(
                disease=disease, subset=subset, n_seeds=n_seeds,
                cache=shared_seed_cache, textual_feedback=textual_feedback,
                df_all=df_all
            )
            if not seeds:
                rec = {
                    "sample_index": idx, "subset": subset, "disease": disease,
                    "relation": relation, "gold_drugs": gold_drugs,
                    "seeds": [], "predictions": [], "ranking": [], "top_recommendations": [],
                    "run_dir": None, "config": {"config_tag": config_tag, "flags": {"textual_feedback": textual_feedback}},
                    "note": "no_seeds"
                }
                append_jsonl(jsonl_path, rec)
                return idx

            # 构造 Orchestrator
            hyps = rp2.seeds_to_hypotheses(seeds, target_type="drug", start_idx=1)
            tag = f"{subset}_{re.sub(r'[^a-z0-9]+','_', disease.lower())[:24]}"
            run_base = os.path.join(out_dir, "runs")
            ensure_dir(run_base)
            orch = rp2.Orchestrator(
                run_dir=run_base,
                query={"entity": disease, "relation": relation},
                hypotheses=hyps,
                init_thresholds={"stop_delta":0.03, "saturation_ratio":0.65},
                tag=tag,
                bio_prompt=(make_bio_prompt(subset, disease, n_seeds) if textual_feedback else ""),
                seed_target_type="drug",
                n_seeds_default=n_seeds
            )
            orch.seed_history.append(seeds)
            final_report = orch.run()

            # 提取预测/排序
            preds = []
            if getattr(orch, "last_scores", None):
                tmp = sorted([(s["hypothesis_id"], float(s.get("score", 0.0))) for s in orch.last_scores],
                             key=lambda x: x[1], reverse=True)
                for hid, sc in tmp:
                    nm = next((h["candidate"]["name"] for h in orch.hypotheses if h["id"] == hid), None)
                    if nm:
                        preds.append({"drug": nm, "score": sc})

            ranking = [p["drug"] for p in preds]
            top_rec_names = []
            if isinstance(final_report, dict):
                for r in (final_report.get("final_recommendations") or []):
                    nm = (r or {}).get("candidate", {}).get("name")
                    if isinstance(nm, str): top_rec_names.append(nm)

            rec = {
                "sample_index": idx,
                "subset": subset,
                "disease": disease,
                "relation": relation,
                "gold_drugs": gold_drugs,
                "seeds": seeds,
                "predictions": preds,
                "ranking": ranking,
                "top_recommendations": top_rec_names,
                "run_dir": orch.run_dir,
                "config": {"config_tag": config_tag, "flags": {"textual_feedback": textual_feedback}},
                "note": "ok"
            }
            append_jsonl(jsonl_path, rec)
            return idx

        except Exception as e:
            rec = {
                "sample_index": idx, "subset": subset, "disease": disease,
                "relation": relation, "gold_drugs": gold_drugs,
                "seeds": [], "predictions": [], "ranking": [], "top_recommendations": [],
                "run_dir": None, "config": {"config_tag": config_tag},
                "note": f"runtime_error:{type(e).__name__}: {e}"
            }
            append_jsonl(jsonl_path, rec)
            return idx

    # 并发执行
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work, job) for job in jobs]
            for f in as_completed(futs):
                try:
                    i = f.result()
                    print(f"[{config_tag}/{subset}] done #{i}")
                except Exception as e:
                    print(f"[{config_tag}/{subset}] worker error: {e}")
    else:
        for job in jobs:
            _work(job)


# -------------------- 主流程：跑所有消融配置 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ind_all", type=str, default="filtered_indication_all.csv", help="filtered_indication_all.csv")
    ap.add_argument("--contra_all", type=str, default="filtered_contraindication_all.csv", help="filtered_contraindication_all.csv")
    ap.add_argument("--out_dir", type=str, default="runs_ablation")
    ap.add_argument("--n_seeds", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--share_seeds", action="store_true", default=False, help="不同配置共享同一批 seeds（推荐开启保证公平）")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # 读取两个 *_all.csv（用新规则打标）
    df_ind = load_all_csv_with_labels(args.ind_all, subset="indication")
    df_contra = load_all_csv_with_labels(args.contra_all, subset="contraindication")

    # 定义消融配置（RareAgent 必须第一个，以便共享种子时由它先生成）
    configs = [
        ("Debate-single", True),          # 单代理（禁用 Skeptic）
        ("Skeptic-no-critique", True),    # Skeptic 不输出反证
        ("PI-final-only", True),          # PI 只在最终阶段介入
        ("No-textual-feedback", False),   # Explorer 去文本提示
        ("No-heuristic-transfer", True),  # 移除 heuristic_priors
    ]

    shared_seed_cache: Dict[str, List[str]] = {} if args.share_seeds else {}

    for tag, textual_feedback in configs:
        print(f"\n==== Running config: {tag} ====")
        cfg_dir = os.path.join(args.out_dir, tag)
        ensure_dir(cfg_dir)

        with AgentPatches(tag):
            # indication
            run_subset(
                subset="indication",
                df_all=df_ind,
                out_dir=os.path.join(cfg_dir, "indication"),
                config_tag=tag,
                textual_feedback=textual_feedback,
                n_seeds=args.n_seeds,
                workers=args.workers,
                resume=args.resume,
                shared_seed_cache=shared_seed_cache
            )
            # contraindication
            run_subset(
                subset="contraindication",
                df_all=df_contra,
                out_dir=os.path.join(cfg_dir, "contraindication"),
                config_tag=tag,
                textual_feedback=textual_feedback,
                n_seeds=args.n_seeds,
                workers=args.workers,
                resume=args.resume,
                shared_seed_cache=shared_seed_cache
            )

    print("\nAll configurations finished. Each _records.jsonl is under <out_dir>/<ConfigTag>/<subset>/")

if __name__ == "__main__":
    main()
