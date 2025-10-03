# -*- coding: utf-8 -*-
"""
Eval on filtered_indication.csv & filtered_contraindication.csv

- Query:
  * filtered_indication.csv: entity = disease, relation = "indication"
    gold answers = 所有 normalized_drug（包含“适应症线索”和“可能适应症且需注意安全”）
  * filtered_contraindication.csv: entity = disease, relation = "contraindication"
    gold answers = 所有 normalized_drug（“禁忌信号”）
- Explorer 生成 seeds 数量 = 12
- 结果落盘：
  * runs_new/indication/_records.jsonl
  * runs_new/contraindication/_records.jsonl
  每行包含：disease, relation, gold_drugs, seeds, predictions(带分数), ranking 等
"""

from dotenv import load_dotenv
load_dotenv()  # 读取 .env

import os, re, csv, json, time, threading, random, argparse
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========= 导入你的主流水线 =========
import run_pipeline2 as rp2  # 请确保该文件可被 import

# ========= 并发写文件锁 =========
_FILE_LOCKS: Dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()

def _get_file_lock(path: str) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        if path not in _FILE_LOCKS:
            _FILE_LOCKS[path] = threading.Lock()
        return _FILE_LOCKS[path]

def append_jsonl(path: str, rec: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = _get_file_lock(path)
    with lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ========= 续跑：读取已处理样本 =========
def load_processed_indices_from_records(path: str, rerun_failed: bool = False) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            sid = rec.get("sample_index")
            if sid is None:
                continue
            note = rec.get("note") or ""
            if rerun_failed and isinstance(note, str) and note.startswith("runtime_error"):
                # 失败样本允许重跑
                continue
            try:
                done.add(int(sid))
            except Exception:
                continue
    return done

def append_error_jsonl(path: str, rec: Dict[str, Any]):
    append_jsonl(path, rec)

def log_sample_error(base_out_dir: str, subset: str, sample_index: int,
                     stage: str, err: Exception, extra: Dict[str, Any]):
    rec = {
        "sample_index": sample_index,
        "subset": subset,     # "indication" / "contraindication"
        "stage": stage,
        "error": str(err)[:2000],
        "extra": extra
    }
    append_error_jsonl(os.path.join(base_out_dir, subset, "_errors.jsonl"), rec)
    append_error_jsonl(os.path.join(base_out_dir, "_failed_samples.jsonl"), rec)

# ========= 规范化 =========
def norm_txt(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def sanitize_tag(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:32]

# ========= 读取 CSV，组装任务 =========
ALLOW_INDICATION_CONCLUSIONS = {"适应症线索", "可能适应症且需注意安全"}

def load_tasks_from_filtered_csv(path: str, relation: str) -> List[Dict[str, Any]]:
    """
    返回任务列表，每个任务：
    {
      "disease": <str>,
      "relation": "indication" | "contraindication",
      "gold_drugs": [normalized_drug 去重排序],
      "rows": [原始行...]
    }
    """
    tasks: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            disease = norm_txt(row.get("disease"))
            if not disease:
                continue
            conclusion = norm_txt(row.get("conclusion"))
            ndrug = norm_txt(row.get("normalized_drug") or row.get("drug"))

            if relation == "indication":
                # 仅保留“适应症线索 / 可能适应症且需注意安全”
                if conclusion not in ALLOW_INDICATION_CONCLUSIONS:
                    continue
            # contraindication 全保留

            bucket = tasks.setdefault(disease, {
                "disease": disease,
                "relation": relation,
                "gold_drugs": set(),
                "rows": []
            })
            if ndrug:
                bucket["gold_drugs"].add(ndrug)
            bucket["rows"].append(row)

    out: List[Dict[str, Any]] = []
    for dis, obj in tasks.items():
        obj["gold_drugs"] = sorted(set(obj["gold_drugs"]))
        out.append(obj)
    return out

# ========= Explorer 种子生成（12个） =========
def get_seeds_for_disease(disease: str, relation: str, n: int = 12) -> List[str]:
    """
    直接复用 run_pipeline2 的 Explorer：
    - bio_prompt: 基于 relation 构造
    - query: {"entity": disease, "relation": relation, "target_type": "drug"}
    """
    if relation == "indication":
        bio_prompt = f"List drugs that could be indicated for the rare disease '{disease}'. Return only names."
    else:
        bio_prompt = f"List drugs that are contraindicated or could worsen the rare disease '{disease}'. Return only names."
    query = {"entity": disease, "relation": relation, "target_type": "drug"}
    try:
        seeds = rp2.run_explorer(bio_prompt, query, target_type="drug", n_seeds=n)
    except Exception as e:
        # 兜底：空列表
        seeds = []
    # 去重并截断
    uniq, seen = [], set()
    for s in seeds:
        ss = norm_txt(s)
        if ss and ss.lower() not in seen:
            uniq.append(ss)
            seen.add(ss.lower())
    return uniq[:n]

# ========= 核心：处理一个任务（一个疾病） =========
def run_one_task(task_index: int, task: Dict[str, Any], out_dir: str,
                 n_seeds: int = 12) -> Dict[str, Any]:
    """
    返回一条记录（会写入 _records.jsonl）
    """
    subset = task["relation"]  # "indication" / "contraindication"
    subdir = os.path.join(out_dir, subset)
    os.makedirs(subdir, exist_ok=True)

    disease = task["disease"]
    relation = task["relation"]
    gold = task["gold_drugs"]

    # 1) Explorer 生成 12 个 seeds（药物）
    try:
        seeds = get_seeds_for_disease(disease, relation, n=n_seeds)
    except Exception as e:
        # 记录并返回占位
        log_sample_error(out_dir, subset, task_index, "explorer", e,
                         {"disease": disease, "relation": relation})
        return {
            "sample_index": task_index,
            "subset": subset,
            "disease": disease,
            "relation": relation,
            "gold_drugs": gold,
            "seeds": [],
            "predictions": [],
            "ranking": [],
            "top_recommendations": [],
            "run_dir": None,
            "note": "runtime_error:explorer"
        }

    if not seeds:
        # 无种子：占位
        return {
            "sample_index": task_index,
            "subset": subset,
            "disease": disease,
            "relation": relation,
            "gold_drugs": gold,
            "seeds": [],
            "predictions": [],
            "ranking": [],
            "top_recommendations": [],
            "run_dir": None,
            "note": "no_seeds"
        }

    # 2) 组织 Orchestrator
    query = {"entity": disease, "relation": relation}
    hyps = rp2.seeds_to_hypotheses(seeds, target_type="drug", start_idx=1)
    tag = f"{subset}_{sanitize_tag(disease)}"

    try:
        orch = rp2.Orchestrator(
            run_dir=subdir,
            query=query,
            hypotheses=hyps,
            init_thresholds={"stop_delta": 0.03, "saturation_ratio": 0.65},
            tag=tag,
            bio_prompt=f"New dataset {subset} for '{disease}'",
            seed_target_type="drug",
            n_seeds_default=n_seeds,
        )
        orch.seed_history.append(seeds)
        final_report = orch.run()
    except Exception as e:
        log_sample_error(out_dir, subset, task_index, "orchestrator", e,
                         {"disease": disease, "relation": relation, "seeds": seeds})
        return {
            "sample_index": task_index,
            "subset": subset,
            "disease": disease,
            "relation": relation,
            "gold_drugs": gold,
            "seeds": seeds,
            "predictions": [],
            "ranking": [],
            "top_recommendations": [],
            "run_dir": None,
            "note": "runtime_error:orchestrator"
        }

    # 3) 提取预测（用于后续评测）
    #   - 优先从最终报告 final_recommendations
    #   - 退化用最后一次评分 orch.last_scores
    predictions: List[Dict[str, Any]] = []
    top_rec: List[str] = []
    # 从 final_report（有些情况下为空）
    if isinstance(final_report, dict):
        fr = final_report.get("final_recommendations") or []
        for item in fr:
            nm = (((item or {}).get("candidate") or {}).get("name")) or ""
            sc = item.get("score")
            if nm:
                predictions.append({"drug": nm, "score": sc})
                top_rec.append(nm)

    # 若为空，尝试从 last_scores
    if not predictions:
        for s in getattr(orch, "last_scores", []) or []:
            hid = s.get("hypothesis_id")
            # 找回对应 drug 名
            name = None
            for h in orch.hypotheses:
                if h.get("id") == hid:
                    name = (((h or {}).get("candidate") or {}).get("name"))
                    break
            if name:
                predictions.append({"drug": name, "score": s.get("score")})
        # 排序
        predictions = sorted(
            predictions,
            key=lambda x: (float(x.get("score") or 0.0)),
            reverse=True
        )
        top_rec = [p["drug"] for p in predictions]

    record = {
        "sample_index": task_index,
        "subset": subset,
        "disease": disease,
        "relation": relation,
        "gold_drugs": gold,
        "seeds": seeds,
        "predictions": predictions,       # [{"drug":..,"score":..}, ...]
        "ranking": top_rec,               # 仅名称排序
        "top_recommendations": top_rec[:10],
        "run_dir": orch.run_dir,
        "note": "ok"
    }

    # 额外保存 per-run meta（方便定位）
    try:
        with open(os.path.join(orch.run_dir, "meta_eval.json"), "w", encoding="utf-8") as f:
            json.dump({
                "subset": subset,
                "disease": disease,
                "relation": relation,
                "gold_drugs": gold,
                "seeds_initial": seeds,
                "seed_history": getattr(orch, "seed_history", []),
                "final_report": final_report
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return record

# ========= 主入口：并发 + 续跑 =========
def run_eval(ind_csv: Optional[str], ctr_csv: Optional[str], out_dir: str,
             n_seeds: int = 12, workers: int = 4,
             resume: bool = True, rerun_failed: bool = False,
             max_samples: Optional[int] = None):
    os.makedirs(out_dir, exist_ok=True)

    tasks: List[Tuple[str, Dict[str, Any]]] = []  # (subset, task)
    if ind_csv and os.path.exists(ind_csv):
        for i, t in enumerate(load_tasks_from_filtered_csv(ind_csv, relation="indication")):
            tasks.append(("indication", t))
    if ctr_csv and os.path.exists(ctr_csv):
        for i, t in enumerate(load_tasks_from_filtered_csv(ctr_csv, relation="contraindication")):
            tasks.append(("contraindication", t))

    if max_samples is not None:
        tasks = tasks[:max_samples]

    # 记录文件
    rec_ind = os.path.join(out_dir, "indication", "_records.jsonl")
    rec_ctr = os.path.join(out_dir, "contraindication", "_records.jsonl")
    if not resume:
        for p in [rec_ind, rec_ctr]:
            if os.path.exists(p):
                os.remove(p)

    # 续跑：已完成集合（按 subset 分开）
    done_ind = load_processed_indices_from_records(rec_ind, rerun_failed=rerun_failed)
    done_ctr = load_processed_indices_from_records(rec_ctr, rerun_failed=rerun_failed)

    def _work(idx_task: int, subset: str, task: Dict[str, Any]):
        # 用 tasks 的索引作为 sample_index（全局唯一）
        out_rec_path = rec_ind if subset == "indication" else rec_ctr
        # 续跑判断
        done_set = done_ind if subset == "indication" else done_ctr
        if idx_task in done_set:
            print(f"[SKIP] {subset} sample #{idx_task} already recorded.")
            return idx_task

        # 跑
        try:
            rec = run_one_task(idx_task, task, out_dir, n_seeds=n_seeds)
        except Exception as e:
            log_sample_error(out_dir, subset, idx_task, "worker", e, {"disease": task.get("disease")})
            rec = {
                "sample_index": idx_task,
                "subset": subset,
                "disease": task.get("disease"),
                "relation": task.get("relation"),
                "gold_drugs": task.get("gold_drugs", []),
                "seeds": [],
                "predictions": [],
                "ranking": [],
                "top_recommendations": [],
                "run_dir": None,
                "note": "runtime_error:worker"
            }
        # 写记录
        append_jsonl(out_rec_path, rec)
        return idx_task

    # 并发
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for idx, (subset, task) in enumerate(tasks):
                futs.append(ex.submit(_work, idx, subset, task))
            for fut in as_completed(futs):
                try:
                    idx_done = fut.result()
                    print(f"[OK] sample #{idx_done} done/skipped.")
                except Exception as e:
                    print(f"[ERR] worker error: {e}")
    else:
        for idx, (subset, task) in enumerate(tasks):
            _work(idx, subset, task)
            print(f"[OK] sample #{idx} done/skipped.")

# ========= CLI =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered_indication", type=str, default="filtered_indication_all.csv")
    ap.add_argument("--filtered_contra", type=str, default="filtered_contraindication_all.csv")
    ap.add_argument("--out_dir", type=str, default="runs_eval")
    ap.add_argument("--n_seeds", type=int, default=12, help="Explorer 生成的候选个数")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_resume", action="store_true", help="禁用续跑（清空 _records.jsonl）")
    ap.add_argument("--rerun_failed", action="store_true", help="续跑时仅重跑失败样本")
    ap.add_argument("--max_samples", type=int, default=-1, help="-1 表示全部")
    args = ap.parse_args()

    max_s = None if args.max_samples is None or args.max_samples < 0 else args.max_samples

    run_eval(
        ind_csv=args.filtered_indication if os.path.exists(args.filtered_indication) else None,
        ctr_csv=args.filtered_contra if os.path.exists(args.filtered_contra) else None,
        out_dir=args.out_dir,
        n_seeds=args.n_seeds,
        workers=max(1, args.workers),
        resume=(not args.no_resume),
        rerun_failed=args.rerun_failed,
        max_samples=max_s
    )

if __name__ == "__main__":
    main()
