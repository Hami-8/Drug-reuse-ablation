# run_baseline_single_llm.py  (fixed)
# 单次 LLM 基线：输入 Query(disease + relation)，输出药物名列表（作为 seeds）
# 结果写入 <out_dir>/<ModelTag>/<subset>/_records.jsonl

import os
import re
import json
import argparse
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import run_pipeline2 as rp2  # 复用其 OpenAI client / 代理设置

# ----------------- 通用 I/O & 并发安全 -----------------
_FILE_LOCKS: Dict[str, threading.Lock] = {}
_GUARD = threading.Lock()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def file_lock(path: str) -> threading.Lock:
    with _GUARD:
        if path not in _FILE_LOCKS:
            _FILE_LOCKS[path] = threading.Lock()
        return _FILE_LOCKS[path]

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

# ----------------- 载入 *_all.csv 并抽取 gold -----------------
POS_CONCLUSIONS = {"适应症线索","可能适应症且需注意安全"}

def load_all_csv_with_labels(path: str, subset: str) -> pd.DataFrame:
    """
    返回列：disease, normalized_drug, label(0/1)
    subset: "indication" 或 "contraindication"
    优先 is_gold_*；若缺失回退到 conclusion 文本；(disease, normalized_drug) 去重后 label 取 max。
    """
    df = pd.read_csv(path)
    disease_col = "disease" if "disease" in df.columns else df.columns[0]
    nd_col = "normalized_drug" if "normalized_drug" in df.columns else df.columns[-3]
    concl_col = "conclusion" if "conclusion" in df.columns else None

    df[disease_col] = df[disease_col].astype(str).str.strip()
    df[nd_col] = df[nd_col].astype(str).str.strip()
    if concl_col:
        df[concl_col] = df[concl_col].astype(str).str.strip()

    if subset == "indication":
        if "is_gold_indication" in df.columns:
            lab = df["is_gold_indication"].fillna(0).astype(int)
        else:
            lab = df[concl_col].apply(lambda x: 1 if str(x) in POS_CONCLUSIONS else 0).astype(int)
    else:
        if "is_gold_contraindication" in df.columns:
            lab = df["is_gold_contraindication"].fillna(0).astype(int)
        else:
            lab = df[concl_col].apply(lambda x: 1 if str(x) == "禁忌信号" else 0).astype(int)

    out = pd.DataFrame({
        "disease": df[disease_col],
        "normalized_drug": df[nd_col],
        "label": lab
    })
    out = out[(out["disease"].astype(str).str.len() > 0) & (out["normalized_drug"].astype(str).str.len() > 0)]
    out = out.groupby(["disease", "normalized_drug"], as_index=False)["label"].max()
    return out

# ----------------- 解析与提取 -----------------
CODE_BLOCK = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", flags=re.S)
BRACKET = re.compile(r"(\[.*\])", flags=re.S)

def _safe_parse_list(text: str, n: int) -> List[str]:
    """
    尝试把模型输出解析为字符串数组（最多 n 个）。
    优先 JSON 数组；否则逐行/分号/逗号切分。
    """
    if not text:
        return []
    # 1) 直接 JSON 数组
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            out = [str(x).strip() for x in obj if isinstance(x, (str, int, float))]
            uniq, seen = [], set()
            for s in out:
                ss = s.strip()
                if ss and ss.lower() not in seen:
                    uniq.append(ss); seen.add(ss.lower())
                if len(uniq) >= n: break
            return uniq
        if isinstance(obj, dict) and "answers" in obj and isinstance(obj["answers"], list):
            out = [str(x).strip() for x in obj["answers"] if isinstance(x, (str, int, float))]
            uniq, seen = [], set()
            for s in out:
                ss = s.strip()
                if ss and ss.lower() not in seen:
                    uniq.append(ss); seen.add(ss.lower())
                if len(uniq) >= n: break
            return uniq
    except Exception:
        pass
    # 2) 代码块中的 JSON
    m = CODE_BLOCK.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list):
                out = [str(x).strip() for x in obj if isinstance(x, (str, int, float))]
                uniq, seen = [], set()
                for s in out:
                    ss = s.strip()
                    if ss and ss.lower() not in seen:
                        uniq.append(ss); seen.add(ss.lower())
                    if len(uniq) >= n: break
                return uniq
        except Exception:
            pass
    # 3) 第一个中括号片段
    m2 = BRACKET.search(text)
    if m2:
        try:
            obj = json.loads(m2.group(1))
            if isinstance(obj, list):
                out = [str(x).strip() for x in obj if isinstance(x, (str, int, float))]
                uniq, seen = [], set()
                for s in out:
                    ss = s.strip()
                    if ss and ss.lower() not in seen:
                        uniq.append(ss); seen.add(ss.lower())
                    if len(uniq) >= n: break
                return uniq
        except Exception:
            pass
    # 4) 兜底：按行/分隔符切
    raw = re.split(r"[\n;,]+", text)
    out = []
    seen = set()
    for s in raw:
        ss = s.strip().strip("-*•").strip()
        if not ss: continue
        ss = re.sub(r"^\d+[\).\s-]+", "", ss).strip()  # 去掉开头编号
        if not ss: continue
        if ss.lower() in seen: continue
        out.append(ss); seen.add(ss.lower())
        if len(out) >= n: break
    return out

def _extract_responses_text(r) -> str:
    """
    稳健提取 responses.create 返回值中的纯文本。
    避免对 None 进行迭代。
    """
    # 1) 直接用 output_text
    raw = getattr(r, "output_text", None)
    if isinstance(raw, str) and raw.strip():
        return raw

    texts: List[str] = []

    # 2) 遍历 r.output（可能是 None / list / 其他）
    out_attr = getattr(r, "output", None)
    if isinstance(out_attr, list):
        for item in out_attr:
            # 可能是对象也可能是 dict
            content = None
            if hasattr(item, "content"):
                content = getattr(item, "content", None)
            elif isinstance(item, dict):
                content = item.get("content")
            if isinstance(content, list):
                for cc in content:
                    t = None
                    if hasattr(cc, "text"):
                        t = getattr(cc, "text", None)
                    elif isinstance(cc, dict):
                        t = cc.get("text") or cc.get("input_text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t)

    if texts:
        return "\n".join(texts)

    # 3) 最后兜底：把对象转字符串
    try:
        return str(r)
    except Exception:
        return ""

# ----------------- 单次 LLM 调用 -----------------
MODEL_TAGS = {
    "gpt-4o": "GPT-4o-single",
    "o3-mini": "O3-mini-single",
    "gpt-4o-mini": "GPT-4o-mini-single",
}

def call_single_llm(model: str, disease: str, relation: str, n: int, retries: int = 3) -> List[str]:
    """
    只调用一次 LLM，让其返回最多 n 个药名。
    - gpt-4o / gpt-4o-mini: 走 chat.completions 优先
    - o3-mini: 专走 responses.create（更稳）
    """
    client = rp2.get_client()
    sys = (
        "You are a biomedical assistant. "
        "Given a Query with a disease entity and a relation ('indication' or 'contraindication'), "
        f"list up to {n} unique drug names (generic preferred). "
        "Return ONLY a JSON array or an object {\"answers\": [...]}. No explanations."
    )
    user = {"query": {"entity": disease, "relation": relation}, "n": n}

    is_o3 = model.lower().startswith("o3")

    last_err = None
    for _ in range(retries + 1):
        try:
            if not is_o3:
                # ---- chat.completions 路径（gpt-*）----
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": sys},
                            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
                        ],
                        # 某些模型需要对象格式，这里允许两种返回
                        response_format={"type": "json_object"},
                        max_tokens=1200
                    )
                    txt = resp.choices[0].message.content
                    # 解析对象的 answers 或直接数组
                    try:
                        obj = json.loads(txt)
                        if isinstance(obj, dict) and "answers" in obj and isinstance(obj["answers"], list):
                            return _safe_parse_list(json.dumps(obj["answers"], ensure_ascii=False), n)
                    except Exception:
                        pass
                    out = _safe_parse_list(txt, n)
                    if out:
                        return out
                except Exception as e1:
                    last_err = e1  # fallthrough to responses

            # ---- responses 路径（o3 或 gpt-* 兜底）----
            try:
                r = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
                    ],
                    max_output_tokens=1200,
                )
                raw = _extract_responses_text(r)
                # 尝试对象 answers 或数组
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict) and "answers" in obj and isinstance(obj["answers"], list):
                        return _safe_parse_list(json.dumps(obj["answers"], ensure_ascii=False), n)
                except Exception:
                    pass
                out = _safe_parse_list(raw, n)
                if out:
                    return out
                last_err = RuntimeError("empty output from responses")
            except Exception as e2:
                last_err = e2

        except Exception as e:
            last_err = e

    raise RuntimeError(f"single LLM call failed: {last_err}")

# ----------------- 主执行（每模型/每疾病各一次调用） -----------------
def run_subset_with_model(
    subset: str,
    df_all: pd.DataFrame,
    out_dir: str,
    model: str,
    n: int,
    workers: int,
    resume: bool
):
    tag = MODEL_TAGS[model]
    subset_dir = os.path.join(out_dir, tag, subset)
    ensure_dir(subset_dir)
    jsonl_path = os.path.join(subset_dir, "_records.jsonl")

    done = read_done_indices(jsonl_path) if resume else set()

    # 唯一病种 + gold
    diseases = sorted(df_all["disease"].astype(str).unique().tolist())
    jobs = []
    for i, dis in enumerate(diseases):
        if i in done:
            continue
        gold = df_all[(df_all["disease"] == dis) & (df_all["label"] == 1)]["normalized_drug"].astype(str).tolist()
        jobs.append((i, dis, gold))

    relation = "indication" if subset == "indication" else "contraindication"
    print(f"[{tag}/{subset}] planned={len(jobs)} (resume={resume}, done={len(done)})")

    def _work(job):
        idx, disease, gold_drugs = job
        try:
            answers = call_single_llm(model=model, disease=disease, relation=relation, n=n)
            # 去重保序
            uniq, seen = [], set()
            for s in answers:
                if not isinstance(s, str): continue
                ss = s.strip()
                if ss and ss.lower() not in seen:
                    uniq.append(ss); seen.add(ss.lower())
            seeds = uniq[:n]

            rec = {
                "sample_index": idx,
                "subset": subset,
                "disease": disease,
                "relation": relation,
                "gold_drugs": gold_drugs,
                "seeds": seeds,                # 评测读取这个
                "predictions": [],
                "ranking": seeds,              # 顺序即排名
                "top_recommendations": seeds[:min(3, len(seeds))],
                "run_dir": None,
                "config": {"config_tag": tag, "flags": {"single_call": True, "n": n, "model": model}},
                "note": "ok"
            }
            append_jsonl(jsonl_path, rec)
            return idx

        except Exception as e:
            rec = {
                "sample_index": idx,
                "subset": subset,
                "disease": disease,
                "relation": relation,
                "gold_drugs": gold_drugs,
                "seeds": [],
                "predictions": [],
                "ranking": [],
                "top_recommendations": [],
                "run_dir": None,
                "config": {"config_tag": tag, "flags": {"single_call": True, "n": n, "model": model}},
                "note": f"runtime_error:{type(e).__name__}: {e}"
            }
            append_jsonl(jsonl_path, rec)
            return idx

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work, job) for job in jobs]
            for f in as_completed(futs):
                try:
                    i = f.result()
                    print(f"[{tag}/{subset}] done #{i}")
                except Exception as e:
                    print(f"[{tag}/{subset}] worker error: {e}")
    else:
        for job in jobs:
            _work(job)

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["indication","contraindication"], required=True)
    ap.add_argument("--all_csv", type=str, default="filtered_contraindication_all.csv", help="filtered_*_all.csv")
    ap.add_argument("--out_dir", type=str, default="runs_baseline")
    ap.add_argument("--model", choices=["gpt-4o","o3-mini","gpt-4o-mini","all"], default="all")
    ap.add_argument("--n", type=int, default=12, help="每个 Query 生成的答案数量")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = load_all_csv_with_labels(args.all_csv, subset=args.subset)

    models = ["gpt-4o","o3-mini","gpt-4o-mini"] if args.model == "all" else [args.model]
    for m in models:
        print(f"\n==== Running baseline model: {m} ====")
        run_subset_with_model(
            subset=args.subset,
            df_all=df,
            out_dir=args.out_dir,
            model=m,
            n=args.n,
            workers=args.workers,
            resume=args.resume
        )

    print("\nAll baseline runs finished. See <out_dir>/<ModelTag>/<subset>/_records.jsonl")

if __name__ == "__main__":
    main()
