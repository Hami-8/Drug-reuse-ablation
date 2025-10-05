# run_baseline_single_llm.py  (self-consistency, robust parsing & noise filter for o3-mini)
# 目的：
#   - 不用 RareAgent，直接用单模型多次采样（~15 次）做 self-consistency 聚合为 seeds
#   - 兼容 gpt-4o / gpt-4o-mini / o3-mini
#   - o3-mini：不传 temperature；仅抽取 output_text；加入噪声清洗，避免把 Response(...) 等串进 seeds
#
# 用法：
#   python run_baseline_single_llm.py \
#     --subset indication \
#     --all_csv filtered_indication_all.csv \
#     --out_dir runs_baseline_sc \
#     --model all --n 12 --sc_samples 15

import os
import re
import json
import argparse
import threading
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import run_pipeline2 as rp2  # 复用其 get_client / 环境设置

# =========================================================
# 并发安全 I/O
# =========================================================
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
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "sample_index" in obj:
                    done.add(int(obj["sample_index"]))
            except Exception:
                continue
    return done

# =========================================================
# 规范化 / 同义宽松
# =========================================================
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
    x = s.lower().replace("+", " + ")
    x = PUNCT_TO_SPACE.sub(" ", x)
    toks = [t for t in MULTISPACE.sub(" ", x).strip().split() if t]
    toks = [t for t in toks if t not in SALT_OR_FORM_WORDS]
    return MULTISPACE.sub(" ", " ".join(toks)).strip()

ALIAS_GROUPS = [
    {"sodium valproate","valproate","valproic acid"},
    {"ursodiol","actigall"},
    {"aztreonam lysine","aztreonam for inhalation","azli"},
    {"omega 3 fatty acids","omega 3 acid ethyl esters","epa dha","fish oil","lovaza","ethyl icosapentate"},
    {"somatropin","growth hormone"},
    {"risedronate","risedronate sodium"},
    {"nuedexta","dextromethorphan + quinidine","dextromethorphan quinidine"},
]

def _alias_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for g in ALIAS_GROUPS:
        rep = sorted(g, key=lambda z: (len(z), z))[0]
        for item in g:
            m[normalize_name(item)] = rep
    return m

ALIAS_MAP = _alias_map()

def canonical(s: str) -> str:
    n = normalize_name(s)
    return ALIAS_MAP.get(n, n)

# =========================================================
# 数据载入：从 *_all.csv 读疾病 & gold
# =========================================================
POS_CONCLUSIONS = {"适应症线索","可能适应症且需注意安全"}

def load_all_csv_with_labels(path: str, subset: str) -> pd.DataFrame:
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

# =========================================================
# LLM 解析 & 噪声过滤
# =========================================================
CODE_BLOCK = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", flags=re.S)
BRACKET = re.compile(r"(\[.*\])", flags=re.S)

def _clean_json_text(s: str) -> str:
    if not s: return s
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.S)
    return m.group(1).strip() if m else s.strip()

def _safe_json_loads(text: str) -> dict:
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    s = text.find("{")
    if s != -1:
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[s:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    raise ValueError("cannot parse JSON from LLM output")

def _extract_responses_text(r) -> str:
    """
    仅提取 responses.create 的 output_text；若没有则返回空串
    —— 避免把 Response(...) 对象 repr 拼进 seeds。
    """
    raw = getattr(r, "output_text", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    texts: List[str] = []
    out_attr = getattr(r, "output", None)
    # 兼容属性对象/字典
    if isinstance(out_attr, list):
        for item in out_attr:
            content = getattr(item, "content", None) if hasattr(item, "content") else (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, list):
                for cc in content:
                    # 仅收集 output_text
                    typ = getattr(cc, "type", None) if hasattr(cc, "type") else (cc.get("type") if isinstance(cc, dict) else None)
                    if typ == "output_text":
                        t = getattr(cc, "text", None) if hasattr(cc, "text") else (cc.get("text") if isinstance(cc, dict) else None)
                        if isinstance(t, str) and t.strip():
                            texts.append(t)
    if texts:
        return "\n".join(texts)
    # 返回空串（不要回退到 repr(r)）
    return ""

_NOISE_PAT = re.compile(
    r"(Response\(|error=|incomplete_details|instructions=|metadata=|object=|created_at=|content=|ResponseReasoningItem|id='resp_|type=)",
    flags=re.I
)

def is_noise_token(s: str) -> bool:
    if not isinstance(s, str): return True
    ss = s.strip()
    if not ss: return True
    if len(ss) > 80: return True
    if _NOISE_PAT.search(ss): return True
    if ss.lower() in {"none","null","n/a"}: return True
    if re.search(r"^\w+\s*=\s*", ss): return True
    if re.fullmatch(r"[\[\]\{\}\(\)]+", ss): return True
    return False

def plausible_drug_name(s: str) -> bool:
    n = normalize_name(s)
    if not n: return False
    if re.search(r"[=:\{\}\[\]\(\)]", s): return False
    # 需包含字母
    if not re.search(r"[a-z]", n): return False
    # 单词数合理
    if len(n.split()) > 6: return False
    return True

def _filter_and_dedup(ans: List[str], cap_n: int) -> List[str]:
    out, seen = [], set()
    for s in ans:
        if not isinstance(s, str): continue
        ss = s.strip()
        if not ss: continue
        if is_noise_token(ss): continue
        if not plausible_drug_name(ss): continue
        key = ss.lower()
        if key in seen: continue
        out.append(ss)
        seen.add(key)
        if len(out) >= cap_n: break
    return out

def _safe_parse_list(text: str, n: int) -> List[str]:
    """
    优先 JSON：数组 或 {"answers":[...]}；否则代码块/中括号/兜底分隔 + 噪声过滤。
    """
    if not text:
        return []
    # 1) JSON 对象或数组
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            src = obj
        elif isinstance(obj, dict) and "answers" in obj and isinstance(obj["answers"], list):
            src = obj["answers"]
        else:
            src = None
        if src is not None:
            raw = [str(x).strip() for x in src if isinstance(x, (str, int, float))]
            return _filter_and_dedup(raw, n)
    except Exception:
        pass
    # 2) 代码块 JSON 数组
    m = CODE_BLOCK.search(text)
    if m:
        try:
            arr = json.loads(m.group(1))
            if isinstance(arr, list):
                raw = [str(x).strip() for x in arr if isinstance(x, (str, int, float))]
                return _filter_and_dedup(raw, n)
        except Exception:
            pass
    # 3) 第一个中括号片段
    m2 = BRACKET.search(text)
    if m2:
        try:
            arr = json.loads(m2.group(1))
            if isinstance(arr, list):
                raw = [str(x).strip() for x in arr if isinstance(x, (str, int, float))]
                return _filter_and_dedup(raw, n)
        except Exception:
            pass
    # 4) 兜底：按行/分隔符切
    raw = re.split(r"[\n;,]+", text)
    raw = [re.sub(r"^\d+[\).\s-]+", "", s).strip("-*• ").strip() for s in raw]
    return _filter_and_dedup([s for s in raw if s], n)

# =========================================================
# 单次 LLM 调用（o3-mini 不传 temperature；仅收集 output_text）
# =========================================================
MODEL_TAGS_SC = {
    "gpt-4o": "GPT-4o-SC",
    "o3-mini": "O3-mini-SC",
    "gpt-4o-mini": "GPT-4o-mini-SC",
}

def _prompt(relation: str, n: int) -> str:
    base = (
        "You are a biomedical assistant. "
        "Given a Query with a disease entity and a relation ('indication' or 'contraindication'), "
        f"list up to {n} unique drug names (generic preferred). "
        "Return ONLY a JSON array or an object {\"answers\": [...]}. No explanations."
    )
    return base

def _prompt_strict_json(relation: str, n: int) -> str:
    return (
        "Return ONLY a JSON array of drug names, nothing else. Example:\n"
        f"[\"drug a\", \"drug b\", \"drug c\"] (max {n} items)"
    )

def llm_once(model: str, disease: str, relation: str, n: int,
             temperature: float = 0.8, max_tokens_o3: int = 1500, max_tokens_gpt: int = 900) -> List[str]:
    client = rp2.get_client()
    sys = _prompt(relation, n)
    user = {"query": {"entity": disease, "relation": relation}, "n": n}
    is_o3 = model.lower().startswith("o3")

    # --- o3: responses（不传 temperature），只取 output_text ---
    if is_o3:
        # 尝试降低推理负载（如可用）
        kwargs = {
            "model": model,
            "input": [
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            "max_output_tokens": max_tokens_o3,
        }
        # 一些版本支持 reasoning.effort，可尝试；若报错 SDK 会忽略
        try:
            kwargs["reasoning"] = {"effort": "low"}
        except Exception:
            pass

        r = client.responses.create(**kwargs)
        raw = _extract_responses_text(r)
        ans = _safe_parse_list(raw, n)
        if len(ans) >= max(3, min(5, n)):  # 有一定数量即可
            return ans

        # 二次更严格提示重试
        r2 = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": sys + "\n" + _prompt_strict_json(relation, n)},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            max_output_tokens=max_tokens_o3,
        )
        raw2 = _extract_responses_text(r2)
        return _safe_parse_list(raw2, n)

    # --- gpt-*：chat.completions 优先 ---
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens_gpt
        )
        txt = resp.choices[0].message.content
        obj = None
        try:
            obj = _safe_json_loads(_clean_json_text(txt))
        except Exception:
            pass
        if isinstance(obj, dict) and "answers" in obj and isinstance(obj["answers"], list):
            out = _safe_parse_list(json.dumps(obj["answers"], ensure_ascii=False), n)
            if out: return out
        if isinstance(obj, list):
            out = _safe_parse_list(json.dumps(obj, ensure_ascii=False), n)
            if out: return out
        out = _safe_parse_list(txt, n)
        if out:
            return out
    except Exception:
        pass

    # --- gpt-* 兜底：responses（允许 temperature） ---
    r = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys + "\n" + _prompt_strict_json(relation, n)},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ],
        max_output_tokens=max_tokens_gpt,
        temperature=temperature,
    )
    raw = _extract_responses_text(r)
    return _safe_parse_list(raw, n)

# =========================================================
# 自洽聚合（self-consistency）
# =========================================================
def aggregate_self_consistency(samples: List[List[str]], cap_n: int) -> List[str]:
    score: Dict[str, float] = {}
    freq: Dict[str, int] = {}
    avg_rank: Dict[str, float] = {}
    first_form: Dict[str, str] = {}

    for sample in samples:
        Ni = max(1, len(sample))
        for r, s in enumerate(sample, start=1):
            can = canonical(s)
            if not can:
                continue
            w = 1.0 - (r - 1) / Ni
            score[can] = score.get(can, 0.0) + w
            freq[can] = freq.get(can, 0) + 1
            avg_rank[can] = avg_rank.get(can, 0.0) + r
            if can not in first_form:
                first_form[can] = s

    for k in avg_rank:
        avg_rank[k] = avg_rank[k] / max(1, freq.get(k, 1))

    ranked = sorted(score.keys(), key=lambda k: (-score[k], -freq[k], avg_rank[k], k))
    out = []
    seen = set()
    for can in ranked:
        nm = first_form.get(can, can)
        if nm.lower() in seen:
            continue
        out.append(nm)
        seen.add(nm.lower())
        if len(out) >= cap_n:
            break
    return out

# =========================================================
# 执行（每模型 / 每疾病做 sc_samples 次）
# =========================================================
MODEL_TAGS_SC = MODEL_TAGS_SC  # already defined

def run_subset_with_model_sc(
    subset: str,
    df_all: pd.DataFrame,
    out_dir: str,
    model: str,
    n: int,
    sc_samples: int,
    temperature: float,
    workers: int,
    resume: bool
):
    tag = f"{MODEL_TAGS_SC[model]}{sc_samples}"
    subset_dir = os.path.join(out_dir, tag, subset)
    ensure_dir(subset_dir)
    jsonl_path = os.path.join(subset_dir, "_records.jsonl")

    done = read_done_indices(jsonl_path) if resume else set()

    diseases = sorted(df_all["disease"].astype(str).unique().tolist())
    jobs: List[Tuple[int, str, List[str]]] = []
    for i, dis in enumerate(diseases):
        if i in done:
            continue
        gold = df_all[(df_all["disease"] == dis) & (df_all["label"] == 1)]["normalized_drug"].astype(str).tolist()
        jobs.append((i, dis, gold))

    relation = "indication" if subset == "indication" else "contraindication"
    print(f"[{tag}/{subset}] planned={len(jobs)} (resume={resume}, done={len(done)}), sc_samples={sc_samples}, n={n}")

    def _work(job):
        idx, disease, gold_drugs = job
        try:
            samples: List[List[str]] = []
            for _ in range(max(1, sc_samples)):
                ans = llm_once(model=model, disease=disease, relation=relation, n=n,
                               temperature=temperature)
                # 去重+噪声过滤（再次保险）
                ans = _filter_and_dedup(ans, n*2)
                samples.append(ans)

            seeds = aggregate_self_consistency(samples, cap_n=n)

            rec = {
                "sample_index": idx,
                "subset": subset,
                "disease": disease,
                "relation": relation,
                "gold_drugs": gold_drugs,
                "seeds": seeds,
                "predictions": [],
                "ranking": seeds,
                "top_recommendations": seeds[:min(3, len(seeds))],
                "run_dir": None,
                "config": {"config_tag": tag, "flags": {"single_call_sc": True, "n": n, "model": model, "sc_samples": sc_samples, "temperature": temperature}},
                "note": "ok" if seeds else "no_seeds"
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
                "config": {"config_tag": tag, "flags": {"single_call_sc": True, "n": n, "model": model, "sc_samples": sc_samples, "temperature": temperature}},
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

# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["indication","contraindication"], required=True)
    ap.add_argument("--all_csv", type=str, required=True, help="filtered_*_all.csv")
    ap.add_argument("--out_dir", type=str, default="runs_baseline_sc5")
    ap.add_argument("--model", choices=["gpt-4o","o3-mini","gpt-4o-mini","all"], default="all")
    ap.add_argument("--n", type=int, default=12, help="每个 Query 最终输出的 seeds 数量")
    ap.add_argument("--sc_samples", type=int, default=15, help="self-consistency 采样次数")
    ap.add_argument("--temperature", type=float, default=0.8, help="gpt-* 采样温度（o3-mini 自动忽略）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = load_all_csv_with_labels(args.all_csv, subset=args.subset)

    models = ["gpt-4o","o3-mini","gpt-4o-mini"] if args.model == "all" else [args.model]
    for m in models:
        print(f"\n==== Running baseline self-consistency model: {m} ====")
        run_subset_with_model_sc(
            subset=args.subset,
            df_all=df,
            out_dir=args.out_dir,
            model=m,
            n=args.n,
            sc_samples=args.sc_samples,
            temperature=args.temperature,
            workers=args.workers,
            resume=args.resume
        )

    print("\nAll baseline self-consistency runs finished. See <out_dir>/<ModelTag>SC<k>/<subset>/_records.jsonl")

if __name__ == "__main__":
    main()
