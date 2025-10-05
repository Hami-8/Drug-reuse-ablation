# run_baseline_single_llm.py
# 单次调用 LLM 的 baseline（不做 self-consistency）
# 每病种调用一次模型，输出 seeds（最多 n 个）到 <out_dir>/<ConfigTag>/<subset>/_records.jsonl
#
# 用法示例：
#   # 仅跑 indication，模型 gpt-4o，每病种 50 个答案
#   python run_baseline_single_llm.py --subset indication \
#       --all_csv filtered_indication_all.csv \
#       --out_dir runs_baseline_1call --model gpt-4o --n 50
#
#   # 跑三种模型（各自独立落盘）
#   python run_baseline_single_llm.py --subset contraindication \
#       --all_csv filtered_contraindication_all.csv \
#       --out_dir runs_baseline_1call --model all --n 12

import os, re, json, time, random, argparse, threading
from typing import List, Dict, Any, Tuple, Optional, Set
import pandas as pd

# ========== OpenAI client ==========
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("缺少 OPENAI_API_KEY。请在环境变量或 .env 中设置。")

_thread_local = threading.local()
def get_client() -> OpenAI:
    cli = getattr(_thread_local, "client", None)
    if cli is None:
        cli = OpenAI(api_key=api_key)
        _thread_local.client = cli
    return cli

# ========== 解析辅助（与 run_pipeline2.py 对齐的鲁棒 JSON 提取） ==========
PUNCT_TO_SPACE = re.compile(r"[®™\-\_/\\,;:()\[\]%]+")
MULTISPACE = re.compile(r"\s+")

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

# ========== 名称清洗/去重（用于 seeds 去重与过滤噪声） ==========
SALT_OR_FORM_WORDS = {
    "sodium","potassium","hydrochloride","hydrobromide","sulfate","phosphate",
    "acetate","tartrate","mesylate","citrate","maleate","succinate","nitrate",
    "gel","pill","capsule","capsules","infusion","dpi","ointment","solution",
    "drops","eye","oral","product","placebo","tablet","tablets","injection",
    "spray","cream","patch","extended","release","er","xr","sr","mg","ug","µg",
    "soft","softgel","intravenous","intranasal","nasal","topical"
}

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

def canonical(s: str) -> str:
    return normalize_name(s)

def looks_like_drug_name(s: str) -> bool:
    if not isinstance(s, str): return False
    s = s.strip()
    if not s or len(s) > 80: return False
    bad_bits = ["Response(", "IncompleteDetails", "instructions=", "metadata=", "object=",
                "model=", "error=", "created_at=", "summary=", "type=", "output=", "content="]
    if any(b in s for b in bad_bits): return False
    if re.search(r"[={}:\[\]]", s): return False
    if not re.search(r"[A-Za-z]", s): return False
    return True

def dedup_ordered(names: List[str], limit: int) -> List[str]:
    uniq, seen = [], set()
    for x in names:
        if not looks_like_drug_name(x):
            continue
        c = canonical(x)
        if c and c not in seen:
            uniq.append(x.strip())
            seen.add(c)
        if len(uniq) >= limit:
            break
    return uniq

# ========== I/O 工具 ==========
_file_locks: Dict[str, threading.Lock] = {}
_guard = threading.Lock()
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

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

# ========== 读取 *_all.csv 并给出正例（用于 gold_drugs 字段） ==========
def load_all_csv_with_labels(path: str, subset: str) -> pd.DataFrame:
    """
    返回列：disease, normalized_drug, label(0/1)
    subset: "indication" 或 "contraindication"
    打标优先用 is_gold_indication / is_gold_contraindication；缺失则回退到 conclusion 文本。
    去重：(disease, normalized_drug) label 取 max。
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
            pos_set = {"适应症线索", "可能适应症且需注意安全"}
            lab = df[concl_col].apply(lambda x: 1 if str(x) in pos_set else 0).astype(int)
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

# ========== LLM 单次调用 ==========
SYSTEM_PROMPT_IND = (
    "You are an expert biomedical researcher.\n"
    "Given a rare disease, list N drugs that could be indicated or repurposed to treat it.\n"
    "Return ONLY a JSON object: {\"answers\": [<drug1>, <drug2>, ...]}.\n"
    "No explanations. Names only. Keep at most N items. Avoid duplicates."
)
SYSTEM_PROMPT_CONTRA = (
    "You are an expert biomedical researcher.\n"
    "Given a rare disease, list N drugs that are contraindicated or should be avoided.\n"
    "Return ONLY a JSON object: {\"answers\": [<drug1>, <drug2>, ...]}.\n"
    "No explanations. Names only. Keep at most N items. Avoid duplicates."
)

def build_user_payload(disease: str, relation: str, n: int) -> dict:
    return {
        "query": {"entity": disease, "relation": relation},
        "N": int(n)
    }

def call_llm_one_shot(model: str, system_prompt: str, user_payload: dict,
                      max_output_tokens: int = 2000,
                      temperature: float = 0.7,
                      retries: int = 4) -> dict:
    """
    - gpt-* 用 chat.completions + response_format json_object
    - o3-mini 用 responses.create；为了兼容老版 SDK：
        先尝试带 response_format={"type":"json_object"}，
        如果 TypeError（不支持该参数）或 400（提示不支持），再不带该参数重试。
    返回：dict（尽量包含 {"answers":[...]}）
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            client = get_client()
            if model.startswith("o3"):
                # --- 优先尝试带 response_format ---
                def _call_o3(with_response_format: bool):
                    if with_response_format:
                        return client.responses.create(
                            model=model,
                            input=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                            ],
                            # 兼容新接口时有效；老接口会抛 TypeError
                            response_format={"type": "json_object"},
                            max_output_tokens=max_output_tokens
                        )
                    else:
                        return client.responses.create(
                            model=model,
                            input=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                            ],
                            max_output_tokens=max_output_tokens
                        )

                try:
                    r = _call_o3(with_response_format=True)
                except TypeError as te:
                    # 老版 SDK：不支持 response_format；去掉后重试
                    r = _call_o3(with_response_format=False)
                except Exception as e:
                    # 某些服务端会返回 400 提示不支持该参数；同样去掉后重试
                    if "response_format" in str(e):
                        r = _call_o3(with_response_format=False)
                    else:
                        raise

                # 提取文本
                raw = getattr(r, "output_text", None)
                if not raw:
                    chunks = []
                    for out in getattr(r, "output", []):
                        for c in getattr(out, "content", []):
                            if getattr(c, "type", "") == "output_text":
                                chunks.append(getattr(c, "text", ""))
                    raw = "\n".join([x for x in chunks if x])

                # 尝试按 JSON 解析；失败则用文本抽取
                try:
                    obj = _safe_json_loads(_clean_json_text(raw))
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
                return {"answers": extract_answers(raw)}

            else:
                # gpt-* 系列：chat.completions + json_object
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": (
                            system_prompt +
                            "\n\nIMPORTANT: 只返回一个 JSON 对象；不要输出解释、不要输出 Markdown。"
                        )},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                    ],
                    response_format={"type": "json_object"},
                    temperature=float(temperature),
                    max_tokens=max_output_tokens
                )
                text = resp.choices[0].message.content
                return _safe_json_loads(_clean_json_text(text))

        except Exception as e:
            last_err = e
            time.sleep(min(10.0, 0.6 * (2 ** attempt)) + random.uniform(0, 0.3))

    raise RuntimeError(f"single LLM call failed: {last_err}")


def extract_answers(obj: Any) -> List[str]:
    """
    从模型返回的 JSON 或文本提取答案列表；尽量鲁棒。
    """
    # 已是 JSON 且有 answers
    if isinstance(obj, dict) and isinstance(obj.get("answers"), list):
        return [x for x in obj["answers"] if isinstance(x, str)]

    # 如果传进来的是字符串（兜底）：尝试解析数组；否则按逗号/换行切分
    if isinstance(obj, str):
        s = obj.strip()
        # 试图解析 {"answers":[...]} 或 [...]：
        try:
            j = json.loads(s)
            if isinstance(j, dict) and isinstance(j.get("answers"), list):
                return [x for x in j["answers"] if isinstance(x, str)]
            if isinstance(j, list):
                return [x for x in j if isinstance(x, str)]
        except Exception:
            pass
        # 文本切分兜底
        parts = re.split(r"[\n,;]+", s)
        return [p.strip() for p in parts if p.strip()]

    return []

# ========== 主流程（单次调用基线） ==========
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_single_model(subset: str, all_csv: str, out_dir: str,
                     model: str, n: int, workers: int,
                     resume: bool, temperature: float,
                     max_output_tokens: int):
    df = load_all_csv_with_labels(all_csv, subset=subset)

    # 配置 tag
    model_tag = {
        "gpt-4o": "GPT-4o-1call",
        "gpt-4o-mini": "GPT-4o-mini-1call",
    }.get(model, "O3-mini-1call") if model != "all" else "ALL-1call"  # 实际会分模型各自目录
    # 统一 relation
    relation = "indication" if subset == "indication" else "contraindication"

    # 病种列表
    diseases = sorted(df["disease"].astype(str).unique().tolist())

    # 每模型一个目录
    cfg_dir = os.path.join(out_dir, (model_tag if model != "all" else "")) if model != "all" else out_dir

    def process_one_model(one_model: str):
        tag = {
            "gpt-4o": "GPT-4o-1call",
            "gpt-4o-mini": "GPT-4o-mini-1call",
        }.get(one_model, "O3-mini-1call")
        subset_dir = os.path.join(cfg_dir, tag, subset)
        ensure_dir(subset_dir)
        jsonl_path = os.path.join(subset_dir, "_records.jsonl")
        done = read_done_indices(jsonl_path) if resume else set()

        print(f"[{tag}/{subset}] total diseases={len(diseases)} (resume={resume}, done={len(done)})")

        def _work(i_dis):
            idx = i_dis
            dis = diseases[idx]
            try:
                # gold（来自 *_all.csv 的 label==1）
                gold = df[(df["disease"] == dis) & (df["label"] == 1)]["normalized_drug"].astype(str).tolist()

                # 构造提示
                system_prompt = SYSTEM_PROMPT_IND if subset == "indication" else SYSTEM_PROMPT_CONTRA
                user_payload = build_user_payload(dis, relation, n)

                # 单次调用
                raw = call_llm_one_shot(
                    model=one_model,
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature
                )
                answers = extract_answers(raw)
                seeds = dedup_ordered(answers, n)

                # 排序/推荐（此基线没有打分，直接用 seeds 顺序）
                ranking = list(seeds)
                top_recs = seeds[:min(3, len(seeds))]

                rec = {
                    "sample_index": idx,
                    "subset": subset,
                    "disease": dis,
                    "relation": relation,
                    "gold_drugs": gold,
                    "seeds": seeds,
                    "predictions": [],             # 单次基线不产打分
                    "ranking": ranking,            # 用 seeds 顺序
                    "top_recommendations": top_recs,
                    "run_dir": None,
                    "config": {"config_tag": tag, "flags": {"single_call_sc": False, "n": n, "model": one_model}},
                    "note": "ok"
                }
                append_jsonl(jsonl_path, rec)
                return idx
            except Exception as e:
                rec = {
                    "sample_index": idx,
                    "subset": subset,
                    "disease": dis,
                    "relation": relation,
                    "gold_drugs": [],
                    "seeds": [],
                    "predictions": [],
                    "ranking": [],
                    "top_recommendations": [],
                    "run_dir": None,
                    "config": {"config_tag": tag, "flags": {"single_call_sc": False, "n": n, "model": one_model}},
                    "note": f"runtime_error:{type(e).__name__}: {e}"
                }
                append_jsonl(jsonl_path, rec)
                return idx

        # 任务列表
        todo = [i for i in range(len(diseases)) if (i not in done)]
        if not todo:
            print(f"[{tag}/{subset}] nothing to do (all done).")
            return

        if workers and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_work, i) for i in todo]
                for f in as_completed(futs):
                    try:
                        i_done = f.result()
                        print(f"[{tag}/{subset}] done #{i_done}")
                    except Exception as e:
                        print(f"[{tag}/{subset}] worker error: {e}")
        else:
            for i in todo:
                _work(i)
                print(f"[{tag}/{subset}] done #{i}")

    # 执行
    if model == "all":
        for m in ["gpt-4o", "gpt-4o-mini", "o3-mini"]:
            process_one_model(m)
    else:
        process_one_model(model)

# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["indication","contraindication"], default="contraindication")
    ap.add_argument("--all_csv", type=str, default="filtered_contraindication_all.csv", help="filtered_*_all.csv")
    ap.add_argument("--out_dir", type=str, default="runs_con_top50")
    ap.add_argument("--model", type=str, default="o3-mini", help="'gpt-4o' | 'gpt-4o-mini' | 'o3-mini' | 'all'")
    ap.add_argument("--n", type=int, default=50, help="每个 Query 最终输出的 seeds 数量")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7, help="仅对 gpt-* 生效；o3-mini 忽略")
    ap.add_argument("--max_tokens", type=int, default=4000, help="最大输出 tokens（o3-mini 用 max_output_tokens）")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    run_single_model(
        subset=args.subset,
        all_csv=args.all_csv,
        out_dir=args.out_dir,
        model=args.model,
        n=args.n,
        workers=args.workers,
        resume=args.resume,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens
    )

if __name__ == "__main__":
    main()
