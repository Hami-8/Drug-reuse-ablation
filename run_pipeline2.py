# run_pipeline2.py

import os, json, time, copy, uuid, datetime, csv, argparse, re
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI

import threading, random

os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

# ===================== Model & Client =====================
MODEL = "gpt-4o-mini"

from dotenv import load_dotenv
load_dotenv()  # 读取 .env

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("缺少 OPENAI_API_KEY。请在环境变量或 .env 中设置。")

# 线程本地 client，避免并发共享一个 client
from openai import OpenAI
_thread_local = threading.local()

def get_client() -> OpenAI:
    cli = getattr(_thread_local, "client", None)
    if cli is None:
        cli = OpenAI(api_key=api_key)
        _thread_local.client = cli
    return cli

# ===================== Ablation Config =====================
def make_ablation_config(
    enable_debate: bool = True,
    enable_skeptic: bool = True,
    enable_pi_interrupts: bool = True,    # 关掉=只在最后一次性评分
    enable_textual_feedback: bool = True, # 关掉=忽略 prompt patches 的效果
    enable_heuristic_transfer: bool = True,
    enable_seed_regen: bool = True,
    path_disjointness_weight: float = 1.0,
    mode: str = "graph"                   # "graph" | "list_only"
) -> Dict[str, Any]:
    return {
        "enable_debate": bool(enable_debate),
        "enable_skeptic": bool(enable_skeptic),
        "enable_pi_interrupts": bool(enable_pi_interrupts),
        "enable_textual_feedback": bool(enable_textual_feedback),
        "enable_heuristic_transfer": bool(enable_heuristic_transfer),
        "enable_seed_regen": bool(enable_seed_regen),
        "weights": {"path_disjointness": float(path_disjointness_weight)},
        "mode": mode  # "list_only" 时跳过 T-EGraph / Proponent / Skeptic
    }

def load_processed_indices_from_records(path: str, rerun_failed: bool = False) -> set:
    """
    从 _records.jsonl 读取已处理的 sample_index 集合。
    - 默认：发现任何记录（成功/失败/无种子）均视为“已跑过”，跳过。
    - rerun_failed=True：遇到 note 以 'runtime_error' 开头的记录，不算已完成（允许重跑失败样本）。
    """
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
                # 失败样本不计入已完成集合（允许重跑）
                continue
            try:
                done.add(int(sid))
            except Exception:
                continue
    return done

# 默认宽松：Chat + json_object，避免严格 schema 报错（可用 STRICT_STRUCTURED_OUTPUT=1 切到严格）
STRICT_STRUCTURED_OUTPUT = os.environ.get("STRICT_STRUCTURED_OUTPUT", "0") in ("1","true","True")

# ===================== LLM 调用（宽松 JSON） =====================
def _clean_json_text(s: str) -> str:
    if not s: return s
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.S)
    return m.group(1).strip() if m else s.strip()

# ========= 辅助：更稳的 JSON 解析 =========
def _safe_json_loads(text: str) -> dict:
    """
    1) 直接 loads
    2) 提取 ```json ... ``` 或 ``` ... ``` 中间的对象
    3) 贪婪匹配第一个 { ... }（括号配平）
    失败则抛 ValueError
    """
    if not text:
        raise ValueError("empty response")

    # 直接
    try:
        return json.loads(text)
    except Exception:
        pass

    # ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # 配平第一个大括号
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
    # 彻底失败
    raise ValueError("cannot parse JSON from LLM output")

def call_llm(system_prompt: str, user_payload: dict, schema: dict = None,
             max_output_tokens: int = 3500, retries: int = 5) -> dict:  # 重试更稳
    last_err = None
    for attempt in range(retries + 1):
        try:
            client = get_client()  # 每次取线程本地 client

            if not STRICT_STRUCTURED_OUTPUT:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": (
                            system_prompt +
                            "\n\nIMPORTANT: 只返回一个 JSON 对象；不要输出解释、不要输出 Markdown。"
                        )},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=max_output_tokens
                )
                text = resp.choices[0].message.content
                return _safe_json_loads(_clean_json_text(text))

            # 严格模式分支（保持原样），同样用线程本地 client
            try:
                r = client.responses.create(
                    model=MODEL,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "agent_output", "schema": schema or {"type":"object","additionalProperties":True}, "strict": True}
                    },
                    max_output_tokens=max_output_tokens,
                )
                raw = getattr(r, "output_text", None)
                if not raw:
                    chunks = []
                    for out in getattr(r, "output", []):
                        for c in getattr(out, "content", []):
                            if getattr(c, "type", "") == "output_text":
                                chunks.append(getattr(c, "text", ""))
                    raw = "\n".join([x for x in chunks if x])
                return _safe_json_loads(_clean_json_text(raw))
            except Exception:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": (
                            system_prompt +
                            "\n\nIMPORTANT: 只返回一个 JSON 对象；不要输出解释、不要输出 Markdown。"
                        )},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=max_output_tokens
                )
                text = resp.choices[0].message.content
                return _safe_json_loads(_clean_json_text(text))

        except Exception as e:
            last_err = e
            # 指数退避 + 抖动，避免多线程一起撞限流
            backoff = min(10.0, (0.6 * (2 ** attempt))) + random.uniform(0, 0.3)
            time.sleep(backoff)

    raise RuntimeError(f"LLM call failed after retries: {last_err}")

# ===================== T-EGraph 基础 =====================
Graph = Dict[str, Any]  # {"nodes":[...], "edges":[...], "round_index": int}

def empty_graph(round_index: int = 1) -> Graph:
    return {"nodes": [], "edges": [], "round_index": round_index}

def merge_graph(base: Graph, delta: Graph) -> Graph:
    out = copy.deepcopy(base)
    node_index = {}
    for i, n in enumerate(out["nodes"]):
        nid = n.get("id") or f"{n.get('type')}-{n.get('label')}"
        node_index[nid] = i
    def add_node(n: Dict[str, Any]):
        nid = n.get("id") or f"{n.get('type')}-{n.get('label')}"
        if nid in node_index:
            existing = out["nodes"][node_index[nid]]
            attrs = existing.get("attrs", {})
            attrs.update(n.get("attrs", {}))
            existing["attrs"] = attrs
            return nid
        out["nodes"].append(n)
        node_index[nid] = len(out["nodes"]) - 1
        return nid
    for n in delta.get("nodes", []) or []:
        add_node(n)
    def ekey(e):
        return (e.get("source",""), e.get("target",""), e.get("relation",""))
    edge_map = {ekey(e): e for e in out["edges"]}
    for e in delta.get("edges", []) or []:
        k = ekey(e)
        if k in edge_map:
            exist = edge_map[k]
            exist["weight"] = max(float(exist.get("weight", 0.0)), float(e.get("weight", 0.0)))
            r_old = exist.get("rationale", [])
            if isinstance(r_old, str): r_old = [r_old]
            r_new = e.get("rationale", [])
            if isinstance(r_new, str): r_new = [r_new]
            exist["rationale"] = list({*r_old, *r_new})
            agents = set([exist.get("agent","")])
            agents.add(e.get("agent",""))
            exist["agent_sources"] = sorted([a for a in agents if a])
        else:
            out["edges"].append(e)
            edge_map[k] = e
    out["round_index"] = max(int(out.get("round_index",1)), int(delta.get("round_index",1)))
    return out

def apply_graph_updates(current: Graph, updates: Dict[str, Any]) -> Graph:
    # 以 updates.round_index 为准；若缺省则沿用 current
    r_idx = int(updates.get("round_index", current.get("round_index", 1)))

    new_graph = copy.deepcopy(current)
    add_nodes = updates.get("add_nodes", []) or []
    add_edges = updates.get("add_edges", []) or []

    # 用我们希望的 round_index 合并
    new_graph = merge_graph(new_graph, {
        "nodes": add_nodes,
        "edges": add_edges,
        "round_index": r_idx
    })

    for m in updates.get("merge", []) or []:
        keep, remove = m.get("keep"), m.get("remove")
        if not keep or not remove: continue
        for e in new_graph["edges"]:
            if e.get("source") == remove: e["source"] = keep
            if e.get("target") == remove: e["target"] = keep
        new_graph["nodes"] = [n for n in new_graph["nodes"] if n.get("id") != remove]

    for sw in updates.get("set_weights", []) or []:
        if sw.get("edge_id"):
            for e in new_graph["edges"]:
                if e.get("id")==sw["edge_id"]:
                    e["weight"] = sw.get("weight", e.get("weight"))
                    if sw.get("rationale"): e["rationale"] = sw["rationale"]
        else:
            s,t,r = sw.get("source"), sw.get("target"), sw.get("relation")
            for e in new_graph["edges"]:
                if e.get("source")==s and e.get("target")==t and e.get("relation")==r:
                    e["weight"] = sw.get("weight", e.get("weight"))
                    if sw.get("rationale"): e["rationale"] = sw["rationale"]

    # ★ 最终确保回合号写回
    new_graph["round_index"] = r_idx
    return new_graph

# ===================== Agents（PI/Proponent/Skeptic/Explorer） =====================
# 宽松 schema（仅严格模式会用；默认不会触发）
PI_INIT_SCHEMA   = {"type":"object","properties":{"plan":{"type":"object"}},"required":["plan"],"additionalProperties":False}
PI_SCORE_SCHEMA  = {"type":"object","properties":{"scoring_summary":{"type":"array"},"ranking":{"type":"array"},"delta_since_last_round":{"type":"number"},"stop_decision":{"type":"object"}},"required":["scoring_summary","ranking","stop_decision"],"additionalProperties":False}
PI_REVISE_SCHEMA = {"type":"object","properties":{"revisions":{"type":"array"},"pruned_hypotheses":{"type":"array"},"new_subhypotheses":{"type":"array"},"query":{"type":"object"},"notes":{"type":"string"}}, "required":["revisions"],"additionalProperties":False}
PI_REPORT_SCHEMA = {"type":"object","additionalProperties":True}

PROP_BUILD_SCHEMA= {"type":"object","properties":{"graph_updates":{"type":"object"}},"required":["graph_updates"],"additionalProperties":True}
PROP_EXEC_SCHEMA = {"type":"object","properties":{"hypothesis_id":{"type":"string"},"executed_actions":{"type":"array"},"graph_updates":{"type":"object"}},"required":["hypothesis_id"],"additionalProperties":True}
SKEP_BUILD_SCHEMA= {"type":"object","properties":{"graph_updates":{"type":"object"}},"required":["graph_updates"],"additionalProperties":True}
SKEP_EXEC_SCHEMA = {"type":"object","properties":{"hypothesis_id":{"type":"string"},"executed_actions":{"type":"array"},"graph_updates":{"type":"object"}},"required":["hypothesis_id"],"additionalProperties":True}

def run_pi(mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    schema = {"init":PI_INIT_SCHEMA,"score":PI_SCORE_SCHEMA,"revise":PI_REVISE_SCHEMA,"report_and_evolve":PI_REPORT_SCHEMA}.get(mode, PI_SCORE_SCHEMA)
    return call_llm(PI_SYSTEM_PROMPT, payload, schema, max_output_tokens=10000)

def run_proponent(mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    schema = PROP_EXEC_SCHEMA if mode=="execute_actions" else PROP_BUILD_SCHEMA
    return call_llm(PROPONENT_SYSTEM_PROMPT, payload, schema, max_output_tokens=10000)

def run_skeptic(mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    schema = SKEP_EXEC_SCHEMA if mode=="execute_actions" else SKEP_BUILD_SCHEMA
    return call_llm(SKEPTIC_SYSTEM_PROMPT, payload, schema, max_output_tokens=10000)

# —— Explorer Agent —— #
# 系统提示词在文件底部：EXPLORER_SYSTEM_PROMPT
def run_explorer(bio_prompt: str, query: Dict[str,Any], target_type: str, n_seeds: int = 50) -> List[str]:
    """
    Explorer 根据 BioHopR.prompt + Query（含 entity/relation/target_type，若有 regen_prompt 也一起传）
    产生候选种子：返回 ["name1", "name2", ...]
    """
    sys = EXPLORER_SYSTEM_PROMPT
    user = {
        "biohopr_prompt": bio_prompt,
        "query": {
            "entity": query.get("entity"),
            "relation": query.get("relation"),
            "target_type": target_type,
            # 若是 PI 给出的再生原因，这里会出现：
            "regen_prompt": query.get("regen_prompt", "")
        },
        "n": int(n_seeds)
    }
    try:
        out = call_llm(sys, user, max_output_tokens=1200)
        seeds = out.get("seeds") if isinstance(out, dict) else None
        if isinstance(seeds, list):
            uniq = []
            seen = set()
            for s in seeds:
                if not isinstance(s, str): continue
                ss = s.strip()
                if ss and ss.lower() not in seen:
                    uniq.append(ss)
                    seen.add(ss.lower())
            return uniq[:n_seeds]
    except Exception:
        pass
    return []

def _extract_multi_updates(agent_output: Dict[str,Any]) -> List[Tuple[str, Dict[str,Any]]]:
    """
    从一次性批处理返回中提取 [(hypothesis_id, graph_updates), ...]
    兼容以下结构：
      - {"graph_updates_per_hypothesis":[{"hypothesis_id":"H1","graph_updates":{...}}, ...]}
      - {"results":[{"hypothesis_id":"H1","graph_updates":{...}}, ...]}
      - {"graph_updates":{"H1":{...},"H2":{...}}}
      - Fallback: {"graph_updates":{...}}（无法归属，hypothesis_id 置为 ""）
    """
    out: List[Tuple[str, Dict[str,Any]]] = []
    if not isinstance(agent_output, dict):
        return out
    if isinstance(agent_output.get("graph_updates_per_hypothesis"), list):
        for item in agent_output["graph_updates_per_hypothesis"]:
            hid = (item or {}).get("hypothesis_id") or ""
            gu = (item or {}).get("graph_updates") or {}
            if isinstance(gu, dict):
                out.append((hid, gu))
        return out
    if isinstance(agent_output.get("results"), list):
        for item in agent_output["results"]:
            hid = (item or {}).get("hypothesis_id") or ""
            gu = (item or {}).get("graph_updates") or {}
            if isinstance(gu, dict):
                out.append((hid, gu))
        return out
    gus = agent_output.get("graph_updates")
    if isinstance(gus, dict):
        looks_like_updates = any(k in gus for k in ("add_nodes","add_edges","merge","set_weights"))
        if looks_like_updates:
            out.append(("", gus))
        else:
            for hid, gu in gus.items():
                if isinstance(gu, dict):
                    out.append((str(hid), gu))
    return out

def _apply_updates_list(graph: Dict[str,Any], updates_list: List[Tuple[str, Dict[str,Any]]], round_idx: int) -> Dict[str,Any]:
    """
    将一批 (hid, graph_updates) 依次应用到图里；强制写入 round_index。
    """
    g = graph
    for hid, gu in updates_list:
        if not isinstance(gu, dict):
            continue
        gu = dict(gu)
        gu["round_index"] = round_idx
        g = apply_graph_updates(g, gu)
    return g

# ===================== Orchestrator（含再生种子逻辑） =====================
class Orchestrator:
    def __init__(self, run_dir: str, query: Dict[str,Any], hypotheses: List[Dict[str,Any]],
                 init_thresholds: Dict[str,Any] = None, tag: str = "",
                 bio_prompt: str = "", seed_target_type: str = "disease",
                 n_seeds_default: int = 50, ablate: Dict[str,Any] = None):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_id = f"run_{ts}_{uuid.uuid4().hex[:6]}"
        self.run_id = f"{base_id}{('_'+tag) if tag else ''}"
        self.run_dir = os.path.join(run_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.query = query
        self.hypotheses = hypotheses
        self.thresholds = init_thresholds or {"stop_delta":0.03, "saturation_ratio":0.65}
        self.plan = {"rounds": 2}
        self.graph = empty_graph(round_index=1)
        self.round = 0
        self.last_scores = []

        # Explorer 相关上下文
        self.bio_prompt = bio_prompt
        self.seed_target_type = seed_target_type  # "disease" or "drug"
        self.n_seeds_default = n_seeds_default
        self.seed_history: List[List[str]] = []  # 每次（初始/再生）的种子列表

        # Ablation
        self.ablate = ablate or make_ablation_config()
        self.config_tag = ""

        # 将权重透传到 thresholds，供 PI 参考（也用于落盘记录）
        self.thresholds = dict(self.thresholds)
        self.thresholds["weights"] = dict(self.ablate.get("weights", {}))

    def _save(self, name: str, obj: Any):
        path = os.path.join(self.run_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # ---- Orchestrator 内新增：确保最小必需节点存在 ----
    def _ensure_core_nodes(self):
        """确保每个假设 Hk 有 Hypothesis 节点，每个候选有实体节点（Drug/Disease）。"""
        need_nodes = []
        have_ids = {n.get("id") for n in self.graph.get("nodes", [])}
        for h in self.hypotheses:
            hid = h["id"]
            if hid not in have_ids:
                need_nodes.append({
                    "id": hid,
                    "type": "Hypothesis",
                    "label": h["candidate"]["name"],
                    "attrs": {"agent": "PI"}
                })
            cid = h["candidate"]["id"]
            if cid not in have_ids:
                ntype = "Drug" if self.seed_target_type == "drug" else "Disease"
                need_nodes.append({
                    "id": cid,
                    "type": ntype,
                    "label": h["candidate"]["name"],
                    "attrs": {"agent": "Proponent"}
                })
        if need_nodes:
            self.graph = apply_graph_updates(self.graph, {"add_nodes": need_nodes, "add_edges": []})

    # Round-0
    def round0_init(self):
        self.round = 0
        heur = [
            {"when":"general biomedical reasoning","then":"区分短期表型与长期结局；明确潜在风险链"},
        ] if self.ablate.get("enable_heuristic_transfer", True) else []

        payload = {
            "mode":"init",
            "query": self.query,
            "hypotheses": self.hypotheses,
            "tegraph_snapshot": self.graph,
            "budgets": {"round_limit": 3, "token_limit": 200000},
            "heuristic_priors": heur,
            "thresholds": self.thresholds,
            "ablation_config": self.ablate
        }
        res = run_pi("init", payload)
        self.plan = res.get("plan", self.plan)
        self._save("round0_pi_init.json", {"input":payload, "output":res})

    # Round-1（构图+评分）
    def round1_build_and_score(self):
        self.round = 1
        self.graph["round_index"] = self.round

        # ---- Proponent: build_chain 批处理 ----
        payload_p = {
            "mode": "build_chain",
            "query": self.query,
            "hypotheses": self.hypotheses,
            "tegraph_snapshot": {"nodes": [], "edges": [], "round_index": self.round},
            "constraints": {"require_disjoint_paths": 2, "max_new_nodes": 50, "max_new_edges": 64, "batched": True},
            "ablation_config": self.ablate
        }
        rp = run_proponent("build_chain", payload_p)
        ups_p = _extract_multi_updates(rp)
        self.graph = _apply_updates_list(self.graph, ups_p, self.round)
        self._save("r1_proponent_batch.json", {"input": payload_p, "output": rp})

        # ---- Skeptic: build_counterchain 批处理（按开关） ----
        if self.ablate.get("enable_debate", True) and self.ablate.get("enable_skeptic", True):
            payload_s = {
                "mode": "build_counterchain",
                "query": self.query,
                "hypotheses": self.hypotheses,
                "tegraph_snapshot": {"nodes": [], "edges": [], "round_index": self.round},
                "constraints": {"max_new_nodes": 50, "max_new_edges": 64, "batched": True},
                "ablation_config": self.ablate
            }
            rs = run_skeptic("build_counterchain", payload_s)
            ups_s = _extract_multi_updates(rs)
            self.graph = _apply_updates_list(self.graph, ups_s, self.round)
            self._save("r1_skeptic_batch.json", {"input": payload_s, "output": rs})

        # —— 最小必需节点（Hypothesis/Drug|Disease）补齐 —— 
        self._ensure_core_nodes()

        # 评分
        score_payload = {
            "mode":"score",
            "query": self.query,
            "hypotheses": [{"id":h["id"], "candidate":h["candidate"]} for h in self.hypotheses],
            "tegraph_snapshot": self.graph,
            "history":{"round":1,"last_scores":[]},
            "thresholds": self.thresholds,
            "ablation_config": self.ablate
        }
        score = run_pi("score", score_payload)
        self._save("round1_pi_score.json", {"input":score_payload, "output":score})
        self.last_scores = score.get("scoring_summary", [])

        # 修订
        revise_payload = {
            "mode":"revise",
            "query": self.query,
            "tegraph_snapshot": self.graph,
            "history":{"round":1,"last_scores": self.last_scores},
            "thresholds": self.thresholds,
            "ablation_config": self.ablate
        }
        revise = run_pi("revise", revise_payload)
        self._save("round1_pi_revise.json", {"input":revise_payload, "output":revise})
        return score, revise

    # Round-1（仅构图，不评分；用于禁用 PI 中断）
    def round1_build_only(self):
        self.round = 1
        self.graph["round_index"] = self.round

        # Proponent 批构图
        payload_p = {
            "mode": "build_chain",
            "query": self.query,
            "hypotheses": self.hypotheses,
            "tegraph_snapshot": {"nodes": [], "edges": [], "round_index": self.round},
            "constraints": {"require_disjoint_paths": 2, "max_new_nodes": 50, "max_new_edges": 64, "batched": True},
            "ablation_config": self.ablate
        }
        rp = run_proponent("build_chain", payload_p)
        ups_p = _extract_multi_updates(rp)
        self.graph = _apply_updates_list(self.graph, ups_p, self.round)
        self._save("r1_proponent_batch.json", {"input": payload_p, "output": rp})

        # Skeptic 批构图（按开关）
        if self.ablate.get("enable_debate", True) and self.ablate.get("enable_skeptic", True):
            payload_s = {
                "mode": "build_counterchain",
                "query": self.query,
                "hypotheses": self.hypotheses,
                "tegraph_snapshot": {"nodes": [], "edges": [], "round_index": self.round},
                "constraints": {"max_new_nodes": 50, "max_new_edges": 64, "batched": True},
                "ablation_config": self.ablate
            }
            rs = run_skeptic("build_counterchain", payload_s)
            ups_s = _extract_multi_updates(rs)
            self.graph = _apply_updates_list(self.graph, ups_s, self.round)
            self._save("r1_skeptic_batch.json", {"input": payload_s, "output": rs})

        self._ensure_core_nodes()

    # —— 从 PI(revise) 中提取是否需要再生 seeds —— 
    @staticmethod
    def extract_seed_request_from_revise(revise: Dict[str,Any]) -> Optional[Dict[str,Any]]:
        def _truthy(x):
            if isinstance(x, bool): return x
            if isinstance(x, (int, float)): return x != 0
            if isinstance(x, str): return x.strip().lower() in ("true", "1", "yes", "y")
            return False
        def _to_int(x, default=50):
            try: return int(x)
            except: return default

        req = revise.get("seed_request") or revise.get("seed_regeneration")
        if isinstance(req, dict) and _truthy(req.get("should_regenerate")):
            return {
                "reason": str(req.get("reason") or req.get("why") or "").strip(),
                "n": _to_int(req.get("n") or req.get("count"), 50)
            }

        for r in revise.get("revisions", []) or []:
            sr = r.get("seed_request") or r.get("seed_regeneration")
            if isinstance(sr, dict) and _truthy(sr.get("should_regenerate")):
                return {
                    "reason": str(sr.get("reason") or sr.get("why") or "").strip(),
                    "n": _to_int(sr.get("n") or sr.get("count"), 50)
                }

        notes = revise.get("notes") or ""
        m = re.search(r"SEED_REGENERATE\s*:\s*(.+)", notes)
        if m:
            return {"reason": m.group(1).strip(), "n": 50}
        return None

    def add_new_hypotheses(self, seeds: List[str]) -> List[Dict[str,Any]]:
        """把新 seeds 追加成新的 Hk；自动编号去重"""
        if not seeds: return []
        existing_names = {h["candidate"]["name"].strip().lower() for h in self.hypotheses}
        new_seeds = [s for s in seeds if s.strip().lower() not in existing_names]
        start_idx = len(self.hypotheses) + 1
        new_hyps = []
        prefix = "drug" if self.seed_target_type=="drug" else "dis"
        for i, s in enumerate(new_seeds, start=start_idx):
            new_hyps.append({
                "id": f"H{i}",
                "candidate": {"id": f"{prefix}_{re.sub(r'[^a-z0-9]+','_', s.lower()).strip('_')}", "name": s},
                "mechanism_outline": "",
                "priority": max(0.2, 1.0 - (i-start_idx)*0.05)
            })
        self.hypotheses.extend(new_hyps)
        return new_hyps

    def maybe_regenerate_seeds_and_build(self, revise: Dict[str,Any]) -> bool:
        if not self.ablate.get("enable_seed_regen", True):
            return False

        req = self.extract_seed_request_from_revise(revise)
        if not req:
            return False

        # 记录事件（初始快照）
        event = {
            "round": self.round,
            "request": req,
            "prev_hypotheses": [h["candidate"]["name"] for h in self.hypotheses],
            "status": "start"
        }
        self._save(f"seed_regen_round{self.round}_event.json", event)

        # 1) 增强 Query（仅供 Explorer 用于生成新 seeds）
        aug_query = dict(self.query)
        aug_query["regen_prompt"] = req.get("reason") or "PI requested regeneration for improved coverage/diversity."

        # 2) Explorer 生成新 seeds
        seeds = run_explorer(self.bio_prompt, aug_query, self.seed_target_type,
                            n_seeds=self.n_seeds_default)

        event.update({"generated_seeds": seeds})
        if not seeds:
            event["status"] = "no_seeds"
            self._save(f"seed_regen_round{self.round}_event.json", event)
            return False

        # 3) 记录种子批次历史；覆盖候选（删除旧种子）
        self.seed_history.append(seeds)
        new_hyps = seeds_to_hypotheses(seeds, self.seed_target_type, start_idx=1)
        self.hypotheses = new_hyps
        self.last_scores = []  # 避免与旧候选的评分混淆

        # 4) 清空 TE 图（保留当前轮次索引）
        self.graph = empty_graph(round_index=self.round)

        # 5) 立刻为新候选跑构图（Proponent / Skeptic）
        payload_p = {
            "mode": "build_chain",
            "query": self.query,
            "hypotheses": new_hyps,
            "tegraph_snapshot": {"nodes": [], "edges": [], "round_index": self.round},
            "constraints": {"require_disjoint_paths": 2, "max_new_nodes": 50, "max_new_edges": 64, "batched": True},
            "ablation_config": self.ablate
        }
        rp = run_proponent("build_chain", payload_p)
        ups_p = _extract_multi_updates(rp)
        self.graph = _apply_updates_list(self.graph, ups_p, self.round)
        self._save(f"r{self.round}_proponent_regen_batch.json", {"input":payload_p, "output":rp})

        if self.ablate.get("enable_debate", True) and self.ablate.get("enable_skeptic", True):
            payload_s = {
                "mode": "build_counterchain",
                "query": self.query,
                "hypotheses": new_hyps,
                "tegraph_snapshot": {"nodes": [], "edges": [], "round_index": self.round},
                "constraints": {"max_new_nodes": 50, "max_new_edges": 64, "batched": True},
                "ablation_config": self.ablate
            }
            rs = run_skeptic("build_counterchain", payload_s)
            ups_s = _extract_multi_updates(rs)
            self.graph = _apply_updates_list(self.graph, ups_s, self.round)
            self._save(f"r{self.round}_skeptic_regen_batch.json", {"input":payload_s, "output":rs})

        # 6) 更新快照 & 事件
        self._save("current_hypotheses.json", {"when": f"regen_round_{self.round}", "hypotheses": self.hypotheses})
        event["status"] = "regenerated"
        event["new_hypotheses"] = [h["candidate"]["name"] for h in self.hypotheses]
        self._save(f"seed_regen_round{self.round}_event.json", event)
        return True

    def execute_round_from_revise(self, revise: Dict[str,Any]):
        self.round += 1
        self.graph["round_index"] = self.round

        regenerated = self.maybe_regenerate_seeds_and_build(revise)

        if not regenerated:
            # 收集各假设的 actions 按 agent 分组
            pro_items: List[Dict[str,Any]] = []
            skp_items: List[Dict[str,Any]] = []

            for rev in (revise.get("revisions") or []):
                hid = rev.get("hypothesis_id")
                acts = rev.get("graph_actions") or []
                if not acts or not hid:
                    continue
                pro_actions = [a for a in acts if a.get("assignee")=="Proponent"]
                skp_actions = [a for a in acts if a.get("assignee")=="Skeptic"]
                if pro_actions:
                    pro_items.append({"hypothesis_id": hid, "graph_actions": pro_actions})
                if skp_actions:
                    skp_items.append({"hypothesis_id": hid, "graph_actions": skp_actions})

            # 批调用 Proponent.execute_actions
            if pro_items:
                payload_p = {
                    "mode": "execute_actions",
                    "query": self.query,
                    "items": pro_items,                  # ★ 批
                    "tegraph_snapshot": self.graph,
                    "constraints":{"max_new_nodes":50,"max_new_edges":64,"batched":True},
                    "ablation_config": self.ablate
                }
                rp = run_proponent("execute_actions", payload_p)
                ups_p = _extract_multi_updates(rp)
                self.graph = _apply_updates_list(self.graph, ups_p, self.round)
                self._save(f"r{self.round}_prop_exec_batch.json", {"input":payload_p, "output":rp})

            # 批调用 Skeptic.execute_actions（按开关）
            if skp_items and self.ablate.get("enable_debate", True) and self.ablate.get("enable_skeptic", True):
                payload_s = {
                    "mode": "execute_actions",
                    "query": self.query,
                    "items": skp_items,                  # ★ 批
                    "tegraph_snapshot": self.graph,
                    "constraints":{"max_new_nodes":50,"max_new_edges":64,"batched":True},
                    "ablation_config": self.ablate
                }
                rs = run_skeptic("execute_actions", payload_s)
                ups_s = _extract_multi_updates(rs)
                self.graph = _apply_updates_list(self.graph, ups_s, self.round)
                self._save(f"r{self.round}_skep_exec_batch.json", {"input":payload_s, "output":rs})

            # 补齐最小必需节点（避免丢失）
            self._ensure_core_nodes()

        # 评分
        score_payload = {
            "mode":"score",
            "query": self.query,
            "hypotheses": [{"id":h["id"], "candidate":h["candidate"]} for h in self.hypotheses],
            "tegraph_snapshot": self.graph,
            "history":{"round": self.round, "last_scores": self.last_scores},
            "thresholds": self.thresholds,
            "ablation_config": self.ablate
        }
        score = run_pi("score", score_payload)
        self._save(f"round{self.round}_pi_score.json", {"input":score_payload, "output":score})
        self.last_scores = score.get("scoring_summary", [])

        # 修订
        revise_payload = {
            "mode":"revise",
            "query": self.query,
            "tegraph_snapshot": self.graph,
            "history":{"round": self.round, "last_scores": self.last_scores},
            "thresholds": self.thresholds,
            "ablation_config": self.ablate
        }
        revise2 = run_pi("revise", revise_payload)
        self._save(f"round{self.round}_pi_revise.json", {"input":revise_payload, "output":revise2})
        return score, revise2

    def should_stop(self, score: Dict[str,Any]) -> bool:
        sd = score.get("stop_decision", {})
        if isinstance(sd.get("should_stop"), bool) and sd["should_stop"]:
            return True
        # 首轮评分（round == 1）忽略 delta 判据，避免还没执行 revise 就提前结束
        if self.round <= 1:
            return False
        delta = score.get("delta_since_last_round", 0.0) or 0.0
        # if delta < float(self.thresholds.get("stop_delta", 0.03)):
        #     return True
        return False

    def finalize(self):
        # 若关闭中断或 list_only，可能没有 last_scores；先兜底打一轮评分
        if not self.last_scores:
            score_payload = {
                "mode": "score",
                "query": self.query,
                "hypotheses": [{"id":h["id"], "candidate":h["candidate"]} for h in self.hypotheses],
                "tegraph_snapshot": (self.graph if self.ablate.get("mode")!="list_only" else {"nodes": [], "edges": [], "round_index": max(1, self.round)}),
                "history":{"round": max(1, self.round), "last_scores": []},
                "thresholds": self.thresholds,
                "ablation_config": self.ablate
            }
            try:
                s = run_pi("score", score_payload)
                self.last_scores = s.get("scoring_summary", []) or []
                self._save("finalize_score_fallback.json", {"input": score_payload, "output": s})
            except Exception as e:
                self._save("finalize_score_error.json", {"error": str(e), "input": score_payload})

        # 排名
        ranking = []
        try:
            ranking = sorted(
                [(s["hypothesis_id"], float(s.get("score", 0.0))) for s in (self.last_scores or [])],
                key=lambda x: x[1], reverse=True
            )
        except Exception:
            ranking = []

        self.graph["round_index"] = max(self.graph.get("round_index", 1), self.round)

        payload = {
            "mode":"report_and_evolve",
            "query": self.query,
            "hypotheses": self.hypotheses,
            "scores": self.last_scores,
            "ranking": [hid for hid, _ in ranking],
            "tegraph_snapshot": (self.graph if self.ablate.get("mode")!="list_only" else {"nodes": [], "edges": [], "round_index": max(1, self.round)}),
            "history":{"round": self.round},
            "thresholds": self.thresholds,
            "ablation_config": self.ablate
        }
        rep = run_pi("report_and_evolve", payload)

        # 如果禁用 textual feedback，就把 rep 中的 prompt_patches 清空以避免“自进化”副作用
        if not self.ablate.get("enable_textual_feedback", True) and isinstance(rep, dict):
            if isinstance(rep.get("prompt_patches"), list):
                rep = dict(rep)
                rep["prompt_patches"] = []

        self._save("final_report.json", {"input":payload, "output":rep})
        return rep

    def run_list_only(self):
        # 不调用 Proponent / Skeptic；仅用 PI.score 排名 + report
        self.round = 1
        score_payload = {
            "mode":"score",
            "query": self.query,
            "hypotheses": [{"id":h["id"], "candidate":h["candidate"]} for h in self.hypotheses],
            "tegraph_snapshot": {"nodes": [], "edges": [], "round_index": 1},
            "history":{"round": 1, "last_scores":[]},
            "thresholds": self.thresholds,
            "ablation_config": self.ablate
        }
        s = run_pi("score", score_payload)
        self._save("list_only_score.json", {"input": score_payload, "output": s})
        self.last_scores = s.get("scoring_summary", []) or []
        return self.finalize()

    def run(self):
        try:
            self.round0_init()

            # list-only 基线：不构图、不辩论、一次性评分
            if self.ablate.get("mode") == "list_only":
                return self.run_list_only()

            plan_rounds = int(self.plan.get("rounds", 2))

            if not self.ablate.get("enable_pi_interrupts", True):
                # 仅构图，不做中间评分/修订；最后统一评分+报告
                self.round1_build_only()
                return self.finalize()

            # 正常多轮：R1 构图+评分+修订
            score1, revise1 = self.round1_build_and_score()
            if self.should_stop(score1):
                self.round = 1
                return self.finalize()

            current_revise = revise1
            while self.round < plan_rounds + 1:
                scoreK, reviseK = self.execute_round_from_revise(current_revise)
                if self.should_stop(scoreK):
                    return self.finalize()
                if not any(len(r.get("graph_actions", []))>0 for r in reviseK.get("revisions", [])) \
                  and not self.extract_seed_request_from_revise(reviseK):
                    return self.finalize()
                current_revise = reviseK
            return self.finalize()
        except Exception as e:
            # 落本 run 的错误快照
            self._save("run_error.json", {
                "round": self.round,
                "query": self.query,
                "hypotheses": self.hypotheses,
                "graph_round_index": self.graph.get("round_index"),
                "error": str(e),
                "ablation_config": self.ablate
            })
            # 抛给上层（上层会标记 runtime_error 并跳过样本）
            raise

# ===================== 数据抽取：CSV & BioHopR.json =====================
def normalize_name(s: str) -> str:
    return (s or "").strip().lower()

def load_csv_relations(csv_path: str) -> Dict[Tuple[str,str], str]:
    mapping: Dict[Tuple[str,str], str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_type = (row.get("x_type") or "").strip().lower()
            y_type = (row.get("y_type") or "").strip().lower()
            x = normalize_name(row.get("x_id") or row.get("x_name") or "")
            y = normalize_name(row.get("y_id") or row.get("y_name") or "")
            rel = (row.get("relation") or "").strip()
            if x_type=="drug" and y_type=="disease" and x and y and rel:
                mapping[(x,y)] = rel
    return mapping

def extract_query(sample: Dict[str,Any], pair_map: Dict[Tuple[str,str], str]) -> Optional[Dict[str,Any]]:
    target_type = (sample.get("target_type") or "").strip().lower()
    hop1_t = (sample.get("hop1_type") or "").strip().lower()
    hop2_t = (sample.get("hop2_type") or "").strip().lower()
    hop1 = sample.get("hop1") or ""
    hop2 = sample.get("hop2") or ""
    answers = sample.get("answer") or []
    if not answers: return None
    first_ans = answers[0]

    if target_type == "disease":
        drug = hop1 if hop1_t=="drug" else (hop2 if hop2_t=="drug" else None)
        disease = first_ans
        if not drug or not disease: return None
        rel = pair_map.get((normalize_name(drug), normalize_name(disease)))
        if not rel: return None
        return {"entity": drug, "relation": rel, "target_type": "disease"}

    if target_type == "drug":
        disease = hop1 if hop1_t=="disease" else (hop2 if hop2_t=="disease" else None)
        drug = first_ans
        if not disease or not drug: return None
        rel = pair_map.get((normalize_name(drug), normalize_name(disease)))
        if not rel: return None
        return {"entity": disease, "relation": rel, "target_type": "drug"}

    return None

def seeds_to_hypotheses(seeds: List[str], target_type: str, start_idx: int = 1) -> List[Dict[str,Any]]:
    hyps = []
    prefix = "drug" if target_type=="drug" else "dis"
    for i, s in enumerate(seeds, start=start_idx):
        hyps.append({
            "id": f"H{i}",
            "candidate": {"id": f"{prefix}_{re.sub(r'[^a-z0-9]+','_', s.lower()).strip('_')}", "name": s},
            "mechanism_outline": "",
            "priority": max(0.2, 1.0 - (i-start_idx)*0.05)
        })
    return hyps

# ---------- HOP 任务：Query 构造 ----------
def make_query_for_hop(sample: Dict[str,Any], hop_key: str, 
                       pair_map: Dict[Tuple[str,str], str],
                       require_drug_disease: bool = True) -> Optional[Dict[str,Any]]:
    """
    只做“药物↔疾病”的 hop；relation 严格从 CSV 查，查不到则置为 "association" 也要跑。
    return:
      - dict: {"entity": <hop实体>, "relation": <relation或'association'>, "target_type": <'disease'|'drug'>}
      - None: 非药物↔疾病配对（例如 phenotype→disease 等），直接跳过
    """
    target_type = (sample.get("target_type") or "").strip().lower()
    hop = sample.get(hop_key, "")
    hop_t = (sample.get(f"{hop_key}_type") or "").strip().lower()
    answers = sample.get("answer") or []
    first_ans = answers[0] if answers else ""

    # 仅处理“药物↔疾病”结构；否则跳过
    if require_drug_disease:
        if not first_ans:
            return None

        # 目标是疾病：hop 必须是 drug；relation 从 (drug, disease) 查，查不到用 association
        if target_type == "disease" and hop_t == "drug":
            drug = hop
            disease = first_ans
            rel = pair_map.get((normalize_name(drug), normalize_name(disease)))
            relation = rel if rel else "association"
            return {"entity": hop, "relation": relation, "target_type": "disease"}

        # 目标是药物：hop 必须是 disease；relation 从 (drug, disease) 查，查不到用 association
        if target_type == "drug" and hop_t == "disease":
            disease = hop
            drug = first_ans
            rel = pair_map.get((normalize_name(drug), normalize_name(disease)))
            relation = rel if rel else "association"
            return {"entity": hop, "relation": relation, "target_type": "drug"}

        # 其他组合（如 phenotype→disease）直接跳过
        return None

    # （不严格时的兜底——一般不再用）
    relation = "association"
    if target_type == "disease" and hop_t == "drug" and first_ans:
        rel = pair_map.get((normalize_name(hop), normalize_name(first_ans)))
        if rel: relation = rel
    elif target_type == "drug" and hop_t == "disease" and first_ans:
        rel = pair_map.get((normalize_name(first_ans), normalize_name(hop)))
        if rel: relation = rel

    return {"entity": hop, "relation": relation, "target_type": target_type}

# ---------- Explorer 种子：按 hop 选择 multi 问题，必要时兜底 ----------
def seeds_from_explorer_or_fallback(sample: Dict[str,Any], hop_key: str, query: Dict[str,Any], n: int) -> List[str]:
    """
    使用 hop1_question_multi / hop2_question_multi 作为 bio_prompt 喂给 Explorer。
    若 Explorer 无输出：
      - target_type==disease：用 answers 前 n 个兜底；
      - target_type==drug：返回空（照样记录，后续人工排查）。
    """
    bio_prompt = sample.get(f"{hop_key}_question_multi") or sample.get(f"{hop_key}_question") or sample.get("prompt","")
    seeds = run_explorer(bio_prompt, {"entity": query["entity"], "relation": query["relation"], "regen_prompt": ""}, query["target_type"], n_seeds=n)

    if seeds:
        return seeds

    if (query.get("target_type") or "").lower() == "disease":
        answers = [a for a in (sample.get("answer") or []) if isinstance(a, str)]
        uniq = []
        seen = set()
        for a in answers:
            s = a.strip()
            if s and s.lower() not in seen:
                uniq.append(s); seen.add(s.lower())
            if len(uniq) >= n: break
        return uniq
    return []

# ========= 并发安全错误写入 =========
def append_error_jsonl(path: str, rec: Dict[str,Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = _get_file_lock(path)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def log_sample_error(base_out_dir: str, hop_key: str, sample_index: int,
                     stage: str, err: Exception, extra: Dict[str,Any]):
    """
    写两份：
      - runs/<hop_key>/_errors.jsonl         （分 hop）
      - runs/_failed_samples.jsonl            （全局）
    """
    err_str = str(err)
    rec = {
        "sample_index": sample_index,
        "hop": hop_key,
        "stage": stage,               # explorer / proponent_build / skeptic_build / prop_exec / skep_exec / score / revise / report
        "error": err_str[:1000],
        "extra": extra
    }
    # hop 层
    perhop = os.path.join(base_out_dir, hop_key, "_errors.jsonl")
    append_error_jsonl(perhop, rec)
    # 全局
    global_err = os.path.join(base_out_dir, "_failed_samples.jsonl")
    append_error_jsonl(global_err, rec)

# 全局锁表
_FILE_LOCKS: Dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()

def _get_file_lock(path: str) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        if path not in _FILE_LOCKS:
            _FILE_LOCKS[path] = threading.Lock()
        return _FILE_LOCKS[path]

def append_record_jsonl(path: str, rec: Dict[str,Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = _get_file_lock(path)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- 单条样本执行 & 记录 ----------
def run_one_sample_for_hop(sample: Dict[str,Any], csv_map: Dict[Tuple[str,str], str],
                           out_dir: str, hop_key: str, n_seeds: int = 50, sample_index: Optional[int] = None,
                           ablate: Dict[str,Any] = None, config_tag: str = "") -> Optional[Dict[str,Any]]:
    """
    运行 hop1 或 hop2 任务。将完整 artifacts 存到 out_dir/hop{1|2}/run_* 下；
    并返回一条用于汇总的记录（写入 _records.jsonl）。
    """
    ablate = ablate or make_ablation_config()

    # 记录 hop 的原始取值 / 类型 / 原始关系
    hop_value = sample.get(hop_key, "")
    hop_type  = (sample.get(f"{hop_key}_type") or "").strip().lower()
    relation_hop = sample.get(f"relation_{hop_key}")  # 例如 "effect/phenotype:disease", "drug:effect/phenotype:disease"
    
    # 构造 Query
    query = make_query_for_hop(sample, hop_key, csv_map)
    if query is None:
        # 记一条跳过记录，保证“断点续跑”不会重复尝试该 hop
        return {
            "sample_index": sample_index,
            "hop": hop_key,
            "hop_value": hop_value,
            "hop_type": hop_type,
            "relation_hop": relation_hop,
            "query": None,
            "target_type": (sample.get("target_type") or "").strip().lower(),
            "answers_all": sample.get("answer") or [],
            "seeds": [],
            "ranking": [],
            "top_recommendations": [],
            "run_dir": None,
            "note": "skip_non_drug_disease",  # 非“药物↔疾病”或 CSV 中查不到 relation
            "config": {"ablation": ablate, "config_tag": config_tag}
        }
    target_type = query["target_type"]
    subdir = os.path.join(out_dir, hop_key)  # hop1 / hop2
    os.makedirs(subdir, exist_ok=True)

    # 种子
    # 生成种子（捕获 RuntimeError）
    try:
        seeds = seeds_from_explorer_or_fallback(sample, hop_key, query, n=n_seeds)
    except Exception as e:
        # 记录并占位
        log_sample_error(out_dir, hop_key, sample_index if sample_index is not None else -1,
                        stage="explorer", err=e,
                        extra={"query": query, "hop_value": hop_value, "hop_type": hop_type, "relation_hop": relation_hop})
        return {
            "sample_index": sample_index,
            "hop": hop_key,
            "hop_value": hop_value,
            "hop_type": hop_type,
            "relation_hop": relation_hop,
            "query": query,
            "target_type": target_type,
            "answers_all": sample.get("answer") or [],
            "seeds": [],
            "ranking": [],
            "top_recommendations": [],
            "run_dir": None,
            "note": "runtime_error:explorer",
            "config": {"ablation": ablate, "config_tag": config_tag}
        }

    # 若连兜底都无（罕见），记录空并返回
    if not seeds:
        record = {
            "sample_index": sample_index, "hop": hop_key, "hop_value": hop_value,"hop_type": hop_type,"relation_hop": relation_hop,"query": query, "seeds": [],
            "target_type": target_type,
            "answers_all": sample.get("answer") or [],
            "ranking": [], "top_recommendations": [],
            "run_dir": None, "note": "no_seeds",
            "config": {"ablation": ablate, "config_tag": config_tag}
        }
        return record

    # hypotheses
    try:
        # hypotheses + Orchestrator + run
        hyps = seeds_to_hypotheses(seeds, target_type, start_idx=1)
        tag = f"{hop_key}_{re.sub(r'[^a-z0-9]+','_', str(query['entity']).lower())[:16]}"
        orch = Orchestrator(run_dir=subdir, query={"entity": query["entity"], "relation": query["relation"]},
                            hypotheses=hyps,
                            init_thresholds={"stop_delta":0.03, "saturation_ratio":0.65},
                            tag=tag,
                            bio_prompt=sample.get(f"{hop_key}_question_multi") or sample.get("prompt",""),
                            seed_target_type=target_type,
                            n_seeds_default=n_seeds,
                            ablate=ablate)
        orch.seed_history.append(seeds)
        orch.config_tag = config_tag or ""
        final_rep = orch.run()

    except Exception as e:
        # 任意阶段失败（包括内部的 PI/Proponent/Skeptic 调用）
        log_sample_error(out_dir, hop_key, sample_index if sample_index is not None else -1,
                        stage="orchestrator", err=e,
                        extra={"query": query, "hop_value": hop_value, "hop_type": hop_type,
                                "relation_hop": relation_hop, "seeds": seeds})
        return {
            "sample_index": sample_index,
            "hop": hop_key,
            "hop_value": hop_value,
            "hop_type": hop_type,
            "relation_hop": relation_hop,
            "query": query,
            "target_type": target_type,
            "answers_all": sample.get("answer") or [],
            "seeds": seeds,
            "ranking": [],
            "top_recommendations": [],
            "run_dir": None,
            "note": "runtime_error:orchestrator",
            "config": {"ablation": ablate, "config_tag": config_tag}
        }

    # 提取结果摘要（用于后续统计）
    final = final_rep or {}
    recs = final.get("final_recommendations") or []
    top_rec_names = [r.get("candidate",{}).get("name") for r in recs if isinstance(r, dict)]
    ranking_names = []
    try:
        # 优先 final.ranking，如果没有，用 last_scores 排名
        rnk = final.get("ranking") or []
        if not rnk and getattr(orch, "last_scores", None):
            tmp = sorted([(s["hypothesis_id"], float(s.get("score",0.0))) for s in orch.last_scores],
                         key=lambda x: x[1], reverse=True)
            rnk = [hid for hid,_ in tmp]
        for hid in rnk:
            cand = next((h["candidate"]["name"] for h in orch.hypotheses if h["id"]==hid), None)
            if cand: ranking_names.append(cand)
    except Exception:
        pass

    # 落 meta
    meta = {
        "sample_index": sample_index,
        "hop": hop_key,
        "hop_value": hop_value,          
        "hop_type": hop_type,            
        "relation_hop": relation_hop,    
        "query": query,
        "target_type": target_type,
        "seeds_initial": seeds,
        "seed_history": orch.seed_history,
        "biohopr_prompt_used": sample.get(f"{hop_key}_question_multi") or sample.get("prompt",""),
        "answers_all": sample.get("answer") or [],
        "final_recommendations": top_rec_names,
        "ranking": ranking_names,
        "config": {
            "ablation": orch.ablate,
            "config_tag": orch.config_tag
        }
    }
    with open(os.path.join(orch.run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 返回一条记录（写 JSONL）
    record = {
        "sample_index": sample_index,
        "hop": hop_key,
        "hop_value": hop_value,         
        "hop_type": hop_type,           
        "relation_hop": relation_hop,   
        "query": {"entity": query["entity"], "relation": query["relation"]},
        "target_type": target_type,
        "answers_all": sample.get("answer") or [],
        "seeds": seeds,
        "ranking": ranking_names,
        "top_recommendations": top_rec_names,
        "run_dir": orch.run_dir,
        "config": {
            "ablation": orch.ablate,
            "config_tag": orch.config_tag
        }
    }
    return record

# ---------- 全量运行入口（hop1 & hop2） ----------
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_biohopr_all(bio_path: str, csv_path: str, out_dir: str,
                    n_seeds: int = 50,
                    start_index: int = 0,
                    num_samples: Optional[int] = None,
                    workers: int = 4, resume: bool = True, rerun_failed: bool = False,
                    ablate: Dict[str,Any] = None, config_tag: str = ""):

    os.makedirs(out_dir, exist_ok=True)
    pair_map = load_csv_relations(csv_path)
    with open(bio_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 准备记录文件路径
    hop1_dir = os.path.join(out_dir, "hop1")
    hop2_dir = os.path.join(out_dir, "hop2")
    os.makedirs(hop1_dir, exist_ok=True)
    os.makedirs(hop2_dir, exist_ok=True)
    rec_hop1 = os.path.join(hop1_dir, "_records.jsonl")
    rec_hop2 = os.path.join(hop2_dir, "_records.jsonl")

    # resume 逻辑：默认不清空；仅 --no_resume 时清空
    if not resume:
        for p in [rec_hop1, rec_hop2]:
            if os.path.exists(p):
                os.remove(p)

    # 读取已完成的样本索引（按 hop 区分）
    done_hop1 = load_processed_indices_from_records(rec_hop1, rerun_failed=rerun_failed)
    done_hop2 = load_processed_indices_from_records(rec_hop2, rerun_failed=rerun_failed)

    # 内部 worker：处理一个样本（含 hop1 + hop2），支持按 hop 跳过
    def _work(idx_sample):
        idx, sample = idx_sample

        # hop1
        if idx in done_hop1:
            print(f"[SKIP] sample #{idx} hop1 already recorded.")
        else:
            try:
                r1 = run_one_sample_for_hop(sample, pair_map, out_dir,
                                            hop_key="hop1", n_seeds=n_seeds, sample_index=idx,
                                            ablate=ablate, config_tag=config_tag)
                if r1:
                    r1["sample_index"] = idx
                    append_record_jsonl(rec_hop1, r1)
            except Exception as e:
                log_sample_error(out_dir, "hop1", idx, "worker", e, {"note":"uncaught at worker level"})
                append_record_jsonl(rec_hop1, {
                    "sample_index": idx, "hop": "hop1", "note": "runtime_error:worker", "error": str(e),
                    "config": {"ablation": ablate, "config_tag": config_tag}
                })

        # hop2
        if idx in done_hop2:
            print(f"[SKIP] sample #{idx} hop2 already recorded.")
        else:
            try:
                r2 = run_one_sample_for_hop(sample, pair_map, out_dir,
                                            hop_key="hop2", n_seeds=n_seeds, sample_index=idx,
                                            ablate=ablate, config_tag=config_tag)
                if r2:
                    r2["sample_index"] = idx
                    append_record_jsonl(rec_hop2, r2)
            except Exception as e:
                log_sample_error(out_dir, "hop2", idx, "worker", e, {"note":"uncaught at worker level"})
                append_record_jsonl(rec_hop2, {
                    "sample_index": idx, "hop": "hop2", "note": "runtime_error:worker", "error": str(e),
                    "config": {"ablation": ablate, "config_tag": config_tag}
                })

        return idx

    # 任务列表（按起始编号 + 条数切片；保持原始 sample_index）
    all_enum = list(enumerate(data))
    total = len(all_enum)
    start = max(0, min(int(start_index or 0), total))
    if num_samples is None or int(num_samples) < 0:
        end = total
    else:
        end = min(total, start + int(num_samples))
    data_enumerated = all_enum[start:end]

    print(f"[PLAN] total={total}, start_index={start}, end={end}, planned={len(data_enumerated)}")

    # 并发 or 串行
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work, item) for item in data_enumerated]
            for fut in as_completed(futs):
                try:
                    idx_done = fut.result()
                    print(f"[OK] sample #{idx_done} done (hop1/hop2 or skipped).")
                except Exception as e:
                    print(f"[ERR] worker error: {e}")
    else:
        for item in data_enumerated:
            _work(item)
            print(f"[OK] sample #{item[0]} done (hop1/hop2 or skipped).")

# ===================== 批处理入口 =====================
def run_batch_from_biohopr(bio_path: str, csv_path: str, out_dir: str,
                           n_seeds: int = 50, max_samples: int = 10, skip_no_query: bool = True,
                           ablate: Dict[str,Any] = None, config_tag: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    pair_map = load_csv_relations(csv_path)
    with open(bio_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    used = 0
    for idx, sample in enumerate(data):
        if used >= max_samples: break

        q = extract_query(sample, pair_map)
        if not q:
            if skip_no_query: continue
            else:
                print(f"[WARN] Sample #{idx} no query; skipping.")
                continue

        bio_prompt = sample.get("prompt","")
        # —— 初始种子由 Explorer 生成（BioHopR prompt + Query）——
        seeds = run_explorer(bio_prompt, {"entity": q["entity"], "relation": q["relation"]},
                             q["target_type"], n_seeds=n_seeds)
        if not seeds:
            print(f"[WARN] Sample #{idx} no seeds from Explorer; skipping.")
            continue

        hyps = seeds_to_hypotheses(seeds, q["target_type"], start_idx=1)
        query = {"entity": q["entity"], "relation": q["relation"]}

        tag = f"s{idx}_{q['target_type']}_{re.sub(r'[^a-z0-9]+','_', q['entity'].lower())[:16]}"
        orch = Orchestrator(run_dir=out_dir, query=query, hypotheses=hyps,
                            init_thresholds={"stop_delta":0.03, "saturation_ratio":0.65},
                            tag=tag, bio_prompt=bio_prompt,
                            seed_target_type=q["target_type"], n_seeds_default=n_seeds,
                            ablate=ablate or make_ablation_config())
        orch.seed_history.append(seeds)  # 记录初始种子
        orch.config_tag = config_tag or ""
        final_report = orch.run()

        # 记录 meta：包含全部正确答案 + 种子历史
        answers_all = sample.get("answer") or []
        meta = {
            "sample_index": idx,
            "query": query,
            "target_type": q["target_type"],
            "seeds_initial": seeds,
            "seed_history": orch.seed_history,   # 初始 + 每次再生的种子批
            "biohopr_prompt": bio_prompt,
            "answers_first": answers_all[0] if answers_all else None,
            "answers_all": answers_all,
            "config": {
                "ablation": orch.ablate,
                "config_tag": orch.config_tag
            }
        }
        with open(os.path.join(orch.run_dir, "meta.json"), "w", encoding="utf-8") as fmeta:
            json.dump(meta, fmeta, ensure_ascii=False, indent=2)

        print(f"[OK] Sample #{idx} finished. Dir = {orch.run_dir}")
        used += 1

# ===================== CLI / main =====================
def default_demo_hypotheses():
    return [
        {"id":"H1","candidate":{"id":"drug_mid","name":"Midodrine"},"mechanism_outline":"", "priority":0.9},
        {"id":"H2","candidate":{"id":"drug_drox","name":"Droxidopa"},"mechanism_outline":"", "priority":0.85},
        {"id":"H3","candidate":{"id":"drug_cortac","name":"Cortisone acetate"},"mechanism_outline":"", "priority":0.55},
        {"id":"H4","candidate":{"id":"drug_doxa","name":"Doxazosin"},"mechanism_outline":"", "priority":0.2}
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biohopr", type=str, default="BioHopR.json")
    parser.add_argument("--csv", type=str, default="test_with_names_filtered.csv")
    parser.add_argument("--out_dir", type=str, default="runs_all_parallel_new")
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 表示全部")
    parser.add_argument("--all_biohopr", action="store_true", default=True, help="对 BioHopR.json 中的所有样本各跑 hop1 与 hop2，并分别记录到 hop1/hop2 目录与 _records.jsonl")
    parser.add_argument("--demo", action="true", default=False)
    parser.add_argument("--workers", type=int, default=8, help="并发 worker 数（<=1 则串行）")
    parser.add_argument("--no_resume", action="store_true", help="禁用续跑（会清空现有 _records.jsonl 后重跑）")
    parser.add_argument("--rerun_failed", action="store_true", help="续跑模式下，仅重跑失败样本（note 以 runtime_error 开头）")

    parser.add_argument("--start_index", type=int, default=2000, help="从该样本编号开始（0-based）")
    parser.add_argument("--num_samples", type=int, default=2000, help="从 start_index 起跑多少条；-1 表示一直到末尾")

    # ---- Ablation switches ----
    parser.add_argument("--list_only", action="store_true", help="退化为 list-only 排序基线（不构图、不辩论）")
    parser.add_argument("--disable_debate", action="store_true")
    parser.add_argument("--disable_skeptic", action="store_true")
    parser.add_argument("--disable_pi_interrupts", action="store_true")
    parser.add_argument("--disable_textual_feedback", action="store_true")
    parser.add_argument("--disable_seed_regen", action="store_true")
    parser.add_argument("--disable_heuristics", action="store_true")
    parser.add_argument("--path_disjointness_w", type=float, default=1.0)
    parser.add_argument("--config_tag", type=str, default="", help="写入到每条记录的 config_tag，便于聚合比较")

    args = parser.parse_args()

    ablate = make_ablation_config(
        enable_debate = not args.disable_debate,
        enable_skeptic = not args.disable_skeptic,
        enable_pi_interrupts = not args.disable_pi_interrupts,
        enable_textual_feedback = not args.disable_textual_feedback,
        enable_heuristic_transfer = not args.disable_heuristics,
        enable_seed_regen = not args.disable_seed_regen,
        path_disjointness_weight = args.path_disjointness_w,
        mode = ("list_only" if args.list_only else "graph")
    )

    if args.demo:
        query = {"entity": "orthostatic hypotension", "relation": "indication"}
        hyps = default_demo_hypotheses()
        orch = Orchestrator(run_dir=args.out_dir, query=query, hypotheses=hyps,
                            init_thresholds={"stop_delta":0.03, "saturation_ratio":0.65},
                            bio_prompt="Demo prompt for OH", seed_target_type="drug", n_seeds_default=args.n_seeds,
                            ablate=ablate)
        orch.seed_history.append([h["candidate"]["name"] for h in hyps])
        final_report = orch.run()
        print(f"Demo run finished. Artifacts in: {orch.run_dir}")
        return

    if args.all_biohopr:
        # 兼容：若用户没填 --num_samples，但填了旧的 --max_samples，则用它当作 num_samples
        if args.num_samples is not None and args.num_samples >= 0:
            num = args.num_samples
        elif args.max_samples is not None and args.max_samples >= 0:
            num = args.max_samples
        else:
            num = None  # 一直到文件末尾

        run_biohopr_all(
            args.biohopr, args.csv, args.out_dir,
            n_seeds=args.n_seeds,
            start_index=max(0, args.start_index),
            num_samples=num,
            workers=max(1, args.workers),
            resume=(not args.no_resume),
            rerun_failed=args.rerun_failed,
            ablate=ablate,
            config_tag=args.config_tag
        )
        print(f"All BioHopR (hop1/hop2) finished. See {args.out_dir}/hop1/_records.jsonl and {args.out_dir}/hop2/_records.jsonl")
        return

    # 兼容旧的批量（抽样子集）
    run_batch_from_biohopr(args.biohopr, args.csv, args.out_dir,
                           n_seeds=args.n_seeds, max_samples=(None if args.max_samples<0 else args.max_samples),
                           ablate=ablate, config_tag=args.config_tag)


EXPLORER_SYSTEM_PROMPT = """
You are the Explorer Agent. Return ONLY a JSON object with this shape:
{"seeds": ["<name1>", "<name2>", "..."]}

Goal:
- Propose at least N distinct candidate names (no duplicates), where N is given in the user payload ("n").
- Candidates must match the target_type (either "disease" or "drug") and be plausible for the provided query.
- Query includes:
    { "entity": <string>, "relation": <string>, "target_type": <"disease"|"drug">, "regen_prompt": <string?> }
- You should use BOTH:
    (1) the raw BioHopR question ("biohopr_prompt") for semantic guidance, and
    (2) the structured "query" (entity/relation/target_type). 
- If "regen_prompt" is non-empty, treat it as a reason for diversification/refinement and bias your list accordingly.
- Do not include explanatory text, only the JSON with key "seeds".

Constraints:
- Unique strings, trimmed. No empty strings. Prefer well-known biomedical names.
- If unsure, return as many as you can (>=40).
"""


# ===================== 三个Agent的提示词 =====================
PI_SYSTEM_PROMPT = """
## PI Agent

你是“极端零样本药物再利用”流程中的**首席研究员（PI）**。
**T-EGraph 由 Proponent / Skeptic 给出**（来源于其领域常识与推理）。你基于**输入的 T-EGraph 快照**与启发式进行：

1. 制定计划与评分准则（`init`）；
2. 依据 T-EGraph 对候选假设 **多维评分与排序**（`score`）；
3. 按增益与缺口 **修订下一轮的推理与图构建指令**（发回 Proponent / Skeptic，`revise`）；
4. 产出阶段性 **结论摘要 + 归因分析 + 提示词补丁 + 启发式沉淀**（`report_and_evolve`）。
5. 你还负责**种子管理决策**（不直接产出种子）：当现有候选（`hypotheses`）表现出**覆盖不足/同质化/冲突难化解**等特征时，你可以在 `revise` 阶段发出**再生种子请求**，由 **Explorer Agent** 依据 `BioHopR.prompt + Query + 你的再生理由` 生成新种子。你**不得**直接输出任何新种子名称；只能在 JSON 中给出**再生请求对象**与**简短理由**。

### 严格规则

* **只输出 JSON**（UTF-8，无 Markdown/解释/多段）。
* **不泄露思维链**：以简短要点给出理由；不得生成长推导过程。
* **不编造来源**：不得虚构论文/PMID/统计数字。
* **不改图**：PI **不直接修改 T-EGraph**；仅下达“图更新指令”（由 Proponent / Skeptic 执行）。
* **不作医疗建议**：输出仅供研究决策。
* **不确定性标注**：当依据不足时，降低 `confidence`，并在 `uncertainties` 中列出假设/缺口。
* 必须对 input.hypotheses 里的每个 id 产出一条评分；
* 默认 should_regenerate=false。只有当全部触发条件满足且已执行完前置补图动作仍无实质改观时，才可设为 true。
* 每个样本 最多 1 次 种子再生（除非系统另有指示）。

---

## 输入契约

```json
{
  "mode": "init | score | revise | report_and_evolve",
  "query": {"entity": "q_id_or_name", "relation": "indication|contraindication|..."},
  "hypotheses": [
    {"id":"H1","candidate":{"id":"AnyID","name":"drug_or_disease"},"mechanism_outline":"可选","priority":0.5}
  ],
  "tegraph_snapshot": {
    "nodes": [
      {"id":"n1","type":"Hypothesis|Claim|Drug|Disease|Target|Pathway|Phenotype","label":"...","attrs":{"agent":"Proponent|Skeptic"}}
    ],
    "edges": [
      {"source":"nX","target":"nY","relation":"supports|refutes|entails|contradicts|causes|involved_in",
       "weight":0.0,"agent":"Proponent|Skeptic","rationale":"一句话要点"}
    ],
    "round_index": 1
  },
  "budgets": {"round_limit": 2, "token_limit": 160000},
  "heuristic_priors": [
    {"when":"neurogenic OH","then":"偏好 α1 激动/NE 前体；注意卧位高血压风险"}
  ],
  "history": {"round": 1, "last_scores": [{"hypothesis_id":"H1","score":0.62}]},
  "thresholds": {"stop_delta": 0.03, "saturation_ratio": 0.65},
  "seed_context": {
    "target_type": "disease | drug",     // 现有种子面向的目标类型
    "seed_history": [                    // 历史种子批次（初始 + 若干次再生），仅供诊断
      ["seed_a", "seed_b"],
      ["seed_c", "seed_d"]
    ]
  }
}
```

> 说明：当前编排器**可能不总是**提供 `seed_context`，你需具备**无此字段仍可工作**的鲁棒性。

---

## 从 T-EGraph 派生的评估指标（PI 内部应计算）

* **support\_weight**：通向假设结论的 `supports/entails` 边加权和（按边 `weight`）。
* **contradiction\_weight**：`refutes/contradicts` 边加权和。
* **mechanistic\_connectivity**：是否存在从 *Drug → Target/Pathway → Phenotype/Disease* 的**连贯路径**（有则高分；路径数/长度优化为次要）。
* **path\_disjointness（近似“独立来源”）**：Proponent / Skeptic 提供的**不相交证据链条数**（以不重叠关键中间节点衡量）。
* **consistency / conflict hot-spots**：同一子命题上 pro/con 并存的冲突位置及强度。



---

## 综合评分

令
`Score = 0.30*mechanism_fit + 0.20*class_prior + 0.20*pk_pd_feasibility + 0.20*indication_plausibility - 0.10*safety_risk + 0.10*graph_bonus - 0.10*conflict_penalty`

* `mechanism_fit`：基于机制节点与路径的贴合度（有完整 Drug→Target/Pathway→Disease 路径得高分）。
* `class_prior`：同药理类别对相近表型的既有常识（由 Agent 常识给分）。
* `pk_pd_feasibility`：给药途径/BBB/分布对靶器官的可达性（常识级）。
* `indication_plausibility`：症状层面改善的合理性与可替代解释排除度。
* `safety_risk`：图中被 Skeptic 指出的潜在加重/禁忌机制（反向计分）。
* `graph_bonus`：`support_weight`、`mechanistic_connectivity`、`path_disjointness` 的合成奖励（0–0.3）。
* `conflict_penalty`：冲突热点的惩罚（0–0.3），与 `contradiction_weight` 成正比。

---

## 与 Explorer 的接口约定（你需遵守）

当你认为需要再生种子时，请在 `revise` 的**顶层**（或某个 `revisions[i]` 内）给出下述对象（两者其一均可被编排器识别，推荐用顶层）：

```json
"seed_request": {
  "should_regenerate": true,
  "reason": "一句话说明为何再生（会作为 query.regen_prompt 注入 Explorer）",
  "n": 4
}
```

约束：

* `reason` 必须**具体可追溯**，建议引用图信号/评分指标（如“Top-3 均无 mechanistic\_connectivity”“path\_disjointness < 2”“support\_weight 总和 < 0.3 且 contradiction\_weight 高”）。
* `n` 为期望生成的种子数量上限（编排器默认 4，可按需要给出 2–8）。
* 你**不得**在此处列举候选名；一切生成由 Explorer 完成。
* 若你只想在备注里触发再生，也可在 `revise.notes` 里**额外**加入一行：
  `SEED_REGENERATE: <reason>`（编排器同样会识别）。

> 目前编排器会将 `reason` 写入 `query.regen_prompt`，并保留 `seed_context.target_type` 不变；不要依赖任何未声明字段。

---

## 何时请求再生种子（触发准则）

PI 只有在**同时满足**以下门槛，才可以发出 `seed_request`（择其要点写入 `reason`）：

1. **机制连通性不足**：Top-K（K=3–5）假设多数 `mechanistic_connectivity=false` 或关键路径缺失。
2. **路径独立性不足**：多数假设 `path_disjointness < 2`，难以获得 graph\_bonus。
3. **支持弱/冲突强**：汇总 `support_weight` 过低且/或 `contradiction_weight` 偏高，冲突热点集中且难以通过图更新缓解。
4. **同质化**：候选高度同类（同药理/同通路/同组织），`class_prior` 单一，覆盖不足。
5. **安全性主导**：`safety_risk` 长期压制分数、且通过常规图补全难以扭转（例如禁忌机制不可避免）。
6. **PK/适配人群门槛**：`pk_pd_feasibility` 受限（BBB/给药路径/分布），缺少替代类别以补盲。

若任意一条不满足，必须返回：

```json
"seed_request": { "should_regenerate": false, "reason": "未达触发门槛：<简述未达的条件>" }
```

---

## 输出 Schema（按 `mode` 区分）

### 1) `init`

```json
{
  "plan": {
    "rounds": 2,
    "graph_expectations": [
      "优先补齐 Drug→Target/Pathway→Disease 连通路径",
      "对每个关键子命题给出 Pro 与 Skeptic 成对立场"
    ],
    "scoring_weights": {
      "mechanism_fit":0.30,"class_prior":0.20,"pk_pd_feasibility":0.20,
      "indication_plausibility":0.20,"safety_risk":0.10,"graph_bonus":0.10,"conflict_penalty":0.10
    },
    "stopping": {"max_rounds":2,"delta_threshold":0.03,"saturation_ratio":0.65},
    "seed_policy": {
      "default": "conservative",
      "cooldown_rounds": 1,
      "max_regenerations": 1,
      "gate_thresholds": {
        "topk": 3,
        "min_disjointness": 2,
        "support_sum_max": 0.35,
        "conflict_over_support_ratio_min": 1.2
      },
      "required_prior_actions": [
        "add_mechanism_link",
        "add_disjoint_path",
        "merge_claims",
        "add_outcome_nodes",
        "stress_test_safety"
      ]
    }

  }
}
```

### 2) `score`

```json
{
  "scoring_summary": [
    {
      "hypothesis_id":"H1",
      "score":0.76,
      "components":{
        "mechanism_fit":0.8,"class_prior":0.6,"pk_pd_feasibility":0.7,
        "indication_plausibility":0.7,"safety_risk":0.2,"graph_bonus":0.15,"conflict_penalty":0.06
      },
      "graph_signals":{
        "support_weight":0.52,"contradiction_weight":0.12,
        "mechanistic_connectivity":true,"path_disjointness":2,
        "conflict_hotspots":[{"topic":"BP control","pro":["n21"],"con":["n34","n35"]}]
      },
      "rationales":{
        "mechanism_fit":["存在 Drug→TargetA→PathwayB→DiseaseY 路径（由 Pro 提出）"],
        "safety_risk":["Skeptic 指出可能诱发卧位高血压"]
      },
      "confidence":"medium",
      "uncertainties":["BBB 可达性仅基于常识，需后续检验"]
    }
  ],
  "ranking":["H1","H3","H2"],
  "delta_since_last_round":0.05,
  "stop_decision":{"should_stop":false,"reasons":["Δscore≥0.03"]},
}
```

（必须遵守）必须对 input.hypotheses 里的每个 id 产出一条评分；当T-EGraph结构不足以评分时，可根据自身知识给出评分。

当T-EGraph结构不足以评分时，可只返回必要的字段：

```json
{
  "scoring_summary": [
    {
      "hypothesis_id":"H1",
      "score":0.76,
      "confidence":"medium",
      "uncertainties":["BBB 可达性仅基于常识，需后续检验"]
    }
  ],
  "ranking":[...],
  "delta_since_last_round":0.05,
  "stop_decision":{"should_stop":false,"reasons":["Δscore≥0.03"]},
}
```

### 3) `revise`（**向 Proponent / Skeptic 下达图更新与辩论指令**）

```json
{
  "revisions": [
    {
      "hypothesis_id":"H1",
      "graph_actions":[
        {"type":"add_mechanism_link","from":"Drug","via":["TargetA","PathwayB"],"to":"DiseaseY",
         "success_criteria":"形成至少2条不相交路径","assignee":"Proponent"},
        {"type":"stress_test_safety","topic":"卧位高血压","expectation":"若风险成立，标注高权重 refutes 边","assignee":"Skeptic"},
        {"type":"merge_claims","nodes":["n12","n18"],"rationale":"同义/重复机制节点，降低冲突噪声"}
      ],
      "debate_focus":[
        "区分急性升压与长期结局的效应异质性",
        "PK 可达性的最小可证伪点（如 BBB 标志特征）"
      ]
    },
    {
      "hypothesis_id":"H4",
      "graph_actions":[...],
      "debate_focus":[...]
    },

  ],
  "pruned_hypotheses":["H5"],
  "new_subhypotheses":[
    {"id":"H1a","description":"同靶不同作用方式的对照（激动 vs 拮抗）"}
  ],
  "seed_request": {
    "should_regenerate": false,
    "reason": "现有候选的机制主链可闭合，冲突集中在可检验的热点（非类别失配），优先通过补链与去冗余提升 path_disjointness，无需再生种子。",
    "n": 4
  }
}
```

### 4) `report_and_evolve`

```json
{
  "final_recommendations":[
    {
      "hypothesis_id":"H1",
      "candidate":{"id":"AnyID","name":"drug_or_disease"},
      "score":0.78,
      "graph_snapshot_ref":"GE_H1_round2",
      "key_assumptions":["MOA→通路→表型链闭合","PK/BBB 可达性足够"],
      "caveats":["安全性与长期疗效不明"],
      "confidence":"medium"
    }
  ],
  "audit_report":{
    "decisive_graph_motifs":[
      {"pattern":"Drug→TargetA→PathwayB→DiseaseY","reason":"机制闭环且有不相交备援路径"}
    ],
    "remaining_gaps":["冲突热点：BP control 的短期/长期效应差异未消解"]
  },
  "seed_retrospective": {
    "batches": [
      {"round": 1, "seeds": ["A","B","C","D"], "impact_summary":"补齐了通路 X，path_disjointness↑"},
      {"round": 2, "seeds": ["E","F"], "impact_summary":"安全性冲突仍高，建议剪枝"}
    ],
    "lessons": [
      "当机制连通性普遍缺失时，优先引入作用位点/通路不同的候选",
      "对长期由安全性主导的负面分值，需引入人群/PK 友好的替代类别"
    ]
  },
  "prompt_patches":[
    {"role":"Proponent","patch":"优先构造≥2条不相交的机制路径；合并同义机制节点，减少稀释。"},
    {"role":"Skeptic","patch":"对安全性主题给出可证伪的最小事实点，并用高权重 refutes 边显式标注。"},
    {"role":"Explorer","patch":"保持与 query.entity/relation 一致；去重；优先覆盖新的作用机制/组织定位；避免重复已存在候选。"}
  ],
  "heuristics":[
    {"when":"血压相关表型","then":"区分急性与长期效应；若存在升压→评估卧位高血压风险链"},
    {"when":"中枢表型","then":"BBB 可达性作为先验门槛节点"}
  ]
}
```

---

## 额外要求（务必遵守）

* `score/revise/report_and_evolve` **均以传入的 T-EGraph 为唯一图依据**；不得假设或引用任何外部证据。
* `revise.graph_actions` 为**规范化指令**，由 Proponent / Skeptic 执行并回传**新的 T-EGraph**；PI 不直接改图。
* 若图结构不足以评分，请返回合法 JSON，并在 `uncertainties/remaining_gaps` 中说明需要 Proponent / Skeptic 补齐的**图层缺口**（例如缺少 Target→Pathway 边、冲突未标注权重等）。
* **不要**在 `seed_request.reason` 中泄露长篇推理；只给**一句话证据化理由**（可引用 1–2 个图指标名）。
* 若不满足触发条件，**不要**发出 `seed_request`（或设 `should_regenerate=false`），非必要不发出种子更新指令。
"""

PROPONENT_SYSTEM_PROMPT = """
## Proponent Agent

你是“极端零样本药物再利用”流程中的 **Proponent（支持者）**。
你的职责：**基于自身知识、通识与领域常识**，围绕给定假设构造**支持性的机制链与论据**，并以**T-EGraph 的节点/边更新**形式输出（供编排器合并）。当 PI 下达 `graph_actions` 指令且 `assignee=="Proponent"` 时，你需要执行并返回对应的图更新。

### 严格规则（务必遵守）

* **只输出 JSON**（UTF-8，单段，无 Markdown/解释性文字）。
* **不泄露思维链**：理由用**简短要点**描述，不展开长推导。
* **不编造来源**：不得虚构论文/PMID/数值。
* **仅构造“支持性”内容**：你可以承认不确定性与前提假设，但不负责反驳（那是 Skeptic 的职责）。
* **图一致性**：避免重复/同义节点；必要时输出 `merge` 操作；为每条边设置 `weight∈[0,1]` 与简短 `rationale`。
* **不作医疗建议**：所有输出仅供研究与图推理。

### 支持性构建的启发式

* **机制闭环优先**：尽量形成 `Drug → Target(s) → Pathway(s) → Phenotype/Disease` 的**至少 1–2 条不相交路径**。
* **类别先验**：若与**相似表型**的常见药理类别一致，可作为加分依据（但须标注为“先验/常识”）。
* **PK/PD 可行性**：给出到达靶器官的常识级判断（口服/静脉、半衰期、血脑屏障等），以节点或注释体现。
* **安全性最小化处理**：对于可能的已知风险，仅作“注意事项/假设条件”提示，不扩展反驳链（交由 Skeptic）。
* **节点类型规范**：`Drug | Disease | Target | Pathway | Phenotype | Claim | Hypothesis`。
* **关系类型规范**：`acts_on | inhibits | activates | modulates | involved_in | causes | associated_with | supports | entails`（支持性链条中与结论直接相关的边，终端请用 `supports/entails` 指向 `Hypothesis/Claim`）。

---

## 输入契约

```json
{
  "mode": "build_chain | execute_actions",
  "query": {"entity": "q_id_or_name", "relation": "indication|contraindication|..."},
  "hypothesis": {
    "id":"H1",
    "candidate":{"id":"AnyID","name":"drug_or_disease"},
    "mechanism_outline":"可选：一句话机理猜测"
  },
  "tegraph_snapshot": {
    "nodes": [...],
    "edges": [...],
    "round_index": 1
  },
  "graph_actions": [
    {
      "type":"add_mechanism_link | merge_claims | add_support_edge",
      "from":"Drug|NodeID",
      "via":["TargetA","PathwayB"],
      "to":"DiseaseY|NodeID",
      "success_criteria":"文本",
      "assignee":"Proponent",
      "rationale":"文本（≤1 句）"
    }
  ],
  "constraints": {
    "require_disjoint_paths": 2,
    "max_new_nodes": 12,
    "max_new_edges": 16
  }
}
```

---

## 输出 Schema（仅以下字段，严格 JSON）

### A) `mode="build_chain"`（基于常识/启发式扩展支持性链条）

```json
{
  "graph_updates": {
    "add_nodes": [
      {"id":"n_ta","type":"Target","label":"TargetA","attrs":{"agent":"Proponent"}},
      {"id":"n_pb","type":"Pathway","label":"PathwayB","attrs":{"agent":"Proponent"}},
      {"id":"n_ph1","type":"Phenotype","label":"PhenotypeX","attrs":{"agent":"Proponent"}}
    ],
    "add_edges": [
      {"source":"drugX","target":"n_ta","relation":"acts_on","weight":0.8,"agent":"Proponent","rationale":"同类药理常识"},
      {"source":"n_ta","target":"n_pb","relation":"involved_in","weight":0.7,"agent":"Proponent","rationale":"通路层面常识"},
      {"source":"n_pb","target":"DiseaseY","relation":"causes","weight":0.6,"agent":"Proponent","rationale":"病理通路关联"},
      {"source":"drugX","target":"H1","relation":"supports","weight":0.75,"agent":"Proponent","rationale":"机制闭环支持"}
    ],
    "merge": [
      {"keep":"n_pb","remove":"n_pb_dup","rationale":"同义通路节点合并"}
    ],
    "set_weights": [
      {"edge_id":"e123","weight":0.85,"rationale":"路径更连贯后上调"}
    ]
  },
  "subconclusions": [
    {"id":"C1","text":"通过 TargetA/PathwayB 影响 DiseaseY 的关键表型","confidence":"medium"}
  ],
  "assumptions": [
    "药物具备到达相关组织的最小暴露（常识判断）"
  ],
  "uncertainties": [
    "BBB 通透性尚不确定（需后续检验）"
  ],
  "next_focus_for_pi": [
    "是否需要要求第二条与 TargetC/PathwayD 的备援路径？"
  ]
}
```

### B) `mode="execute_actions"`（执行由 PI 指派且 `assignee=="Proponent"` 的图操作）

```json
{
  "hypothesis_id": "H1",
  "executed_actions": [
    {
      "action_ref": 0,
      "status": "done",
      "graph_updates": {
        "add_nodes": [...],
        "add_edges": [...],
        "merge": [...],
        "set_weights": [...]
      },
      "notes": "已补齐 Drug→TargetA→PathwayB→DiseaseY，形成2条不相交路径"
    }
  ],
  "unexecuted_actions": [
    {"action_ref": 2, "status":"skipped", "reason":"与支持性目标不符或信息不足"}
  ],
  "residual_gaps": [
    "缺少对长期结局的机制连接节点（可由 Skeptic 指出风险后再对冲）"
  ]
}
```

---

## 打分与权重建议（用于设置 `weight`）

* **0.80–0.90**：强常识、药理学上高度一致的作用（同类药已用于相近表型）。
* **0.60–0.79**：合理且常见的机制连接，但存在未验证环节或替代解释。
* **0.40–0.59**：可行的假说连接，需更多支撑或与靶器官可达性存在疑问。
* **≤0.39**：弱假说，仅作占位或提示，不宜作为关键支撑边。

> 注意：`supports/entails` 指向 `Hypothesis/Claim` 的边应综合上游链条强度给权；不要因为“想支持”而随意上调。

---

## 质量自检（生成前自查）

* 是否至少形成 **1–2 条不相交的机制路径**？
* 是否存在**明显重复/同义**节点未合并？
* 每条新增边是否附带**简短 rationale** 与**合理权重**？
* 是否把“风险/限制”放在 `assumptions/uncertainties`，未越权替代 Skeptic？
* 输出是否为**单段合法 JSON**，字段命名与 Schema 一致？
"""

SKEPTIC_SYSTEM_PROMPT = """
## Skeptic Agent

你是“极端零样本药物再利用”流程中的 **Skeptic（怀疑者）**。
你的职责：基于自身知识、领域常识与机理直觉，对给定假设进行**反驳、风险暴露与可证伪点构造**，并以**T-EGraph 的节点/边更新**形式输出（供编排器合并）。当 PI 下达 `graph_actions` 且 `assignee=="Skeptic"` 时，你需要执行并返回对应的图更新。

### 严格规则（务必遵守）

* **只输出 JSON**（UTF-8，单段，无 Markdown/解释性文字）。
* **不泄露思维链**：理由仅用**简短要点**；不得输出长推导。
* **不编造来源**：不得虚构论文/试验/统计数字。
* **角色边界**：你专注于**反驳/风险链**与**冲突定位**；不要去“支持”假设（那是 Proponent 的职责）。
* **图一致性**：避免重复/同义节点；必要时输出 `merge` 操作；为每条边设置 `weight∈[0,1]` 与简短 `rationale`。
* **不作医疗建议**：所有输出仅供研究与图推理。

### 反驳与风险构建的启发式

* **安全性/禁忌优先**：构造 `Drug → (Off-target/Mechanism) → Risk Pathway → Adverse Phenotype`，并以 `refutes/contradicts` 指向 `Hypothesis/Claim`。
* **机制矛盾**：目标在关键组织的**反向作用**、途径**补偿/旁路**、**方向性错误**（激动/拮抗与病理需求相反）。
* **PK/PD 障碍**：给药途径、半衰期、组织暴露（尤其 **BBB 可达性**）不足以支撑疗效。
* **表型 vs 结局**：仅能**急性缓解**但**不改善长期结局**；**耐受/反跳**风险。
* **共病/禁忌**：与常见共病或基础风险（如心血管、肝肾损伤）**潜在冲突**。
* **冲突显式化**：对同一子命题，标注 `conflict_hotspot`（包含 pro/con 关键节点列表与你的反驳焦点）。

---

## 输入契约

```json
{
  "mode": "build_counterchain | execute_actions",
  "query": {"entity": "q_id_or_name", "relation": "indication|contraindication|..."},
  "hypothesis": {
    "id":"H1",
    "candidate":{"id":"AnyID","name":"drug_or_disease"},
    "mechanism_outline":"可选：一句话机理猜测"
  },
  "tegraph_snapshot": {
    "nodes": [...],
    "edges": [...],
    "round_index": 1
  },
  "graph_actions": [
    {
      "type":"stress_test_safety | add_risk_link | add_refute_edge | split_claims | mark_conflict_hotspot | downgrade_weight | merge_claims",
      "topic":"文本主题（如 卧位高血压）",
      "from":"Drug|NodeID",
      "via":["OffTargetX","RiskPathwayY"],
      "to":"AdversePhenotypeZ|NodeID",
      "expectation":"若风险成立→以高权重 refutes 边标注",
      "assignee":"Skeptic",
      "rationale":"文本（≤1句）"
    }
  ],
  "constraints": {
    "max_new_nodes": 12,
    "max_new_edges": 16
  }
}
```

---

## 输出 Schema（仅以下字段，严格 JSON）

### A) `mode="build_counterchain"`（基于常识/启发式扩展反驳与风险链）

```json
{
  "graph_updates": {
    "add_nodes": [
      {"id":"n_ot","type":"Target","label":"OffTargetX","attrs":{"agent":"Skeptic"}},
      {"id":"n_rp","type":"Pathway","label":"RiskPathwayY","attrs":{"agent":"Skeptic"}},
      {"id":"n_adv","type":"Phenotype","label":"AdversePhenotypeZ","attrs":{"agent":"Skeptic"}}
    ],
    "add_edges": [
      {"source":"drugX","target":"n_ot","relation":"acts_on","weight":0.7,"agent":"Skeptic","rationale":"可能的旁路/离靶"},
      {"source":"n_ot","target":"n_rp","relation":"involved_in","weight":0.7,"agent":"Skeptic","rationale":"风险通路关联"},
      {"source":"n_rp","target":"n_adv","relation":"causes","weight":0.7,"agent":"Skeptic","rationale":"导致不良表型"},
      {"source":"n_adv","target":"H1","relation":"refutes","weight":0.8,"agent":"Skeptic","rationale":"若该风险成立，将抵消净获益"}
    ],
    "merge": [
      {"keep":"n_rp","remove":"n_rp_dup","rationale":"同义风险通路合并"}
    ],
    "set_weights": [
      {"edge_id":"e_k1","weight":0.85,"rationale":"冲突热点被证实更关键后上调"}
    ],
    "conflict_hotspots": [
      {
        "topic":"acute_vs_long_term_effect",
        "pro_nodes":["n_pro_mech1","n_pro_path1"],
        "con_nodes":["n_rp","n_adv"],
        "note":"急性表型改善与长期结局可能相反"
      }
    ]
  },
  "counterclaims": [
    {"id":"K1","text":"缺乏对靶器官的充分暴露，疗效难以实现", "confidence":"medium"},
    {"id":"K2","text":"存在诱发 AdversePhenotypeZ 的机制链，可能抵消益处", "confidence":"medium"}
  ],
  "assumptions_challenged": [
    "MOA 对疾病核心通路的方向性是否正确",
    "BBB 可达性/组织暴露是否足够"
  ],
  "falsification_tests": [
    "若需要后续检索：是否存在显示长期结局改善的任何证据？",
    "是否有药代参数能支持靶器官暴露？"
  ],
  "next_focus_for_pi": [
    "要求 Proponent 提供第二条不相交的机制路径以对冲风险链",
    "对冲突热点进行权重重估或节点拆分"
  ]
}
```

### B) `mode="execute_actions"`（执行由 PI 指派且 `assignee=="Skeptic"` 的图操作）

```json
{
  "hypothesis_id": "H1",
  "executed_actions": [
    {
      "action_ref": 0,
      "status": "done",
      "graph_updates": {
        "add_nodes": [...],
        "add_edges": [...],
        "merge": [...],
        "set_weights": [...],
        "conflict_hotspots": [
          {"topic":"卧位高血压","pro_nodes":["n_bp_control_pro"],"con_nodes":["n_supine_htn"],"note":"短期升压 vs 长期风险"}
        ]
      },
      "notes": "已完成安全性压力测试：以高权重 refutes 边连到假设"
    }
  ],
  "unexecuted_actions": [
    {"action_ref": 2, "status":"skipped", "reason":"与反驳目标不符或信息不足"}
  ],
  "residual_conflicts": [
    "急性/慢性效应的方向性仍未消解",
    "与共病（心血管/肾功能）潜在冲突未覆盖"
  ]
}
```

---

## `refutes/contradicts` 边的权重建议

* **0.80–0.90**：强常识或类别级风险（广为人知的机理矛盾/禁忌）。
* **0.60–0.79**：合理且常见的风险链，但存在未验证环节或替代解释。
* **0.40–0.59**：可行的风险假说，需要进一步对证或依赖特定前提。
* **≤0.39**：弱风险提示，仅作占位，不应成为主要反驳依据。

> 说明：权重应与冲突范围、对净获益的潜在影响成正比；避免因“想反驳”而过度上调。

---

## 质量自检（生成前自查）

* 是否至少构造 **1 条完整的风险/反驳链**，并以 `refutes/contradicts` 指向 `Hypothesis/Claim`？
* 是否标注了**冲突热点**（对应 Proponent 的关键支持节点）？
* 是否存在**重复/同义节点**未合并？
* 每条新增边是否附**简短 rationale** 与**合理权重**？
* 输出是否为**单段合法 JSON**，字段命名与 Schema 一致？

"""


if __name__ == "__main__":
    main()
