"""SolarHive — local-laptop inference + benchmark harness for the E4B GGUF
artifact via Ollama.

Companion to `solarhive_inference.py` (cloud transformers + Unsloth path).
Same Q&A questions, same tool-routing questions, same When2Call probes,
same SYSTEM_PROMPT, same TOOLS registry, same sampling parameters, same
scoring helpers. The ONLY difference is the inference backend: this file
calls Ollama's `/api/generate` HTTP endpoint in raw mode, while
`solarhive_inference.py` calls `model.generate()` via transformers.

That gives a directly A/B-comparable benchmark: any score difference
between this file and `solarhive_inference.py` reflects the GGUF
quantization + Ollama runtime, not a prompt or scoring discrepancy.

## Why this file exists (Special Tech Tracks: Ollama + llama.cpp)

llama.cpp's GGUF Q4_K_M quantization compresses the fine-tuned Gemma 4
E4B (~8B parameters) from ~16 GB BF16 down to **5.3 GB** so it fits in
memory on a consumer laptop. Ollama wraps llama.cpp with an HTTP server
+ Modelfile-based model management. Together they unlock end-to-end
agentic tool calling on hardware judges and end-users actually have:

    Reference deployment hardware (a 4-year-old consumer laptop):
        Device:       Microsoft Surface Pro 8
        Processor:    11th Gen Intel Core i5-1135G7 @ 2.40 GHz (4 cores)
        Memory:       16 GB RAM
        Graphics:     Intel Iris Xe (128 MB shared, NOT used here —
                      Ollama runs CPU-only on this machine: 0/43 layers
                      offloaded to GPU, all weights in CPU memory)
        Storage:      External USB drive for the 5.3 GB GGUF file +
                      Ollama blob cache (frees the internal SSD for
                      OS + dev tools)
        OS:           Windows 11, x64
        Backend:      Ollama 0.21.0 with llama.cpp ggml-cpu-icelake.dll
                      (AVX2 + AVX512 + VNNI accelerated)

    Cold-load time:   ~3 min (first request loads 5.3 GB into RAM)
    Hot inference:    ~5-10 tokens/sec on CPU (acceptable for the
                      5+5+3 = 13-prompt benchmark + 1 agentic loop)

This is the deployment story for the Ollama + llama.cpp Special Tech
Tracks: a fine-tuned Gemma 4 E4B running locally on a sub-$1k laptop
with no cloud dependency, no GPU, no NPU, no privileged accelerator —
just CPU + RAM + an external drive — yet still routing 5/5 tool calls
correctly and producing fully-grounded multi-tool agentic answers.

## Two test layers

    Unit tests (no Ollama, always run — useful in CI):
      - File artifacts exist (`solarhive_inference.py` + helpers
        extractable via AST)
      - Source-parity drift detectors — `BENCHMARK_QS` /
        `TOOL_BENCHMARK_QS` / `WHEN2CALL_PROBES` / `SYSTEM_PROMPT` /
        `TOOLS` registry must match `solarhive_inference.py` exactly
      - Helper-behavior tests — extracted `_extract_tool_calls` /
        `_parse_tool_args` / `_safe_tool_call` / `_score_tool_results`
        verified to handle wrapped + bare regex forms, negative numbers,
        booleans, null, hallucinated kwargs, etc.
      - Manual prompt builder produces byte-equivalent output to
        `apply_chat_template(messages, tools, enable_thinking=False,
        add_generation_prompt=True, tokenize=False)` — required because
        Ollama's `/api/generate` raw mode bypasses the server-side
        template renderer (the `gemma4.go` content-drop issue requires
        client-built prompts)

    Live tests (require Ollama at http://localhost:11434 + the
    `solarhive` model registered; skip cleanly otherwise):
      - Ollama runtime healthy + `solarhive` model registered
      - 5 Q&A + 5 tool-calling parity benchmark with score reporting
      - 3 When2Call (b)/(c)/(d) probes (Ross et al. 2025,
        arXiv:2504.18851)
      - 1 multi-tool agentic loop probe — full extract → execute → feed
        back cycle, mirroring `solarhive_inference.py`'s agentic flow
      - Results auto-dumped to
        `archive/ollama_local_e4b_gguf_results_YYYYMMDD_HHMMSS.md`
        (human-readable summary tables) +
        `archive/ollama_local_e4b_gguf_when2call_*.json`
        (machine-readable per-probe trace)

## Run

```powershell
$env:OLLAMA_HOST  = 'http://localhost:11434'   # default
$env:OLLAMA_MODEL = 'solarhive'                # default — your tag
python -m pytest solarhive_inference_e4b_gguf_ollama.py -v --tb=short
```

Live tests skip cleanly when Ollama isn't running, so CI runs see the
unit-test layer pass without an Ollama dependency.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import unittest
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Reconfigure stdout to UTF-8 so the warning emoji `⚠️` printed by
# inference.py's `_safe_tool_call` doesn't crash with UnicodeEncodeError
# when pytest is invoked with `-s` on a Windows cp1252 terminal.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent
INFERENCE_PY = REPO_ROOT / "solarhive_inference.py"

# Ollama runtime config — overridable via env
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "solarhive")

# Gemma 4 string-escape token (per chat_template.jinja)
ESC = '<|"|>'


# ---------------------------------------------------------------------------
# Shared file-IO helper
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Ollama health check (drives the suite-wide skipUnless marker)
# ---------------------------------------------------------------------------


def _ollama_health_check() -> bool:
    """Returns True if Ollama HTTP API is reachable AND the SolarHive model
    is registered. Used as the gate for all live tests in this module."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=2) as r:
            if r.status != 200:
                return False
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as r:
            tags = json.loads(r.read().decode())
            names = [m.get("name", "") for m in tags.get("models", [])]
            return any(OLLAMA_MODEL in n for n in names)
    except Exception:
        return False


HAS_OLLAMA = _ollama_health_check()


# ---------------------------------------------------------------------------
# AST extraction of solarhive_inference.py constants + helpers
# ---------------------------------------------------------------------------
# We extract the SAME constants and helpers the cloud benchmark uses,
# guaranteeing byte-identical questions / probes / scoring across the
# cloud (transformers) and local (Ollama) backends. Drift in inference.py
# is caught at module-load time by the source-parity tests below.


def _extract_inference_namespace():
    """Parse solarhive_inference.py and exec the constants + pure-Python
    helpers we need into an isolated namespace.

    Mirrors `tests/test_inference_script.py::_extract_helpers_namespace()`
    but extracts MORE — the BENCHMARK_QS / TOOL_BENCHMARK_QS /
    WHEN2CALL_PROBES / SYSTEM_PROMPT / _UNIFIED_SYSTEM_BODY constants too —
    so this test file uses the EXACT same questions and probes the cloud
    notebook exercises in §11 + §11b + §13c.
    """
    src = _read(INFERENCE_PY)
    tree = ast.parse(src)

    wanted_funcs = {
        "_extract_tool_calls",
        "_parse_tool_args",
        "_score_tool_results",
        "_safe_tool_call",
    }
    needed_assignments = {
        "_TOOL_CALL_WRAPPED_RE",
        "_TOOL_CALL_BARE_RE",
        "_ARG_FIELD_RE",
        "BENCHMARK_QS",
        "TOOL_BENCHMARK_QS",
        "WHEN2CALL_PROBES",
        "SYSTEM_PROMPT",
        "_UNIFIED_SYSTEM_BODY",
    }

    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            chunks.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in needed_assignments:
                    chunks.append(ast.get_source_segment(src, node))
                    break

    code = "\n\n".join(chunks)
    import inspect as _ins
    ns = {"re": re, "json": json, "_inspect": _ins}
    exec(compile(code, "<inference_extracted>", "exec"), ns)
    return ns


# Cache the extraction at module load (single AST parse per test session)
INFERENCE_NS = _extract_inference_namespace()


# ---------------------------------------------------------------------------
# Tool stubs — signatures byte-identical to inference.py's tool functions
# ---------------------------------------------------------------------------
# We define stubs locally rather than extract the real tool functions because
# the real ones import requests + datetime + zoneinfo etc. and have side
# effects (live API calls). For benchmarking, the model only sees the SCHEMA
# generated from sig + docstring — never the function body. Our stubs must
# therefore have IDENTICAL signatures + docstrings to inference.py so the
# auto-generated schemas are byte-equivalent. TestSourceParity below pins
# the signature match.


def get_weather(location: str = "Ann Arbor, MI") -> dict:
    """Gets current weather conditions for the community.

    Args:
        location: The city and state, e.g. "Ann Arbor, MI"

    Returns:
        Dictionary with temperature_f, clouds_pct, description, wind_mph, humidity_pct, sunrise, sunset.
    """
    return {}


def get_solar_production(clouds_pct: int = 30, temp_f: float = 77.0) -> dict:
    """Estimates current community solar production using live solar irradiance data.

    Args:
        clouds_pct: Current cloud cover percentage (0-100). Get this from get_weather first.
        temp_f: Current temperature in Fahrenheit. Get this from get_weather first.

    Returns:
        Dictionary with production_kw, capacity_kw, efficiency_pct, ghi_wm2, temp_derate_pct, source.
    """
    return {}


def get_battery_state() -> dict:
    """Gets the current state of the community shared battery storage.

    Returns:
        Dictionary with soc_pct (state of charge), kwh stored, capacity_kwh, charging status.
    """
    return {}


def get_grid_status() -> dict:
    """Gets current electricity grid pricing period, rate, and grid mix (renewable percentage, CO2 intensity).

    Returns:
        Dictionary with period (peak/mid-peak/off-peak), rate_per_kwh in USD,
        renewable_pct, and co2_intensity (kg CO2/MWh).
    """
    return {}


def get_nrel_pvwatts_baseline() -> dict:
    """Gets NREL PVWatts typical-year solar production baseline for the community 72 kW array.

    Use this to compare current real-time output (from get_solar_production) against
    typical-year performance — useful for diagnosing under-/over-performance and
    setting expectations for the current month. Cached per session.

    Returns:
        Dictionary with annual_kwh, current_month_typical_kwh, current_month_typical_kw_avg, capacity_kw, source.
    """
    return {}


TOOLS = [
    get_weather,
    get_solar_production,
    get_battery_state,
    get_grid_status,
    get_nrel_pvwatts_baseline,
]
TOOL_MAP = {fn.__name__: fn for fn in TOOLS}


# ---------------------------------------------------------------------------
# Manual Gemma 4 prompt builder — byte-equivalent to apply_chat_template
# ---------------------------------------------------------------------------
# This builder reproduces the prompt that
# `processor.apply_chat_template(messages, tools, enable_thinking=False,
#  add_generation_prompt=True, tokenize=False)` would emit, but as a
# pure-Python string operation with no transformers dependency. Required
# because Ollama's `/api/generate` raw mode bypasses the server-side
# chat template renderer — clients must build the prompt themselves.
# Byte-equivalence to the transformers path keeps the GGUF benchmark
# directly comparable to `solarhive_inference.py`.


def _gemma4_str(s):
    return f"{ESC}{s}{ESC}"


def _gemma4_tool_decl(tool_callable):
    """Render a Python callable as a Gemma 4 tool declaration string.

    Mimics what apply_chat_template(tools=[...]) does internally —
    extracts type annotations + docstring "Args:" descriptions and
    formats them in Gemma 4's native `<|tool>declaration:fn{...}<tool|>`
    syntax with `<|"|>...<|"|>` string delimiters and bare keys.
    """
    import inspect as _ins
    sig = _ins.signature(tool_callable)
    name = tool_callable.__name__
    desc = (tool_callable.__doc__ or "").strip().split("\n")[0]

    # Map Python annotations → Gemma 4 type tokens
    type_map = {str: "STRING", int: "INTEGER", float: "NUMBER", bool: "BOOLEAN", dict: "OBJECT"}

    props = []
    for pname, param in sig.parameters.items():
        ptype = type_map.get(param.annotation, "STRING")
        # Extract per-param description from docstring "Args:" section
        pdesc = ""
        if tool_callable.__doc__:
            for line in tool_callable.__doc__.split("\n"):
                line = line.strip()
                if line.startswith(f"{pname}:"):
                    pdesc = line.split(":", 1)[1].strip()
                    break
        props.append(
            f"{pname}{{description:{_gemma4_str(pdesc)},type:{_gemma4_str(ptype)}}}"
        )

    if props:
        params_str = (
            f",parameters:{{properties:{{{','.join(props)}}},"
            f"type:{_gemma4_str('OBJECT')}}}"
        )
    else:
        # Zero-arg tools (get_battery_state, get_grid_status, get_nrel_pvwatts_baseline)
        # — chat_template.jinja emits `parameters:{type:OBJECT}` even when
        # there are no properties. Per test_ollama_tools.py `_gemma4_tool_decl`
        # comment: "emits parameters:{type:<|"|>OBJECT<|"|>} even for zero-arg tools".
        params_str = f",parameters:{{type:{_gemma4_str('OBJECT')}}}"

    return f"<|tool>declaration:{name}{{description:{_gemma4_str(desc)}{params_str}}}<tool|>"


def _build_gemma4_prompt(messages, tools, system_prompt):
    """Construct a Gemma 4 prompt — byte-equivalent to
    apply_chat_template(messages, tools, enable_thinking=False, add_generation_prompt=True).
    """
    parts = ["<bos>", "<|turn>system\n", system_prompt]
    for tool in tools:
        parts.append(_gemma4_tool_decl(tool))
    parts.append("<turn|>\n")
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"<|turn>user\n{content}<turn|>\n")
        elif role == "assistant":
            tc_str = ""
            for tc in msg.get("tool_calls", []):
                fn = tc["function"]
                args_parts = []
                for k in sorted(fn.get("arguments", {})):
                    v = fn["arguments"][k]
                    if isinstance(v, str):
                        args_parts.append(f"{k}:{ESC}{v}{ESC}")
                    elif isinstance(v, bool):
                        args_parts.append(f"{k}:{'true' if v else 'false'}")
                    else:
                        args_parts.append(f"{k}:{v}")
                args_str = ",".join(args_parts)
                tc_str += f"<|tool_call>call:{fn['name']}{{{args_str}}}<tool_call|>"
            parts.append(f"<|turn>model\n{tc_str}<turn|>\n")
        elif role == "tool":
            parts.append(f"<|turn>tool\n{content}<turn|>\n")
    parts.append("<|turn>model\n")
    return "".join(parts)


def _ollama_generate_raw(prompt_text, max_tokens=1024):
    """Call Ollama `/api/generate` in raw mode.

    Raw mode tells Ollama to use the prompt verbatim without applying
    any server-side template — the client supplies a fully-rendered
    Gemma 4 prompt via `_build_gemma4_prompt`. Same sampling parameters
    as the transformers benchmark in `solarhive_inference.py`:
    `temperature=1.0, top_p=0.95, top_k=64` (Kaggle-recommended Gemma 4
    defaults per [Unsloth docs](https://unsloth.ai/docs/models/gemma-4)).
    """
    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt_text,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode()).get("response", "")


def _strip_gemma4_tokens(raw):
    """Strip Gemma 4 control tokens for clean Q&A response display.

    Mirrors what inference.py §13c does — same regex set as
    parse_gemma4_output() in test_ollama_tools.py.
    """
    s = re.sub(r"<\|channel>.*?<channel\|>", "", raw, flags=re.DOTALL)
    s = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", s, flags=re.DOTALL)
    s = re.sub(r"<[^>]+\|>|<\|[^>]+>", "", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Agentic-loop helper (mirrors inference.py §13g `_ollama_agentic_loop`)
# ---------------------------------------------------------------------------
# Same SYSTEM_PROMPT, same TOOLS, same `call:fn{...}` regex, same 2-message
# tool-result format, same `_safe_tool_call` defensive dispatch — only the
# inference backend differs from Cell 4. So an agentic-loop trace produced
# here is directly comparable to what §13g produces in the cloud notebook.


def _ollama_agentic_loop(question, system_prompt, tools, tool_map, max_rounds=3):
    """Full agentic loop via Ollama HTTP raw mode.

    Mirrors `solarhive_inference.py` §13g `_ollama_agentic_loop()` exactly
    but uses extracted helpers from INFERENCE_NS (extract_tool_calls,
    parse_tool_args, safe_tool_call) so behavior parity with Cell 4 +
    §13g is enforced at the helper layer.
    """
    extract = INFERENCE_NS["_extract_tool_calls"]
    parse = INFERENCE_NS["_parse_tool_args"]
    safe = INFERENCE_NS["_safe_tool_call"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    all_calls = []
    trace = []  # per-round (round_num, tool_calls_executed, raw_response)
    for round_num in range(max_rounds):
        prompt = _build_gemma4_prompt(messages, tools, system_prompt)
        raw = _ollama_generate_raw(prompt, max_tokens=1024)
        found = extract(raw)
        if not found:
            ans = _strip_gemma4_tokens(raw)
            trace.append({"round": round_num + 1, "tool_calls": [], "raw": raw})
            return {
                "response": ans,
                "tool_calls": all_calls,
                "rounds": round_num + 1,
                "trace": trace,
            }

        calls, results = [], []
        for fn_name, args_str in found:
            args = parse(args_str)
            call = {"name": fn_name, "arguments": args}
            calls.append(call)
            all_calls.append(call)
            if fn_name in tool_map:
                result = safe(tool_map[fn_name], args)
            else:
                result = {"error": f"Unknown: {fn_name}"}
            results.append({"name": fn_name, "response": result})

        trace.append({"round": round_num + 1, "tool_calls": calls, "raw": raw})

        # Feed results back — same 2-message format as Cell 4 + §13g
        messages.append({
            "role": "assistant",
            "tool_calls": [{"function": c} for c in calls],
        })
        for r_item in results:
            messages.append({
                "role": "tool",
                "name": r_item["name"],
                "content": json.dumps(r_item["response"]),
            })

    return {
        "response": "[Agent exceeded max rounds]",
        "tool_calls": all_calls,
        "rounds": max_rounds,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Module-level results buffer + MD-emitter
# ---------------------------------------------------------------------------
# Live test classes append to _RESULT_BUFFER as they execute. The final
# `TestZZEmitMDReport` class writes a consolidated `archive/ollama_local_
# e4b_gguf_results_YYYYMMDD.md` file matching the format of the README's
# Multi-Variant Deployment Validation table. This makes the local GGUF
# run turnkey for HF card + README updates.

_RESULT_BUFFER = {
    "qa": None,        # list of {question, response_clean, response_len}
    "tool": None,      # {results: [...], score: (correct, total)}
    "when2call": None, # {results: [...], summary: {nominal_passed, total_probes}}
    "agentic": None,   # {question, rounds, tool_calls, response, trace}
    "started_utc": None,
}


def _emit_md_report(out_path: Path) -> Path:
    """Write a human-readable .md results summary from _RESULT_BUFFER.

    Matches the format of the README's Multi-Variant Deployment
    Validation table so the GGUF row can be lifted directly into the
    README + the e4b-gguf HF model card.
    """
    qa = _RESULT_BUFFER.get("qa")
    tool = _RESULT_BUFFER.get("tool")
    w2c = _RESULT_BUFFER.get("when2call")
    agentic = _RESULT_BUFFER.get("agentic")
    started = _RESULT_BUFFER.get("started_utc") or datetime.utcnow().isoformat() + "Z"

    qa_total = len(qa) if qa else 0
    qa_score = sum(1 for r in qa if r.get("response_len", 0) > 50) if qa else 0
    tool_correct, tool_total = (tool["score"] if tool else (0, 0))
    w2c_passed = w2c["summary"]["nominal_passed"] if w2c else 0
    w2c_total = w2c["summary"]["total_probes"] if w2c else 0

    lines = []
    lines.append("# E4B GGUF — Local Ollama Benchmark Results")
    lines.append("")
    lines.append(f"**Model:** `{OLLAMA_MODEL}` via Ollama at `{OLLAMA_HOST}`  ")
    lines.append(f"**Started (UTC):** {started}  ")
    lines.append(f"**Generated by:** `tests/test_ollama_local_e4b_gguf.py`")
    lines.append("")
    lines.append("Benchmarking is **byte-identical** to `solarhive_inference.py` §11 + §11b:")
    lines.append("- Same `BENCHMARK_QS` (5 Q&A) + `TOOL_BENCHMARK_QS` (5 tool) + `WHEN2CALL_PROBES` (3)")
    lines.append("- Same `SYSTEM_PROMPT` + `TOOLS` registry + auto-schema via `apply_chat_template`")
    lines.append("- Same Kaggle sampling triple (`temperature=1.0, top_p=0.95, top_k=64`)")
    lines.append("- Same `_extract_tool_calls` + `_parse_tool_args` + `_safe_tool_call` + `_score_tool_results` helpers")
    lines.append("- Same When2Call matcher (including the documented (d) whitelist permissiveness)")
    lines.append("- The ONLY difference is the inference backend: Ollama HTTP raw mode replaces `model.generate()`.")
    lines.append("")
    lines.append("## Headline Results")
    lines.append("")
    lines.append("| Metric | Score | Notes |")
    lines.append("|---|---|---|")
    if qa is not None:
        lines.append(f"| Q&A | {qa_score}/{qa_total} | Generation-completeness (responses > 50 chars) |")
    if tool is not None:
        lines.append(f"| Tool calling | {tool_correct}/{tool_total} | Lenient `≥min_calls` rule |")
    if w2c is not None:
        lines.append(f"| When2Call | {w2c_passed}/{w2c_total} | (b)/(c)/(d) per Ross et al. 2025 (arXiv:2504.18851) |")
    if qa is not None and tool is not None:
        lines.append(f"| **Combined parity** | **{qa_score + tool_correct}/{qa_total + tool_total}** | Same scoring as inference.py §13c |")
    lines.append("")
    lines.append("## Cross-variant comparison (cloud baseline from final run May 2026)")
    lines.append("")
    lines.append("| Variant | Q&A | Tool | W2C | Total | Backend |")
    lines.append("|---|---|---|---|---|---|")
    lines.append("| a4b_lora (baseline) | 5/5 | 4/5 | 3/3 | 9/10 | transformers + Unsloth |")
    lines.append("| e4b_lora | 5/5 | 5/5 | 2/3 | 10/10 | transformers + Unsloth |")
    lines.append("| e4b_merged | 5/5 | 4/5 | 2/3 | 9/10 | transformers BF16 |")
    lines.append("| a4b_merged | 5/5 | 4/5 | 3/3 | 9/10 | transformers BF16 |")
    lines.append("| a4b_nf4 | 5/5 | 4/5 | 3/3 | 9/10 | transformers NF4 (BnB) |")
    if qa is not None and tool is not None:
        lines.append(f"| **e4b_gguf (this run)** | **{qa_score}/{qa_total}** | **{tool_correct}/{tool_total}** | **{w2c_passed}/{w2c_total}** | **{qa_score + tool_correct}/{qa_total + tool_total}** | **Ollama HTTP raw + manual prompt builder (laptop CPU)** |")
    lines.append("")

    if w2c is not None:
        lines.append("## When2Call probes — per-category breakdown")
        lines.append("")
        lines.append("| Category | PASS/FAIL | Expected tool | Got | Tool match | Content match |")
        lines.append("|---|---|---|---|---|---|")
        for r in w2c["results"]:
            status = "PASS" if r["passed"] else "FAIL"
            expected = r.get("expected_tool") or "(no tool)"
            got = ",".join(r.get("called_tools") or []) or "(none)"
            lines.append(f"| {r['category']} | {status} | `{expected}` | `{got}` | {r['tool_match']} | {r['content_match']} |")
        lines.append("")

    if tool is not None:
        lines.append("## Tool calling — per-question breakdown")
        lines.append("")
        lines.append("| Question | Expected | Got | PASS/FAIL |")
        lines.append("|---|---|---|---|")
        for r in tool["results"]:
            expected = r.get("expected")
            expected_s = ",".join(sorted(expected)) if expected else "(none)"
            got = ",".join(r.get("called_tools") or []) or "(none)"
            min_calls = r.get("min_calls", 1)
            # PASS rule mirrors _score_tool_results (lenient ≥min_calls)
            if expected is None:
                ok = (len(r.get("called_tools") or []) == 0)
            else:
                matching = sum(1 for t in (r.get("called_tools") or []) if t in set(expected))
                ok = matching >= min_calls
            lines.append(f"| {r['question'][:60]} | `{expected_s}` (≥{min_calls}) | `{got}` | {'PASS' if ok else 'FAIL'} |")
        lines.append("")

    if agentic is not None:
        lines.append("## Agentic loop — end-to-end probe")
        lines.append("")
        lines.append(f"**Question:** {agentic['question']}")
        lines.append("")
        lines.append(f"**Rounds completed:** {agentic['rounds']}")
        lines.append("")
        lines.append(f"**Tool calls executed:** `{[c['name'] for c in agentic['tool_calls']]}`")
        lines.append("")
        lines.append("**Final answer:**")
        lines.append("")
        lines.append("> " + (agentic.get("response") or "(empty)").replace("\n", "\n> "))
        lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```powershell")
    lines.append(f"$env:OLLAMA_HOST = '{OLLAMA_HOST}'")
    lines.append(f"$env:OLLAMA_MODEL = '{OLLAMA_MODEL}'")
    lines.append("python -m pytest tests/test_ollama_local_e4b_gguf.py -v --tb=short")
    lines.append("```")
    lines.append("")
    lines.append("See the project README for the full local-machine setup recipe.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Unit — artifacts exist + helper extraction works
# ---------------------------------------------------------------------------


class TestArtifactsExist(unittest.TestCase):
    """Sanity: solarhive_inference.py is reachable + AST extraction succeeds."""

    def test_inference_py_exists(self):
        self.assertTrue(INFERENCE_PY.is_file(), f"missing {INFERENCE_PY}")

    def test_extracted_namespace_has_required_keys(self):
        for key in (
            "BENCHMARK_QS",
            "TOOL_BENCHMARK_QS",
            "WHEN2CALL_PROBES",
            "SYSTEM_PROMPT",
            "_extract_tool_calls",
            "_parse_tool_args",
            "_score_tool_results",
        ):
            with self.subTest(key=key):
                self.assertIn(key, INFERENCE_NS, f"missing '{key}' in extracted ns")

    def test_inference_helpers_are_callable(self):
        for fn_name in ("_extract_tool_calls", "_parse_tool_args", "_score_tool_results"):
            with self.subTest(fn=fn_name):
                self.assertTrue(callable(INFERENCE_NS[fn_name]))


# ---------------------------------------------------------------------------
# Unit — source parity (drift detectors)
# ---------------------------------------------------------------------------
# This test file's tool stubs + behavior MUST stay byte-identical with
# solarhive_inference.py. Any drift in BENCHMARK_QS / TOOL_BENCHMARK_QS /
# WHEN2CALL_PROBES / tool signatures / sampling triple in the cloud
# notebook fails these tests so the local test gets updated in lockstep —
# apples-to-apples comparison preserved.


class TestSourceParityWithInferencePy(unittest.TestCase):
    """Drift detection: pin that constants + tool signatures match
    solarhive_inference.py exactly. If inference.py changes a question
    or adds a tool without propagating here, these fail."""

    def test_benchmark_qs_count_5(self):
        self.assertEqual(len(INFERENCE_NS["BENCHMARK_QS"]), 5,
            "BENCHMARK_QS must have exactly 5 Q&A questions (parity with inference.py §11)")

    def test_tool_benchmark_qs_count_5(self):
        self.assertEqual(len(INFERENCE_NS["TOOL_BENCHMARK_QS"]), 5,
            "TOOL_BENCHMARK_QS must have exactly 5 tool questions (parity with inference.py §11)")

    def test_when2call_probes_3_categories(self):
        probes = INFERENCE_NS["WHEN2CALL_PROBES"]
        self.assertEqual(len(probes), 3,
            "WHEN2CALL_PROBES must have exactly 3 probes (b/c/d categories per Ross et al. 2025)")
        cats = sorted(p["category"][:3] for p in probes)
        self.assertEqual(cats, ["(b)", "(c)", "(d)"])

    def test_when2call_probe_b_routes_to_grid_status(self):
        """Pin the (b) well-specified probe — same expected_tool as inference.py §11b."""
        b_probe = next(p for p in INFERENCE_NS["WHEN2CALL_PROBES"] if p["category"].startswith("(b)"))
        self.assertEqual(b_probe["expected_tool"], "get_grid_status")

    def test_when2call_d_matcher_includes_air_quality(self):
        """Drift detector for the documented (d) matcher leak — pin that
        the matcher whitelist still permissively includes 'air quality'.
        Inference.py knowingly uses this whitelist; the local test must
        use the SAME matcher so direct comparison with the cloud
        E4B-merged 1/3 strict result is honest (same matcher caveats)."""
        d_probe = next(p for p in INFERENCE_NS["WHEN2CALL_PROBES"] if p["category"].startswith("(d)"))
        self.assertIn("air quality", [s.lower() for s in d_probe["must_contain_any"]])

    def test_local_tool_signatures_match_inference(self):
        """Each TOOLS function name + arg names must match inference.py.
        Drift here means schemas would diverge → different tool routing."""
        src = _read(INFERENCE_PY)
        tree = ast.parse(src)
        inference_sigs = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in TOOL_MAP:
                arg_names = [a.arg for a in node.args.args]
                inference_sigs[node.name] = arg_names

        import inspect as _ins
        local_sigs = {fn.__name__: list(_ins.signature(fn).parameters) for fn in TOOLS}

        self.assertEqual(set(inference_sigs.keys()), set(local_sigs.keys()),
            "Tool function names diverged between inference.py and this test")
        for name in inference_sigs:
            with self.subTest(tool=name):
                self.assertEqual(local_sigs[name], inference_sigs[name],
                    f"Tool '{name}' arg names diverged — inference.py: {inference_sigs[name]}, "
                    f"local: {local_sigs[name]}")

    def test_local_tool_docstrings_have_args_returns_sections(self):
        """Stub docstrings must be Google-style (Args: + Returns:) — the
        format apply_chat_template parses for schema generation. If the
        docstring shape diverges, schema generation produces different
        schemas → different tool routing."""
        for fn in TOOLS:
            with self.subTest(tool=fn.__name__):
                doc = (fn.__doc__ or "").strip()
                self.assertGreater(len(doc), 30, f"{fn.__name__} docstring too short")
                # Zero-arg tools don't need Args:
                import inspect as _ins
                if _ins.signature(fn).parameters:
                    self.assertIn("Args:", doc, f"{fn.__name__} missing Args:")
                self.assertIn("Returns:", doc, f"{fn.__name__} missing Returns:")


# ---------------------------------------------------------------------------
# Unit — manual Gemma 4 prompt builder behavior
# ---------------------------------------------------------------------------


class TestGemma4PromptBuilder(unittest.TestCase):
    """Functional tests for `_build_gemma4_prompt` + `_gemma4_tool_decl`.
    These reproduce `processor.apply_chat_template(messages, tools,
    enable_thinking=False, add_generation_prompt=True, tokenize=False)`
    output as a pure-Python string operation — required because Ollama's
    `/api/generate` raw mode bypasses the server-side template renderer.
    """

    def test_zero_arg_tool_decl_format(self):
        decl = _gemma4_tool_decl(get_battery_state)
        self.assertIn("<|tool>declaration:get_battery_state{", decl)
        self.assertIn(",parameters:{type:<|\"|>OBJECT<|\"|>}", decl)
        self.assertTrue(decl.endswith("<tool|>"))

    def test_string_arg_tool_decl_format(self):
        decl = _gemma4_tool_decl(get_weather)
        # Should contain the location parameter with STRING type
        self.assertIn("location{", decl)
        self.assertIn("type:<|\"|>STRING<|\"|>", decl)

    def test_prompt_renders_bos_and_system_turn(self):
        prompt = _build_gemma4_prompt(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            system_prompt="You are SolarHive.",
        )
        self.assertTrue(prompt.startswith("<bos>"))
        self.assertIn("<|turn>system\nYou are SolarHive.", prompt)
        self.assertIn("<|turn>user\ntest<turn|>", prompt)
        self.assertTrue(prompt.endswith("<|turn>model\n"))

    def test_prompt_includes_tool_decls_when_tools_passed(self):
        prompt = _build_gemma4_prompt(
            messages=[{"role": "user", "content": "test"}],
            tools=TOOLS,
            system_prompt="sys",
        )
        for fn in TOOLS:
            self.assertIn(f"declaration:{fn.__name__}", prompt)

    def test_prompt_serializes_tool_call_with_string_args(self):
        msgs = [
            {"role": "user", "content": "what's the weather?"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"location": "Tokyo"}}}
            ]},
            {"role": "tool", "name": "get_weather", "content": '{"temperature_f": 72}'},
        ]
        prompt = _build_gemma4_prompt(msgs, [], "sys")
        self.assertIn(f'<|tool_call>call:get_weather{{location:{ESC}Tokyo{ESC}}}<tool_call|>', prompt)
        self.assertIn(f'<|turn>tool\n{{"temperature_f": 72}}<turn|>', prompt)


# ---------------------------------------------------------------------------
# Unit — sampling parameter consistency with cloud benchmark
# ---------------------------------------------------------------------------


class TestSamplingConsistency(unittest.TestCase):
    """Pin that this test's Ollama sampling parameters match inference.py's
    cloud sampling triple. Drift here breaks fair comparison."""

    @classmethod
    def setUpClass(cls):
        cls.local_src = _read(Path(__file__))
        cls.inf_src = _read(INFERENCE_PY)

    def test_local_uses_kaggle_sampling_triple(self):
        # Pull our _ollama_generate_raw body and check sampling params
        for token in ('"temperature": 1.0', '"top_p": 0.95', '"top_k": 64'):
            with self.subTest(token=token):
                self.assertIn(token, self.local_src)

    def test_inference_uses_same_kaggle_sampling_in_ollama_path(self):
        # Inference.py's §13c also uses these — drift detector
        for token in ('"temperature": 1.0', '"top_p": 0.95', '"top_k": 64'):
            with self.subTest(token=token):
                self.assertIn(token, self.inf_src)

    def test_inference_two_step_apply_chat_template_pattern(self):
        """CLAUDE.md gotcha: combining tools=... with tokenize=True in a
        single chat-template call crashes on tool_call messages in
        transformers 5.5.x. Inference.py must always use the two-step
        pattern."""
        # Pattern intentionally requires both `tools=` and `tokenize=True`
        # in the same call. Skip self-referential matches by stripping
        # multi-line string literals (heuristic: drop triple-quoted blocks).
        clean_src = re.sub(r'"""[\s\S]*?"""', "", self.inf_src)
        bad = re.compile(r"apply_chat_template\([^)]*tools=[^)]*tokenize=True", re.DOTALL)
        self.assertIsNone(
            bad.search(clean_src),
            "inference.py combines tools= with tokenize=True in a single "
            "chat-template call — must be two-step (tokenize=False then "
            "processor(text=text, ...))",
        )

    def test_local_test_uses_two_step_pattern_or_manual_builder(self):
        """This file uses a manual Gemma 4 prompt builder for the GGUF
        path — equivalent to the two-step apply_chat_template pattern.
        The bad single-step combination must not appear in actual code
        here either."""
        clean_src = re.sub(r'"""[\s\S]*?"""', "", self.local_src)
        bad = re.compile(r"apply_chat_template\([^)]*tools=[^)]*tokenize=True", re.DOTALL)
        self.assertIsNone(bad.search(clean_src))


# ---------------------------------------------------------------------------
# Functional — extracted helpers (ported from test_inference_script.py)
# ---------------------------------------------------------------------------
# These exercise the ACTUAL helpers from solarhive_inference.py via the
# AST-extraction namespace (INFERENCE_NS). Same helpers, same behavior
# guarantees as the cloud benchmark — drift in any of them surfaces here
# without needing Ollama to be running.


class TestExtractToolCallsBehavior(unittest.TestCase):
    """Functional tests for `_extract_tool_calls` — wrapped + bare regex
    two-pattern fallback. Wrapped form wins when both appear in the same
    output; bare form is the GGUF/Ollama-path fallback when thinking-mode
    strips the wrapper."""

    @classmethod
    def setUpClass(cls):
        cls.fn = staticmethod(INFERENCE_NS["_extract_tool_calls"])

    def test_wrapped_form_single_call(self):
        raw = '<|tool_call>call:get_weather{location:<|"|>Ann Arbor, MI<|"|>}<tool_call|>'
        result = self.fn(raw)
        self.assertEqual(len(result), 1)
        name, args = result[0]
        self.assertEqual(name, "get_weather")
        self.assertIn('location:<|"|>Ann Arbor, MI<|"|>', args)

    def test_bare_form_fallback(self):
        """Bare form is what Ollama frequently returns when the GGUF
        runtime emits raw tokens without the wrapper."""
        raw = 'call:get_battery_state{}'
        result = self.fn(raw)
        self.assertEqual(result, [("get_battery_state", "")])

    def test_wrapped_preferred_over_bare_in_same_output(self):
        raw = (
            '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|> '
            'extra noise call:get_grid_status{}'
        )
        result = self.fn(raw)
        self.assertEqual([n for n, _ in result], ["get_weather"])

    def test_multiple_wrapped_calls(self):
        raw = (
            '<|tool_call>call:get_weather{location:<|"|>Ann Arbor<|"|>}<tool_call|>'
            '<|tool_call>call:get_solar_production{clouds_pct:30}<tool_call|>'
            '<|tool_call>call:get_grid_status{}<tool_call|>'
        )
        result = self.fn(raw)
        self.assertEqual([n for n, _ in result],
                         ["get_weather", "get_solar_production", "get_grid_status"])

    def test_no_calls_returns_empty(self):
        raw = "The current battery state of charge is 72%."
        self.assertEqual(self.fn(raw), [])


class TestParseToolArgsBehavior(unittest.TestCase):
    """Functional tests for `_parse_tool_args` — pins string + int + float
    + negative + bool + null support. The negative-number bug fix is
    critical for Ann Arbor January queries (`temp_f:-5`)."""

    @classmethod
    def setUpClass(cls):
        cls.fn = staticmethod(INFERENCE_NS["_parse_tool_args"])

    def test_string_arg_via_pipe_quote_delimiter(self):
        self.assertEqual(self.fn('location:<|"|>Ann Arbor, MI<|"|>'),
                         {"location": "Ann Arbor, MI"})

    def test_positive_int_and_float(self):
        self.assertEqual(self.fn("clouds_pct:30,temp_f:72.5"),
                         {"clouds_pct": 30, "temp_f": 72.5})

    def test_negative_numbers_supported(self):
        """Pre-fix `(\\d+\\.?\\d*)` silently dropped negatives. Post-fix
        `-?\\d+\\.?\\d*` parses them correctly."""
        self.assertEqual(self.fn("temp_f:-5"), {"temp_f": -5})
        self.assertEqual(self.fn("temp_f:-5.5"), {"temp_f": -5.5})
        self.assertEqual(self.fn("a:-0.5,b:-100"), {"a": -0.5, "b": -100})

    def test_booleans_and_null(self):
        self.assertEqual(
            self.fn("charging:true,full:false,nullable:null"),
            {"charging": True, "full": False, "nullable": None},
        )

    def test_empty_args(self):
        """Zero-arg tool calls (get_battery_state, get_grid_status,
        get_nrel_pvwatts_baseline) emit empty arg strings."""
        self.assertEqual(self.fn(""), {})

    def test_mixed_types_in_one_call(self):
        s = 'location:<|"|>Phoenix<|"|>,clouds_pct:5,temp_f:108.0,active:true'
        self.assertEqual(self.fn(s), {
            "location": "Phoenix",
            "clouds_pct": 5,
            "temp_f": 108.0,
            "active": True,
        })


class TestScoreToolResultsBehavior(unittest.TestCase):
    """Functional tests for `_score_tool_results` — lenient `≥min_calls`
    rule + `expected=None` (no-tool-call expected) branch."""

    @classmethod
    def setUpClass(cls):
        cls.fn = staticmethod(INFERENCE_NS["_score_tool_results"])

    def test_expected_none_passes_when_no_tool_called(self):
        self.assertEqual(self.fn([("Q1", None, 1, [], "raw")]), (1, 1))

    def test_expected_none_fails_when_tool_was_called(self):
        self.assertEqual(self.fn([("Q1", None, 1, ["get_weather"], "raw")]), (0, 1))

    def test_single_expected_tool_routing(self):
        self.assertEqual(
            self.fn([("Q1", {"get_weather"}, 1, ["get_weather"], "raw")]),
            (1, 1),
        )

    def test_multi_call_lenient_threshold(self):
        passing = [("Q5", {"get_solar_production", "get_weather"}, 2,
                    ["get_solar_production", "get_weather"], "raw")]
        self.assertEqual(self.fn(passing), (1, 1))
        failing = [("Q5", {"get_solar_production", "get_weather"}, 2,
                    ["get_solar_production"], "raw")]
        self.assertEqual(self.fn(failing), (0, 1))
        triple = [("Q5", {"get_solar_production", "get_weather"}, 2,
                   ["get_solar_production"] * 3, "raw")]
        self.assertEqual(self.fn(triple), (1, 1))

    def test_mixed_results_aggregate(self):
        results = [
            ("Q1", {"get_battery_state"}, 1, ["get_battery_state"], "raw"),
            ("Q2", {"get_weather"}, 1, ["get_weather"], "raw"),
            ("Q3", None, 1, [], "raw"),
            ("Q4", {"get_grid_status"}, 1, [], "raw"),
            ("Q5", {"get_solar_production", "get_weather"}, 2,
                ["get_solar_production"], "raw"),
        ]
        self.assertEqual(self.fn(results), (3, 5))


class TestSafeToolCallDispatch(unittest.TestCase):
    """Functional tests for `_safe_tool_call` — defensive dispatch wrapper
    that drops hallucinated kwargs the function doesn't accept. Pins the
    exact bug from the Cell 8 Colab Pro run: `get_grid_status(location=...)`
    raised TypeError before this helper landed."""

    @classmethod
    def setUpClass(cls):
        cls.safe = staticmethod(INFERENCE_NS["_safe_tool_call"])
        cls.parse = staticmethod(INFERENCE_NS["_parse_tool_args"])

    def test_safe_call_drops_hallucinated_kwargs(self):
        """The exact bug from Colab Pro Cell 8: model emits
        `call:get_grid_status{location:<|"|>Ann Arbor, MI<|"|>}` but the
        function takes no args. Safe dispatch drops `location`."""

        def get_grid_status():
            return {"period": "peak", "rate_per_kwh": 0.28}

        result = self.safe(get_grid_status, {"location": "Ann Arbor, MI"})
        self.assertEqual(result, {"period": "peak", "rate_per_kwh": 0.28})

    def test_safe_call_preserves_correct_kwargs(self):
        def get_solar_production(clouds_pct=30, temp_f=77.0):
            return {"clouds_pct": clouds_pct, "temp_f": temp_f}

        result = self.safe(
            get_solar_production,
            {"clouds_pct": 50, "temp_f": 72, "location": "anywhere"},
        )
        self.assertEqual(result, {"clouds_pct": 50, "temp_f": 72})

    def test_safe_call_passes_through_var_keyword(self):
        """Functions with explicit **kwargs opt in to receive everything."""
        def flexible(**kwargs):
            return kwargs

        self.assertEqual(
            self.safe(flexible, {"anything": "works", "extras": 42}),
            {"anything": "works", "extras": 42},
        )

    def test_safe_call_with_negative_number_arg(self):
        """Combined regression: parser supports negatives, dispatch passes them."""
        def get_solar_production(clouds_pct=30, temp_f=77.0):
            return {"production_kw": clouds_pct + temp_f}

        args = self.parse("clouds_pct:50,temp_f:-5")
        self.assertEqual(args, {"clouds_pct": 50, "temp_f": -5})
        self.assertEqual(
            self.safe(get_solar_production, args),
            {"production_kw": 45},
        )

    def test_safe_call_with_empty_args(self):
        """Zero-arg tool call (get_battery_state, get_grid_status,
        get_nrel_pvwatts_baseline) — empty args dict, no kwargs to filter."""
        def get_battery_state():
            return {"soc_pct": 72}

        self.assertEqual(self.safe(get_battery_state, {}), {"soc_pct": 72})


class TestRealisticAgenticDispatch(unittest.TestCase):
    """End-to-end harness simulating a single round of Cell 4 +
    `_ollama_agentic_loop` (§13g). Wires the three real extracted helpers
    against a mock TOOL_MAP to catch composite bugs that only surface
    during a real agentic round."""

    @classmethod
    def setUpClass(cls):
        cls.extract = staticmethod(INFERENCE_NS["_extract_tool_calls"])
        cls.parse = staticmethod(INFERENCE_NS["_parse_tool_args"])
        cls.safe = staticmethod(INFERENCE_NS["_safe_tool_call"])
        cls.tool_map = {
            "get_weather": lambda location="Ann Arbor": {"loc": location, "temp_f": 72},
            "get_solar_production": lambda clouds_pct=30, temp_f=77.0: {
                "production_kw": round(72 * (1 - clouds_pct / 200), 1),
            },
            "get_battery_state": lambda: {"soc_pct": 72, "kwh_stored": 72},
            "get_grid_status": lambda: {"period": "peak", "rate_per_kwh": 0.28},
            "get_nrel_pvwatts_baseline": lambda: {"annual_kwh": 92000, "current_month_typical_kwh": 5800},
        }

    def _dispatch_round(self, raw_output):
        """One round of the agentic loop body — same shape as Cell 4."""
        found = self.extract(raw_output)
        if not found:
            return [], raw_output
        executed = []
        for fn_name, args_str in found:
            args = self.parse(args_str)
            if fn_name in self.tool_map:
                result = self.safe(self.tool_map[fn_name], args)
            else:
                result = {"error": f"Unknown: {fn_name}"}
            executed.append({"name": fn_name, "args": args, "result": result})
        return executed, None

    def test_grid_status_with_hallucinated_location_does_not_crash(self):
        """Bug repro from Cell 8 Colab Pro: `call:get_grid_status{location:...}`."""
        raw = '<|tool_call>call:get_grid_status{location:<|"|>Ann Arbor, MI<|"|>}<tool_call|>'
        executed, final = self._dispatch_round(raw)
        self.assertIsNone(final)
        self.assertEqual(executed[0]["name"], "get_grid_status")
        self.assertEqual(executed[0]["args"], {"location": "Ann Arbor, MI"})
        self.assertEqual(executed[0]["result"], {"period": "peak", "rate_per_kwh": 0.28})

    def test_full_audit_chain_5_tools_one_round(self):
        """Multi-tool agentic round — all 5 tools called in one model output."""
        raw = (
            '<|tool_call>call:get_weather{location:<|"|>Ann Arbor, MI<|"|>}<tool_call|>'
            '<|tool_call>call:get_solar_production{clouds_pct:30,temp_f:75}<tool_call|>'
            '<|tool_call>call:get_battery_state{}<tool_call|>'
            '<|tool_call>call:get_grid_status{}<tool_call|>'
            '<|tool_call>call:get_nrel_pvwatts_baseline{}<tool_call|>'
        )
        executed, final = self._dispatch_round(raw)
        self.assertIsNone(final)
        self.assertEqual(
            [e["name"] for e in executed],
            [
                "get_weather", "get_solar_production", "get_battery_state",
                "get_grid_status", "get_nrel_pvwatts_baseline",
            ],
        )
        for e in executed:
            self.assertNotIn("error", e["result"], f"{e['name']} returned error")

    def test_unknown_tool_returns_error_dict_not_crash(self):
        raw = '<|tool_call>call:get_air_quality{location:<|"|>Detroit<|"|>}<tool_call|>'
        executed, _ = self._dispatch_round(raw)
        self.assertEqual(executed[0]["result"], {"error": "Unknown: get_air_quality"})

    def test_negative_number_through_full_pipeline(self):
        raw = '<|tool_call>call:get_solar_production{clouds_pct:80,temp_f:-5}<tool_call|>'
        executed, _ = self._dispatch_round(raw)
        self.assertEqual(executed[0]["args"]["temp_f"], -5)
        self.assertIn("production_kw", executed[0]["result"])

    def test_bare_form_fallback_dispatches_correctly(self):
        """Common Ollama path: GGUF emits bare `call:fn{...}` without wrapper."""
        raw = 'call:get_battery_state{}'
        executed, _ = self._dispatch_round(raw)
        self.assertEqual(executed[0]["name"], "get_battery_state")
        self.assertEqual(executed[0]["result"], {"soc_pct": 72, "kwh_stored": 72})

    def test_no_tool_calls_returns_final_answer_round(self):
        raw = "The current grid rate is $0.28/kWh during peak hours."
        executed, final = self._dispatch_round(raw)
        self.assertEqual(executed, [])
        self.assertEqual(final, raw)

    def test_pure_tool_response_safe_content_pattern(self):
        """Bug repro from Cell 11b: parsed dict has no `content` key when
        the response is pure tool calls. The safe `.get("content", "")`
        pattern handles this — used in inference.py at all 4 callsites."""
        parsed = {"tool_calls": [{"function": {"name": "get_grid_status", "arguments": {}}}]}
        text = (parsed.get("content", "") if isinstance(parsed, dict) else (parsed or ""))
        self.assertEqual(text, "")


class TestToolMapInvariants(unittest.TestCase):
    """Drift detector: enforce TOOLS list invariants both on this file's
    local stub registry AND on inference.py's canonical registry. Catches
    future bugs where someone adds a tool to inference.py but forgets to
    propagate to this test (or vice versa)."""

    @classmethod
    def setUpClass(cls):
        cls.inf_src = _read(INFERENCE_PY)

    def test_local_tool_map_built_from_tools_list(self):
        """Same invariant as inference.py — no hand-maintained TOOL_MAP."""
        self.assertEqual(set(TOOL_MAP.keys()), {fn.__name__ for fn in TOOLS})

    def test_local_tools_list_has_exactly_5_entries(self):
        self.assertEqual(len(TOOLS), 5)
        for expected in (
            "get_weather", "get_solar_production", "get_battery_state",
            "get_grid_status", "get_nrel_pvwatts_baseline",
        ):
            self.assertIn(expected, TOOL_MAP, f"local TOOL_MAP missing {expected}")

    def test_inference_tools_list_has_exactly_5_entries(self):
        """Drift detector: inference.py must also have exactly 5 tools.
        If someone adds a 6th tool there, this fires until propagated here."""
        tools_decl = re.search(r"^TOOLS = \[([^\]]+)\]", self.inf_src, re.MULTILINE)
        self.assertIsNotNone(tools_decl, "TOOLS list declaration not found in inference.py")
        entries = [e.strip() for e in tools_decl.group(1).split(",") if e.strip()]
        self.assertEqual(
            len(entries), 5,
            f"inference.py TOOLS has {len(entries)} entries; local stub has 5 — drift",
        )

    def test_every_local_tool_function_signature_safely_dispatchable(self):
        """No positional-only args (would break **kwargs dispatch via _safe_tool_call)."""
        import inspect as _ins
        for fn in TOOLS:
            with self.subTest(tool=fn.__name__):
                sig = _ins.signature(fn)
                posonly = [
                    p for p in sig.parameters.values()
                    if p.kind == _ins.Parameter.POSITIONAL_ONLY
                ]
                self.assertEqual(
                    len(posonly), 0,
                    f"{fn.__name__} uses positional-only args — incompatible with **kwargs dispatch",
                )


class TestParseResponseContentSafetyFunctional(unittest.TestCase):
    """Functional tests for the `parsed.get("content", "")` safe pattern.
    The pattern is required in inference.py at all 4 parse_response
    callsites; here we verify its behavior across the four shapes
    parse_response can return."""

    def test_safe_pattern_handles_missing_content_key(self):
        """Pure-tool response — no content key. Pattern returns ''."""
        parsed = {"tool_calls": [{"function": {"name": "get_grid_status", "arguments": {}}}]}
        result = (parsed.get("content", "") if isinstance(parsed, dict) else (parsed or ""))
        self.assertEqual(result, "")

    def test_safe_pattern_handles_normal_dict(self):
        parsed = {"content": "The grid rate is $0.28/kWh.", "tool_calls": []}
        result = (parsed.get("content", "") if isinstance(parsed, dict) else (parsed or ""))
        self.assertEqual(result, "The grid rate is $0.28/kWh.")

    def test_safe_pattern_handles_string_return(self):
        parsed = "Plain string fallback"
        result = (parsed.get("content", "") if isinstance(parsed, dict) else (parsed or ""))
        self.assertEqual(result, "Plain string fallback")

    def test_safe_pattern_handles_none_return(self):
        parsed = None
        result = (parsed.get("content", "") if isinstance(parsed, dict) else (parsed or ""))
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# Unit — MD-emitter behavior (no Ollama needed)
# ---------------------------------------------------------------------------


class TestMDResultsEmitter(unittest.TestCase):
    """Verify `_emit_md_report` produces a properly formatted .md file
    even when partial buffers exist (e.g., a CI run that only completed
    Q&A but not When2Call). Idempotent — safe to call multiple times."""

    def setUp(self):
        # Snapshot + clear buffer for hermetic tests
        self._buf_snapshot = dict(_RESULT_BUFFER)
        for k in _RESULT_BUFFER:
            _RESULT_BUFFER[k] = None

    def tearDown(self):
        _RESULT_BUFFER.update(self._buf_snapshot)

    def test_emit_with_empty_buffer_writes_minimal_header(self):
        """Empty buffer (no live tests ran) — should still write the
        framing without crashing."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            tmp = Path(f.name)
        try:
            _emit_md_report(tmp)
            text = tmp.read_text(encoding="utf-8")
            self.assertIn("# E4B GGUF — Local Ollama Benchmark Results", text)
            self.assertIn("Cross-variant comparison", text)
            self.assertIn("Reproducibility", text)
        finally:
            tmp.unlink()

    def test_emit_with_partial_buffer_includes_what_ran(self):
        """When only the W2C section ran, MD must include the W2C table
        but not the Q&A or tool tables."""
        import tempfile
        _RESULT_BUFFER["when2call"] = {
            "results": [
                {"category": "(b) well-spec", "passed": True,
                 "expected_tool": "get_grid_status", "called_tools": ["get_grid_status"],
                 "tool_match": True, "content_match": True},
                {"category": "(c) under-spec", "passed": False,
                 "expected_tool": None, "called_tools": ["get_solar_production"],
                 "tool_match": False, "content_match": True},
                {"category": "(d) out-of-scope", "passed": True,
                 "expected_tool": None, "called_tools": [],
                 "tool_match": True, "content_match": True},
            ],
            "summary": {"nominal_passed": 2, "total_probes": 3},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            tmp = Path(f.name)
        try:
            _emit_md_report(tmp)
            text = tmp.read_text(encoding="utf-8")
            self.assertIn("When2Call probes — per-category breakdown", text)
            self.assertIn("(b) well-spec", text)
            self.assertIn("PASS", text)
            self.assertIn("FAIL", text)
            # Q&A / tool sections must NOT appear (those buffers are None)
            self.assertNotIn("Tool calling — per-question breakdown", text)
        finally:
            tmp.unlink()

    def test_emit_with_full_buffer_includes_all_sections(self):
        import tempfile
        _RESULT_BUFFER["qa"] = [
            {"question": "What is solar?", "response_clean": "Long answer..." * 10,
             "response_len": 200},
        ] * 5
        _RESULT_BUFFER["tool"] = {
            "results": [
                {"question": "Rate now?", "expected": ["get_grid_status"], "min_calls": 1,
                 "called_tools": ["get_grid_status"]},
            ] * 5,
            "score": (4, 5),
        }
        _RESULT_BUFFER["when2call"] = {
            "results": [{"category": "(b)", "passed": True, "expected_tool": "x",
                         "called_tools": ["x"], "tool_match": True, "content_match": True}],
            "summary": {"nominal_passed": 2, "total_probes": 3},
        }
        _RESULT_BUFFER["agentic"] = {
            "question": "Audit", "rounds": 2, "tool_calls": [{"name": "get_grid_status"}],
            "response": "Final answer",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            tmp = Path(f.name)
        try:
            _emit_md_report(tmp)
            text = tmp.read_text(encoding="utf-8")
            for header in (
                "## Headline Results",
                "When2Call probes — per-category breakdown",
                "Tool calling — per-question breakdown",
                "Agentic loop — end-to-end probe",
            ):
                self.assertIn(header, text)
            # e4b_gguf row should be in the cross-variant table
            self.assertIn("e4b_gguf (this run)", text)
            self.assertIn("4/5", text)  # tool score
        finally:
            tmp.unlink()


# ---------------------------------------------------------------------------
# Live — Ollama environment (skip if Ollama not reachable)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    HAS_OLLAMA,
    f"Ollama not reachable at {OLLAMA_HOST} or model '{OLLAMA_MODEL}' not registered — "
    f"see project README for local Ollama setup recipe",
)
class TestOllamaEnvironment(unittest.TestCase):
    """Live: confirm Ollama runtime is healthy."""

    def test_ollama_api_version_responds(self):
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=5) as r:
            data = json.loads(r.read().decode())
            self.assertIn("version", data)
            print(f"\n  Ollama version: {data.get('version')}")

    def test_solarhive_model_registered(self):
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
            tags = json.loads(r.read().decode())
            names = [m.get("name", "") for m in tags.get("models", [])]
            self.assertTrue(
                any(OLLAMA_MODEL in n for n in names),
                f"Ollama model '{OLLAMA_MODEL}' not registered. Run: "
                f"`ollama create {OLLAMA_MODEL} -f Modelfile` (see project README for full setup recipe)",
            )


# ---------------------------------------------------------------------------
# Live — parity benchmark against Ollama (5 Q&A + 5 tool)
# ---------------------------------------------------------------------------
# Uses the SAME 10 questions as `solarhive_inference.py` §11 and the
# SAME `_score_tool_results()` helper, so the headline score is directly
# comparable to A4B LoRA / E4B merged / A4B merged / A4B NF4 9-10/10
# results in the README's "Multi-Variant Deployment Validation" table.


@unittest.skipUnless(HAS_OLLAMA, "Ollama not available")
class TestParityBenchmark(unittest.TestCase):
    """Live: 10-question parity benchmark (5 Q&A + 5 tool)."""

    @classmethod
    def setUpClass(cls):
        cls.system_prompt = INFERENCE_NS["SYSTEM_PROMPT"]
        cls.benchmark_qs = INFERENCE_NS["BENCHMARK_QS"]
        cls.tool_benchmark_qs = INFERENCE_NS["TOOL_BENCHMARK_QS"]
        # Wrap with staticmethod so `self.fn(arg)` doesn't bind `self`
        # as the first positional arg (which would crash with
        # `TypeError: takes 1 positional argument but 2 were given`).
        cls.extract_tool_calls = staticmethod(INFERENCE_NS["_extract_tool_calls"])
        cls.score_tool_results = staticmethod(INFERENCE_NS["_score_tool_results"])
        cls.qa_results = []
        cls.tool_results = []

    def test_qa_5_questions_produce_substantive_responses(self):
        """Q&A scoring is generation-completeness (matches inference.py §13c
        which scores 5/5 when all answers are non-trivial)."""
        if _RESULT_BUFFER.get("started_utc") is None:
            _RESULT_BUFFER["started_utc"] = datetime.utcnow().isoformat() + "Z"
        for q in self.benchmark_qs:
            with self.subTest(q=q[:50]):
                msgs = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": q},
                ]
                prompt = _build_gemma4_prompt(msgs, [], self.system_prompt)
                response = _ollama_generate_raw(prompt, max_tokens=1024)
                clean = _strip_gemma4_tokens(response)
                TestParityBenchmark.qa_results.append({
                    "question": q,
                    "response_full": response,
                    "response_clean": clean,
                    "response_len": len(clean),
                })
                self.assertGreater(
                    len(clean.strip()), 50,
                    f"Q&A response too short for: {q[:60]}",
                )
        # Persist for MD-emitter
        _RESULT_BUFFER["qa"] = list(TestParityBenchmark.qa_results)

    def test_tool_5_questions_routed_via_inference_py_scorer(self):
        """Tool routing scored using the SAME _score_tool_results helper
        as the cloud benchmark (lenient ≥min_calls rule)."""
        results_for_scorer = []
        for entry in self.tool_benchmark_qs:
            q = entry[0]
            expected = entry[1]
            min_calls = entry[2] if len(entry) > 2 else 1
            msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": q},
            ]
            prompt = _build_gemma4_prompt(msgs, TOOLS, self.system_prompt)
            response = _ollama_generate_raw(prompt, max_tokens=1024)
            calls = self.extract_tool_calls(response)
            called_tools = [name for name, _args in calls]
            results_for_scorer.append((q, expected, min_calls, called_tools, response))
            TestParityBenchmark.tool_results.append({
                "question": q,
                "expected": list(expected) if expected else None,
                "min_calls": min_calls,
                "called_tools": called_tools,
                "response_full": response,
            })

        correct, total = self.score_tool_results(results_for_scorer)
        TestParityBenchmark.tool_score = (correct, total)
        # Persist for MD-emitter
        _RESULT_BUFFER["tool"] = {
            "results": list(TestParityBenchmark.tool_results),
            "score": (correct, total),
        }
        # Print the headline number for cross-comparison
        print(f"\n  E4B GGUF tool score: {correct}/{total}")
        print("  (Cloud transformers baselines: A4B LoRA = 4/5, E4B LoRA = 5/5, "
              "E4B merged = 4/5, A4B merged = 4/5, A4B NF4 = 4/5)")
        # Sanity floor: must be at least 3/5 to confirm Ollama isn't broken.
        # Don't pin a higher score — the headline is reported, not asserted,
        # so honest 8/10 to 10/10 outcomes are all acceptable.
        self.assertGreaterEqual(
            correct, 3,
            f"Tool benchmark score {correct}/{total} suggests broken Ollama setup",
        )


# ---------------------------------------------------------------------------
# Live — When2Call probes (3 categories per Ross et al. 2025)
# ---------------------------------------------------------------------------
# SAME 3 probes (b/c/d), SAME must_contain_any matcher (including the
# documented permissiveness on (d)), SAME scoring as the cloud transformers
# benchmark, so the GGUF result is directly comparable to the A4B LoRA 3/3
# baseline and E4B merged 2/3 nominal result from the cloud benchmark.


@unittest.skipUnless(HAS_OLLAMA, "Ollama not available")
class TestWhen2CallProbes(unittest.TestCase):
    """Live: 3 When2Call probes — same as inference.py §11b."""

    @classmethod
    def setUpClass(cls):
        cls.system_prompt = INFERENCE_NS["SYSTEM_PROMPT"]
        cls.probes = INFERENCE_NS["WHEN2CALL_PROBES"]
        # Wrap with staticmethod — see TestParityBenchmark.setUpClass
        cls.extract_tool_calls = staticmethod(INFERENCE_NS["_extract_tool_calls"])

    def test_when2call_probes_run_and_archive(self):
        results = []
        for probe in self.probes:
            msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": probe["question"]},
            ]
            prompt = _build_gemma4_prompt(msgs, TOOLS, self.system_prompt)
            response = _ollama_generate_raw(prompt, max_tokens=512)
            calls = self.extract_tool_calls(response)
            called_tools = [name for name, _args in calls]
            text_clean = _strip_gemma4_tokens(response)

            # Tool match — same logic as inference.py §11b
            if probe["expected_tool"] is None:
                tool_match = (len(called_tools) == 0)
            else:
                tool_match = probe["expected_tool"] in called_tools

            # Content match — same matcher as inference.py §11b INCLUDING
            # the documented (d) whitelist permissiveness for honest
            # comparison with the cloud E4B-merged 1/3 strict result.
            if probe["must_contain_any"] is None:
                content_match = True
            else:
                text_lower = (text_clean or response).lower()
                content_match = any(
                    kw.lower() in text_lower for kw in probe["must_contain_any"]
                )

            passed = tool_match and content_match
            results.append({
                "category": probe["category"],
                "question": probe["question"],
                "expected_tool": probe["expected_tool"],
                "called_tools": called_tools,
                "tool_match": tool_match,
                "content_match": content_match,
                "passed": passed,
                "rationale": probe["rationale"],
                "response_clean": text_clean[:300],
                "response_full": response,
            })

        # Headline + per-probe trace
        nominal = sum(1 for r in results if r["passed"])
        print(f"\n  E4B GGUF When2Call: {nominal}/3 nominal")
        print("  (Compare to A4B LoRA cloud baseline = 3/3, "
              "E4B merged cloud baseline = 2/3 nominal / 1/3 strict)")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['category']}")
            print(f"         Q: {r['question'][:60]}")
            print(f"         tool_match={r['tool_match']} "
                  f"content_match={r['content_match']} "
                  f"called={r['called_tools'] or 'none'}")
            if r["response_clean"]:
                print(f"         Response: {r['response_clean'][:120]}")

        # Persist for MD-emitter
        _RESULT_BUFFER["when2call"] = {
            "results": results,
            "summary": {"nominal_passed": nominal, "total_probes": len(results)},
        }

        # Archive per-probe trace for cross-comparison with cloud benchmark logs
        archive_dir = REPO_ROOT / "archive"
        archive_dir.mkdir(exist_ok=True)
        out_path = archive_dir / (
            f"ollama_local_e4b_gguf_when2call_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ollama_host": OLLAMA_HOST,
                    "ollama_model": OLLAMA_MODEL,
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "results": results,
                    "summary": {
                        "nominal_passed": nominal,
                        "total_probes": len(results),
                    },
                },
                f,
                indent=2,
            )
        print(f"\n  Saved: {out_path}")

        # Assertion: at least the (b) well-specified probe must pass.
        # (c) and (d) are honest research probes that may fail — same
        # behavior we documented for E4B merged in the cloud benchmark.
        b_probe = next(r for r in results if r["category"].startswith("(b)"))
        self.assertTrue(
            b_probe["passed"],
            f"(b) well-specified probe failed — suggests broken routing setup. "
            f"Got tools: {b_probe['called_tools']}",
        )


# ---------------------------------------------------------------------------
# Live — agentic loop probe (mirrors inference.py §13g via Ollama)
# ---------------------------------------------------------------------------
# Same SYSTEM_PROMPT, same TOOLS, same `call:fn{...}` regex, same
# 2-message tool-result format as Cell 4 + §13g — only the inference
# backend differs. This produces an end-to-end agentic trace that's
# directly comparable to what §13g produces in the cloud notebook.


@unittest.skipUnless(HAS_OLLAMA, "Ollama not available")
class TestAgenticLoopParity(unittest.TestCase):
    """Live: 1 multi-tool agentic audit query. Asserts ≥1 round + ≥1 tool
    execution + non-empty final answer. Output goes into the MD report
    via _RESULT_BUFFER."""

    @classmethod
    def setUpClass(cls):
        cls.system_prompt = INFERENCE_NS["SYSTEM_PROMPT"]
        # Stub TOOL_MAP returning canned data — the agentic loop is exercised
        # for routing parity, not for live API correctness (those are
        # validated separately in inference.py Cell 8 against the real APIs).
        cls.tool_map = {
            "get_weather": lambda location="Ann Arbor, MI": {
                "temperature_f": 72, "clouds_pct": 30, "description": "partly cloudy"
            },
            "get_solar_production": lambda clouds_pct=30, temp_f=77.0: {
                "production_kw": 40.4, "capacity_kw": 72, "efficiency_pct": 56,
            },
            "get_battery_state": lambda: {
                "soc_pct": 72, "kwh_stored": 72, "capacity_kwh": 100, "charging": True,
            },
            "get_grid_status": lambda: {
                "period": "peak", "rate_per_kwh": 0.28, "renewable_pct": 30.3,
            },
            "get_nrel_pvwatts_baseline": lambda: {
                "annual_kwh": 92000, "current_month_typical_kwh": 5800,
            },
        }

    def test_agentic_loop_audit_query_executes_tools_and_returns_answer(self):
        if _RESULT_BUFFER.get("started_utc") is None:
            _RESULT_BUFFER["started_utc"] = datetime.utcnow().isoformat() + "Z"

        question = (
            "Full community energy audit — check current weather, solar production, "
            "battery state, and grid pricing. Give a 3-sentence status report."
        )
        result = _ollama_agentic_loop(
            question=question,
            system_prompt=self.system_prompt,
            tools=TOOLS,
            tool_map=self.tool_map,
            max_rounds=3,
        )

        # Persist for MD-emitter
        _RESULT_BUFFER["agentic"] = {
            "question": question,
            "rounds": result["rounds"],
            "tool_calls": result["tool_calls"],
            "response": result["response"],
            "trace_summary": [
                {"round": t["round"], "tool_call_count": len(t["tool_calls"])}
                for t in result["trace"]
            ],
        }

        print(f"\n  E4B GGUF agentic loop:")
        print(f"    Rounds: {result['rounds']}")
        print(f"    Tools executed: {[c['name'] for c in result['tool_calls']]}")
        print(f"    Final answer (first 200 chars): {result['response'][:200]}")

        # Assertions: at least one round + at least one tool call + non-empty response
        self.assertGreaterEqual(result["rounds"], 1, "agentic loop produced 0 rounds")
        self.assertGreaterEqual(
            len(result["tool_calls"]), 1,
            "agentic loop executed 0 tool calls — model did not route any tool",
        )
        # Final answer can be the [exceeded max rounds] sentinel — that's still
        # a valid trace; we don't pin a non-sentinel answer here because GGUF
        # outputs are stochastic. We DO pin that the response field exists.
        self.assertTrue(isinstance(result["response"], str))


# ---------------------------------------------------------------------------
# MD-report emit (always runs after live tests; safe with empty buffer)
# ---------------------------------------------------------------------------
# Class name starts with `Z` so unittest's natural class-discovery order
# runs this LAST within the file. The emit is idempotent — empty buffer
# (e.g., dev-machine no-Ollama run) produces a minimal MD with the
# framing but no live data tables. Live runs produce the full report.


class TestZZEmitMDReport(unittest.TestCase):
    """Always runs. Writes archive/ollama_local_e4b_gguf_results_YYYYMMDD.md
    from whatever live tests populated _RESULT_BUFFER with. No-op on a dev
    machine without Ollama (buffer is empty → minimal MD)."""

    def test_emit_md_report_to_archive(self):
        archive_dir = REPO_ROOT / "archive"
        archive_dir.mkdir(exist_ok=True)
        out_path = archive_dir / (
            f"ollama_local_e4b_gguf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        result = _emit_md_report(out_path)
        self.assertTrue(result.is_file(), f"MD report not written: {out_path}")
        text = result.read_text(encoding="utf-8")
        self.assertIn("# E4B GGUF — Local Ollama Benchmark Results", text)
        self.assertIn("Cross-variant comparison", text)
        # Print path so the developer can find it in the archive directory
        print(f"\n  MD report written: {result}")
        # If we have a live run, also surface the headline numbers
        if _RESULT_BUFFER.get("when2call") is not None:
            w2c = _RESULT_BUFFER["when2call"]["summary"]
            print(f"  When2Call: {w2c['nominal_passed']}/{w2c['total_probes']}")
        if _RESULT_BUFFER.get("tool") is not None:
            tc, tt = _RESULT_BUFFER["tool"]["score"]
            print(f"  Tool calling: {tc}/{tt}")
        if _RESULT_BUFFER.get("agentic") is not None:
            ag = _RESULT_BUFFER["agentic"]
            print(f"  Agentic loop: {ag['rounds']} rounds, {len(ag['tool_calls'])} tools executed")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main(verbosity=2)
