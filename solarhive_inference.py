# -*- coding: utf-8 -*-
"""SolarHive — Gemma 4 Inference & Agent Notebook
================================================
SolarHive is an open-source intelligence layer designed to coordinate
community microgrids & community-based storage via fuel cells, pool
midday energy surplus across these microgrids, and eliminate stranded
capacity. It also helps forecast solar irradiance and cloud cover to
plan ahead.

PURPOSE: Core SolarHive prototype using Gemma 4's NATIVE tool-use protocol.
The model autonomously decides which tools to call via <|tool_call> tokens.

Features:
  1. Native function calling — model decides which APIs to invoke
  2. VQA sky analysis — multimodal + tool calling in one turn
  3. VQA panel inspection
  4. Full agentic loop: Define → Model Decides → Execute → Respond

SETUP: Google Colab Pro with GPU (RTX PRO 6000 Blackwell 96GB recommended)

Gemma is a trademark of Google LLC.
PRIZE TARGETS: Main Track, Global Resilience, Ollama

## 0: Dependencies (RUN FIRST, THEN RESTART RUNTIME IF WARNED)
"""

# Colab Pro: RTX PRO 6000 Blackwell 96GB (BF16). Auto-detects VRAM for NF4 fallback.
import subprocess as _sp, sys as _sys

_sp.check_call([
    _sys.executable, "-m", "pip", "install", "-q",
    "unsloth",               # LoRA adapter loading via FastVisionModel (Cell 2b)
    "transformers>=5.5.0",   # Gemma 4 (gemma4 model_type) added in 5.5.0
    "-U", "accelerate",      # must be latest for transformers 5.5.x compatibility
    "bitsandbytes",          # 4-bit NF4 backend for Cell 2
])

# Read on-disk versions via pip show — bypasses any stale sys.modules cache.
def _pkg_version(pkg):
    try:
        out = _sp.check_output([_sys.executable, "-m", "pip", "show", pkg], text=True)
        for line in out.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"

for _pkg in ("transformers", "accelerate", "bitsandbytes"):
    print(f"  {_pkg:<14}: {_pkg_version(_pkg)}")

# Warn if any package is already cached in this kernel — requires restart.
_cached_mods = [m for m in ("transformers", "accelerate", "bitsandbytes") if m in _sys.modules]
if _cached_mods:
    print("=" * 60)
    for _m in _cached_mods:
        print(f"⚠️  {_m} {getattr(_sys.modules[_m], '__version__', '?')} cached in memory.")
    print("⚠️  Kernel → Restart, then re-run from Cell 0.")
    print("=" * 60)
    raise SystemExit("Kernel restart required.")

print("✓ Cell 0 complete. Proceed to Cell 1.")

"""## 1: Environment & API Keys"""

import unsloth  # must be imported before transformers for Unsloth optimizations
import kagglehub
import os, torch, json, re, random, requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from PIL import Image

assert torch.cuda.is_available(), "Enable GPU: Runtime → Change runtime type → GPU"

# GPU diagnostic
_p = torch.cuda.get_device_properties(0)
_vram_gb = _p.total_memory / 1e9
print(f"GPU: {_p.name} ({_vram_gb:.0f} GB VRAM)")

import transformers
print(f"transformers: {transformers.__version__}")
_ver = tuple(int(p) for p in transformers.__version__.split(".")[:3] if p.isdigit())
if _ver < (5, 5, 0):
    raise RuntimeError(f"transformers {transformers.__version__} too old — need >=5.5.0 for Gemma 4.")

# API keys — load from Kaggle Secrets (Kaggle) or Colab userdata (Colab)
import os as _os
_on_kaggle = _os.path.exists("/kaggle/working")
if _on_kaggle:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    OWM_API_KEY  = secrets.get_secret("OWM_API_KEY")
    NREL_API_KEY = secrets.get_secret("NREL_API_KEY")
    EIA_API_KEY  = secrets.get_secret("EIA_API_KEY")
    print("   Keys loaded from Kaggle Secrets")
else:
    from google.colab import userdata
    OWM_API_KEY  = userdata.get("OWM_API_KEY")
    NREL_API_KEY = userdata.get("NREL_API_KEY")
    EIA_API_KEY  = userdata.get("EIA_API_KEY")
    print("   Keys loaded from Colab Secrets")

LAT, LON = 42.2808, -83.7430
COMMUNITY_CAPACITY_KW = 72
BATTERY_CAPACITY_KWH = 100

# Mount Google Drive for persistent model cache (Colab only)
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass

print("✅ Cell 1 complete — proceed to Cell 2")

"""## 2a: Download Model"""

# Checks Google Drive cache first (persists across runtime restarts).
# Falls back to kagglehub download (~48 GB, 10-20 min) if not cached.
# After first download, copy to Drive:
#   !cp -r /root/.cache/kagglehub/models/google/gemma-4 /content/drive/MyDrive/models/
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import time as _time

_DRIVE_MODEL_PATH = "/content/drive/MyDrive/models/gemma-4/transformers/gemma-4-26b-a4b-it/1"

if os.path.isdir(_DRIVE_MODEL_PATH):
    MODEL_PATH = _DRIVE_MODEL_PATH
    print(f"✅ Loading from Google Drive cache: {MODEL_PATH}")
else:
    print("Model not in Drive — downloading via kagglehub (this takes 10-20 min)...")
    print("   (output suppressed to prevent Colab display overflow)")
    _dl_start = _time.time()
    # Suppress kagglehub's verbose download output — it floods Colab's display
    # buffer and causes blank output cells.
    from IPython.utils.capture import capture_output as _capture
    with _capture():
        MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-26b-a4b-it")
    print(f"✅ Download complete in {_time.time() - _dl_start:.0f}s: {MODEL_PATH}")
    print("💡 To cache for next time, run in a new cell:")
    print("   !cp -r /root/.cache/kagglehub/models/google/gemma-4 /content/drive/MyDrive/models/")

"""## 2b: Load Model"""

# Auto-detect: >=55GB free → BF16 (needs ~48GB + headroom) | <55GB → 4-bit NF4
_free_vram_gb = torch.cuda.mem_get_info(0)[0] / 1e9
_load_4bit = _free_vram_gb < 55
print(f"GPU has {_free_vram_gb:.0f} GB free — loading in {'4-bit NF4' if _load_4bit else 'BF16 (full precision)'}")

# --- Try Unsloth LoRA loading (handles Gemma4ClippableLinear targets that PEFT rejects) ---
_LORA_PATHS = [
    "/content/drive/MyDrive/models/solarhive_a4b_lora",
    "solarhive_a4b_lora",
]
_lora_loaded = False
for _lp in _LORA_PATHS:
    if os.path.isdir(_lp):
        try:
            from unsloth import FastVisionModel
            # Ensure adapter config points to current base model path on this runtime
            _cfg_path = os.path.join(_lp, "adapter_config.json")
            if os.path.exists(_cfg_path):
                with open(_cfg_path) as f:
                    _cfg = json.load(f)
                if _cfg.get("base_model_name_or_path") != MODEL_PATH:
                    _cfg["base_model_name_or_path"] = MODEL_PATH
                    with open(_cfg_path, "w") as f:
                        json.dump(_cfg, f, indent=2)
                    print(f"   Patched adapter config → {MODEL_PATH}")
            model, processor = FastVisionModel.from_pretrained(
                model_name=_lp,
                load_in_4bit=_load_4bit,
                dtype=torch.bfloat16,
            )
            FastVisionModel.for_inference(model)
            print(f"✅ Fine-tuned LoRA adapters loaded from: {_lp}")
            _lora_loaded = True
        except Exception as e:
            print(f"⚠️  Unsloth LoRA load failed: {e} — falling back to base model")
        break

# --- Fallback: load base model without LoRA via transformers ---
if not _lora_loaded:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if not _load_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb_config, device_map="cuda:0", trust_remote_code=True,
        )
    print("ℹ️  No fine-tuned adapters — using base 26B A4B model")
    print("   Run solarhive_finetune.py Part B to create fine-tuned adapters.")

print(f"✅ Gemma 4 26B A4B loaded on {model.device}")

"""## 2c: Smoke Test"""

# Confirm model + processor + chat template + parse_response work end-to-end.
_smoke_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]
_smoke_text = processor.apply_chat_template(
    _smoke_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
_smoke_inputs = processor(text=_smoke_text, return_tensors="pt").to(model.device)
_smoke_in_len = _smoke_inputs["input_ids"].shape[-1]

with torch.no_grad():
    _smoke_out = model.generate(
        **_smoke_inputs, max_new_tokens=1024,
        temperature=1.0, top_p=0.95, top_k=64,
    )

_smoke_raw = processor.decode(_smoke_out[0][_smoke_in_len:], skip_special_tokens=False)
_smoke_parsed = processor.parse_response(_smoke_raw)
_smoke_clean = _smoke_parsed["content"] if isinstance(_smoke_parsed, dict) else _smoke_parsed
print("─" * 60)
print("Smoke test — Gemma 4 response:")
print(_smoke_clean)
print("─" * 60)
print("✅ Chat template, generate, and parse_response all working")

"""## 3: Tool Definitions (typed + Google-style docstrings)"""

# These are passed to apply_chat_template(tools=[...]) so Gemma 4 can
# autonomously decide which to call. Type hints + docstrings = auto-schema.

def get_weather(location: str = "Ann Arbor, MI") -> dict:
    """Gets current weather conditions for the community.

    Args:
        location: The city and state, e.g. "Ann Arbor, MI"

    Returns:
        Dictionary with temperature_f, clouds_pct, description, wind_mph, humidity_pct, sunrise, sunset.
    """
    _tz = ZoneInfo("America/New_York")
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": LAT, "lon": LON, "appid": OWM_API_KEY, "units": "imperial"},
            timeout=10,
        ).json()
        return {
            "temperature_f": r["main"]["temp"],
            "clouds_pct": r["clouds"]["all"],
            "description": r["weather"][0]["description"],
            "wind_mph": r["wind"]["speed"],
            "humidity_pct": r["main"]["humidity"],
            "sunrise": datetime.fromtimestamp(r["sys"]["sunrise"], tz=_tz).strftime("%H:%M"),
            "sunset": datetime.fromtimestamp(r["sys"]["sunset"], tz=_tz).strftime("%H:%M"),
        }
    except Exception as e:
        return {"error": str(e), "clouds_pct": 30, "temperature_f": 72,
                "description": "partly cloudy", "wind_mph": 5.0,
                "humidity_pct": 50, "sunrise": "07:00", "sunset": "20:00"}


def _get_current_ghi():
    """Fetch current Global Horizontal Irradiance (W/m²) from Open-Meteo.

    Free API, no key required. Uses NOAA GFS + HRRR satellite models.
    Inherently accounts for cloud thickness, sun angle, atmosphere, and season.
    """
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": LAT, "longitude": LON, "current": "shortwave_radiation"},
            timeout=10,
        ).json()
        return r["current"]["shortwave_radiation"]
    except Exception:
        return None


def get_solar_production(clouds_pct: int = 30, temp_f: float = 77.0) -> dict:
    """Estimates current community solar production using live solar irradiance data.

    Args:
        clouds_pct: Current cloud cover percentage (0-100). Get this from get_weather first.
        temp_f: Current temperature in Fahrenheit. Get this from get_weather first.

    Returns:
        Dictionary with production_kw, capacity_kw, efficiency_pct, ghi_wm2, temp_derate_pct, source.
    """
    # Clamp inputs to valid ranges
    clouds_pct = max(0, min(100, int(clouds_pct)))
    temp_f = max(-40, min(130, float(temp_f)))

    # System losses: inverter 97% × wiring 98% × soiling 97% × mismatch 98% ≈ 0.85
    SYSTEM_EFF = 0.85

    # Temperature derating: silicon panels lose ~0.4%/°F above 77°F (25°C)
    temp_derate = max(0.75, 1.0 - 0.004 * max(0, temp_f - 77))

    ghi = _get_current_ghi()
    if ghi is not None:
        # GHI-based: satellite-measured irradiance already factors clouds, sun angle, season
        production = round(max(0, COMMUNITY_CAPACITY_KW * (ghi / 1000) * SYSTEM_EFF * temp_derate), 1)
        return {
            "production_kw": production,
            "capacity_kw": COMMUNITY_CAPACITY_KW,
            "efficiency_pct": round(production / COMMUNITY_CAPACITY_KW * 100, 1),
            "ghi_wm2": round(ghi, 1),
            "temp_derate_pct": round(temp_derate * 100, 1),
            "source": "open-meteo",
        }

    # Fallback: cloud%-based estimate (less accurate — no seasonal sun angle)
    efficiency = max(0.15, 0.85 - (clouds_pct / 100) * 0.70)
    hour = datetime.now().hour
    time_factor = max(0, 1 - ((hour - 12) / 6) ** 2) if 6 <= hour <= 18 else 0
    production = round(COMMUNITY_CAPACITY_KW * efficiency * time_factor * temp_derate, 1)
    return {
        "production_kw": production,
        "capacity_kw": COMMUNITY_CAPACITY_KW,
        "efficiency_pct": round(production / COMMUNITY_CAPACITY_KW * 100, 1),
        "temp_derate_pct": round(temp_derate * 100, 1),
        "source": "fallback",
    }


class _BatterySimulator:
    """Maintains consistent battery SOC across tool calls within a session."""

    def __init__(self, capacity_kwh=BATTERY_CAPACITY_KWH):
        self.capacity = capacity_kwh
        self.soc = round(random.uniform(55, 85), 1)  # randomize once at session start

    def get_state(self):
        kwh = round(self.soc / 100 * self.capacity)
        return {
            "soc_pct": self.soc,
            "kwh_stored": kwh,
            "capacity_kwh": self.capacity,
            "charging": self.soc < 50,
        }

_battery = _BatterySimulator()


def get_battery_state() -> dict:
    """Gets the current state of the community shared battery storage.

    Returns:
        Dictionary with soc_pct (state of charge), kwh stored, capacity_kwh, charging status.
    """
    return _battery.get_state()


_EIA_RESPONDENT = {"MISO": "MISO", "CAISO": "CISO"}
_FALLBACK_GRID = {
    "MISO": {"renewable_pct": 12.5, "co2_intensity": 520},
    "CAISO": {"renewable_pct": 38.0, "co2_intensity": 280},
}


def _fetch_eia_grid_mix(region="MISO"):
    """Fetch current grid mix from EIA API v2. Returns (renewable_pct, co2_intensity) or fallback."""
    eia_code = _EIA_RESPONDENT.get(region, region)
    try:
        end = datetime.now(timezone.utc) - timedelta(days=1)
        start = end - timedelta(days=1)
        r = requests.get(
            "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/",
            params={
                "api_key": EIA_API_KEY,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": eia_code,
                "start": start.strftime("%Y-%m-%dT%H"),
                "end": end.strftime("%Y-%m-%dT%H"),
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "length": 200,
            },
            timeout=15,
        ).json()
        rows = r.get("response", {}).get("data", [])
        if not rows:
            fb = _FALLBACK_GRID.get(region, _FALLBACK_GRID["MISO"])
            return fb["renewable_pct"], fb["co2_intensity"]
        latest_period = rows[0].get("period")
        latest = [row for row in rows if row.get("period") == latest_period]
        total_mw, renewable_mw = 0, 0
        _RENEWABLE = {"SUN", "WND", "WAT", "GEO"}
        _FOSSIL_CO2 = {"COL": 1000, "NG": 450, "PET": 900, "OTH": 500}
        co2_total = 0
        for row in latest:
            mw = float(row.get("value") or 0)
            fuel = row.get("fueltype", "")
            total_mw += mw
            if fuel in _RENEWABLE:
                renewable_mw += mw
            co2_total += mw * _FOSSIL_CO2.get(fuel, 0)
        if total_mw > 0:
            renewable_pct = min(100.0, round(renewable_mw / total_mw * 100, 1))
            co2_intensity = max(0, round(co2_total / total_mw, 1))
            return renewable_pct, co2_intensity
    except Exception:
        pass
    fb = _FALLBACK_GRID.get(region, _FALLBACK_GRID["MISO"])
    return fb["renewable_pct"], fb["co2_intensity"]


def get_grid_status() -> dict:
    """Gets current electricity grid pricing period, rate, and grid mix (renewable percentage, CO2 intensity).

    Returns:
        Dictionary with period (peak/mid-peak/off-peak), rate_per_kwh in USD,
        renewable_pct, and co2_intensity (kg CO2/MWh).
    """
    hour = datetime.now().hour
    if 14 <= hour < 19:                            # 2pm-6:59pm
        period, rate = "peak", 0.28
    elif (7 <= hour < 14) or (19 <= hour < 23):    # 7am-1:59pm OR 7pm-10:59pm
        period, rate = "mid-peak", 0.18
    else:
        period, rate = "off-peak", 0.10            # 11pm-6:59am
    renewable_pct, co2_intensity = _fetch_eia_grid_mix("MISO")
    return {
        "period": period,
        "rate_per_kwh": rate,
        "renewable_pct": renewable_pct,
        "co2_intensity": co2_intensity,
    }


# Registry: maps function names to callables
TOOLS = [get_weather, get_solar_production, get_battery_state, get_grid_status]
TOOL_MAP = {fn.__name__: fn for fn in TOOLS}

# Quick test
for fn in TOOLS:
    print(f"  {fn.__name__}(): {fn()}")

"""## 4: Agentic Loop — Native Tool-Use Protocol"""

# === CELL 4: Agentic Loop — Native Tool-Use Protocol =========================

# System prompt is repeated twice — prompt repetition improves instruction
# following in causal LLMs by allowing each token to attend to every other
# prompt token, winning 47/70 benchmark-model tests with zero losses and no
# latency increase. See: Leviathan, Kalman & Matias (2024), "Repeat to
# Improve Non-Reasoning LLMs", Google Research. https://arxiv.org/abs/2512.14982
_UNIFIED_SYSTEM_BODY = (
    "You are SolarHive, an AI energy advisor for a community of 12 homes "
    "with rooftop solar and shared battery storage in Ann Arbor, Michigan. "
    "Provide specific, data-grounded advice on solar production, energy storage, "
    "grid coordination, and panel maintenance. When the user asks about current "
    "conditions (production, weather, battery state, grid status), call the "
    "available tools to retrieve real-time data before answering. For general "
    "guidance, scenario planning, or domain knowledge, answer directly. "
    "Be specific, reference actual data, and keep responses concise (3-5 sentences)."
)
SYSTEM_PROMPT = _UNIFIED_SYSTEM_BODY + "\n\n" + _UNIFIED_SYSTEM_BODY

def generate_with_tools(messages, tools=TOOLS, max_tokens=1024, max_rounds=3, thinking=False):
    """
    Gemma 4 native agentic loop.

    1. Send messages + tool schemas to the model
    2. If model emits tool_calls, execute them and feed results back
    3. Repeat until model produces a final text response (no more tool calls)

    Args:
        thinking: Enable thinking mode for complex reasoning (slower). Default False.
    """
    all_calls = []

    for round_num in range(max_rounds):
        # Two-step approach: render text first, then tokenize separately.
        # Single-step (tokenize=True) triggers a visual scanning path in
        # transformers 5.5.x that breaks on messages without "content" key.
        # The two-step approach matches the working smoke test pattern.

        # Extract images from messages for the processor
        _images = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        _images.append(item["image"])

        text = processor.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            enable_thinking=thinking,
            tokenize=False,
        )
        if _images:
            inputs = processor(text=text, images=_images, return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=1.0, top_p=0.95, top_k=64,
            )

        raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        # Detect tool calls via Gemma 4's control token pattern
        tool_call_pattern = r'call:(\w+)\{([^}]*)\}'
        found = re.findall(tool_call_pattern, raw)

        if not found:
            # No tool calls — final answer.
            # processor.parse_response() strips thinking tokens and returns the clean text.
            # Official Kaggle card: processor.parse_response(response) — called directly.
            # HF blog shows it may return dict {"content":..., "thinking":...}.
            # Safe: handle both string and dict return types.
            parsed = processor.parse_response(raw)
            clean = parsed["content"] if isinstance(parsed, dict) else parsed
            return {"response": clean, "tool_calls": all_calls, "rounds": round_num + 1}

        # Parse and execute each tool call
        calls, results = [], []
        for fn_name, args_str in found:
            args = {}
            for key, str_val, num_val in re.findall(
                r'(\w+):\s*(?:<\|"\|>([^<]*)<\|"\|>|(\d+\.?\d*))', args_str
            ):
                args[key] = str_val if str_val else (
                    float(num_val) if '.' in num_val else int(num_val)
                )

            call = {"name": fn_name, "arguments": args}
            calls.append(call)
            all_calls.append(call)

            result = TOOL_MAP[fn_name](**args) if fn_name in TOOL_MAP else {"error": f"Unknown: {fn_name}"}
            results.append({"name": fn_name, "response": result})

        # Feed results back — match finetune/datagen training format exactly:
        # 1) assistant message with tool_calls only
        # 2) one role=tool message per tool result
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

    return {"response": "[Agent exceeded max rounds]", "tool_calls": all_calls, "rounds": max_rounds}

"""## 5: SolarHive Agent (text + optional image)"""

def solarhive_agent(question, image=None, thinking=False):
    """
    Full SolarHive agent using native tool-use protocol.
    Model decides which tools to call based on the question.
    Optionally accepts an image for VQA + tool calling in one turn.

    Args:
        thinking: Enable thinking mode for complex reasoning (slower).
    """
    content = []
    if image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})

    # When image is provided, tell the model to describe what it SEES first.
    # Without this, the model calls get_weather() and bases its answer entirely
    # on live API data, ignoring the visual content (all images get same answer).
    sys_prompt = SYSTEM_PROMPT
    if image:
        _vqa_inst = (
            " When an image is provided, FIRST describe what you observe in the "
            "image (e.g., cloud cover, sky color, panel condition). Base your "
            "primary assessment on visual observation. You may call tools for "
            "additional context, but note any differences between what the image "
            "shows and what the station data reports."
        )
        sys_prompt += _vqa_inst + _vqa_inst

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content if image else question},
    ]
    return generate_with_tools(messages, thinking=thinking)

"""## 6: VQA — Sky Analysis (Hero Feature)"""

def analyze_sky(image, question="How will this sky affect our solar production?"):
    """VQA Mode 1: Sky photo + native tool calling in one agentic turn."""
    return solarhive_agent(
        f"Look at this sky photo. Describe the cloud cover and sky conditions "
        f"you observe in the image. Then answer: {question} "
        "Give: 1) Visual sky assessment (what the photo shows — estimated cloud %) "
        "2) Weather station comparison (if different from what you see) "
        "3) Production impact 4) Recommendation.",
        image=image,
    )

"""## 7: VQA — Panel Inspection"""

def inspect_panels(image, home_id="Home #7", delta="-15%"):
    """VQA Mode 2: Panel photo assessment with data grounding."""
    return solarhive_agent(
        f"Photo of panels from {home_id} (production {delta} vs neighborhood avg). "
        "Assess: 1) Visible issues 2) Estimated efficiency loss "
        "3) Does this explain the gap? 4) Actions.",
        image=image,
    )

"""## 7b: VQA — Neighborhood Assessment"""

def assess_neighborhood(image, context="12-home community, Ann Arbor, MI"):
    """VQA Mode 3: Aerial image → community solar expansion assessment."""
    return solarhive_agent(
        f"Aerial image of {context}. Identify: "
        "1) Which roofs appear to have solar panels installed "
        "2) Which south/southwest-facing roofs have no panels and their estimated kW potential "
        "3) Any visible shading issues from trees or buildings "
        "4) Priority recommendation for expanding community capacity.",
        image=image,
    )

"""## 8: Demo — Native Agentic Behavior"""

print("=" * 60)
print("SolarHive Agent — Native Tool-Use Demo")
print("=" * 60)

tests = [
    ("What time does peak pricing start?", None),
    ("Should I run my pool heater now or wait?", None),
    ("What does our solar production look like based on this sky?",
     Image.new("RGB", (640, 480), (135, 206, 235))),
    ("Quick community energy status update.", None),
    # Multi-tool chain: should trigger weather → solar → battery → grid
    ("Full community energy audit — check weather, solar production, battery state, "
     "and grid pricing. Give a complete status report with recommendations.", None),
]

for q, img in tests:
    tag = "📸" if img else "📝"
    print(f"\n{tag} Q: {q}")
    r = solarhive_agent(q, img)
    called = [c["name"] for c in r.get("tool_calls", [])]
    print(f"   Tools called: {called or 'none (direct answer)'}")
    print(f"   Rounds: {r['rounds']}")
    print(f"   A: {r['response']}")
    print("-" * 60)


"""## 9: VQA Test — Sky Analysis with Real Photos"""

# ─── HOW TO USE REAL PHOTOS ─────────────────────────────────────────────
# Option A: Colab file browser → click folder icon (left sidebar) → Upload
# Option B: Programmatic upload:
#   from google.colab import files
#   uploaded = files.upload()  # opens file picker dialog
#   img = Image.open(list(uploaded.keys())[0])
# Option C: From Google Drive (after mounting in Cell 1):
#   img = Image.open("/content/drive/MyDrive/photos/clear_sky.jpg")
#
# Then replace Image.new() below with your loaded image.
# ─────────────────────────────────────────────────────────────────────────

_sky_tests = [
    ("clear sky", Image.new("RGB", (640, 480), (70, 130, 230)),
     "How will this clear sky affect our solar production today?"),
    ("partly cloudy", Image.new("RGB", (640, 480), (180, 190, 200)),
     "There are some clouds moving in. What's the production outlook?"),
    ("overcast", Image.new("RGB", (640, 480), (140, 140, 145)),
     "Heavy overcast sky — should we adjust our energy strategy?"),
]

print("=" * 60)
print("VQA Test: Sky Analysis (Task 1.7)")
print("=" * 60)

for label, img, question in _sky_tests:
    print(f"\n🌤️ Test: {label}")
    print(f"   Q: {question}")
    r = analyze_sky(img, question)
    called = [c["name"] for c in r.get("tool_calls", [])]
    print(f"   Tools: {called or 'none'}")
    print(f"   Rounds: {r['rounds']}")
    print(f"   A: {r['response'][:600]}")
    print("-" * 60)

print("\n✅ Sky VQA test complete")
print("   Replace synthetic images with real photos for full validation.")


"""## 10: VQA Test — Panel Inspection"""

# Upload a real panel photo, or use this synthetic test image.
# For the real demo: use a photo of solar panels (clean, dirty, or damaged).

_panel_img = Image.new("RGB", (640, 480), (40, 60, 90))  # dark blue placeholder

print("=" * 60)
print("VQA Test: Panel Inspection (Task 1.8)")
print("=" * 60)

_panel_tests = [
    ("Home #7", "-15%", "This home's output has been 15% below neighbors for 3 weeks."),
    ("Home #3", "-22%", "Significant underperformance — possible shading or soiling issue."),
]

for home_id, delta, context in _panel_tests:
    print(f"\n🔍 Test: {home_id} (delta {delta})")
    r = inspect_panels(_panel_img, home_id=home_id, delta=delta)
    called = [c["name"] for c in r.get("tool_calls", [])]
    print(f"   Tools: {called or 'none'}")
    print(f"   Rounds: {r['rounds']}")
    print(f"   A: {r['response'][:600]}")
    print("-" * 60)

print("\n✅ Panel inspection VQA test complete")
print("   Replace synthetic images with real panel photos for full validation.")

# --- Neighborhood VQA test (Mode 3) ---
_aerial_img = Image.new("RGB", (640, 480), (90, 120, 60))  # green-ish aerial placeholder

print("\n" + "=" * 60)
print("VQA Test: Neighborhood Assessment (Task 3.6)")
print("=" * 60)

print("\n🏘️ Test: Aerial neighborhood assessment")
r = assess_neighborhood(_aerial_img)
called = [c["name"] for c in r.get("tool_calls", [])]
print(f"   Tools: {called or 'none'}")
print(f"   Rounds: {r['rounds']}")
print(f"   A: {r['response'][:600]}")
print("-" * 60)
print("\n✅ Neighborhood VQA test complete")

"""## 11: Benchmark — Held-Out Evaluation"""

# Same held-out questions as finetune Cell 6b — measures domain accuracy.
# Adapted for inference's two-step tokenization + processor pattern.

BENCHMARK_QS = [
    "What happens to solar production when humidity exceeds 80%?",
    "At what battery SOC should we stop exporting to the grid?",
    "Home #3 has been underperforming by 22% for three weeks. What's the diagnostic checklist?",
    "It's winter in Ann Arbor and panels have snow. Prioritize actions.",
    "Grid frequency dropped to 59.8 Hz. What does that mean for our microgrid?",
]

TOOL_BENCHMARK_QS = [
    ("What's the current battery state?", "get_battery_state"),
    ("What's the current weather in Ann Arbor and how does it affect solar production?", "get_weather"),
    ("What are the general maintenance tips for panels?", None),  # should NOT call a tool
]


def _run_benchmark():
    """Generate answers for held-out Q&A benchmark questions."""
    results = []
    for q in BENCHMARK_QS:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=1024,
                temperature=1.0, top_p=0.95, top_k=64,
            )
        raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        parsed = processor.parse_response(raw)
        answer = parsed["content"] if isinstance(parsed, dict) else parsed
        results.append(answer)
    return results


def _run_tool_benchmark():
    """Test tool-calling behavior on held-out questions."""
    results = []
    for q, expected_tool in TOOL_BENCHMARK_QS:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = processor.apply_chat_template(
            msgs, tools=TOOLS, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=1024,
                temperature=1.0, top_p=0.95, top_k=64,
            )
        raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        calls = re.findall(r'call:(\w+)\{([^}]*)\}', raw)
        called_tools = [c[0] for c in calls] if calls else []
        results.append((q, expected_tool, called_tools, raw))
    return results


print("=" * 60)
print("Benchmark: Held-Out Evaluation")
print("=" * 60)

# Q&A Benchmark
print("\n--- Q&A Benchmark (5 questions) ---")
_qa_results = _run_benchmark()
for i, (q, a) in enumerate(zip(BENCHMARK_QS, _qa_results)):
    print(f"\n  Q{i+1}: {q}")
    print(f"  A: {a[:300]}")

# Tool-Calling Benchmark
print("\n--- Tool-Calling Benchmark (3 questions) ---")
_tool_results = _run_tool_benchmark()
_tool_correct = 0
for q, expected, called, raw in _tool_results:
    if expected is None:
        match = len(called) == 0
    else:
        match = expected in called
    _tool_correct += int(match)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] Q: {q}")
    print(f"         Expected: {expected or 'no tool call'} | Got: {called or 'no tool call'}")

print(f"\n  Tool accuracy: {_tool_correct}/{len(TOOL_BENCHMARK_QS)}")
print("\n✅ Benchmark complete")

"""## 11b: When2Call-Style Held-Out Probes

Three probes added per `when2call_plan.md` Task W6, validating coverage of the
4-way taxonomy from Ross, H., Mahabaleshwarkar, A. S., & Suhara, Y. (2025).
*When2Call: When (not) to Call Tools.* arXiv:2504.18851.
URL: https://arxiv.org/abs/2504.18851

These probes check the failure modes the paper documents in untrained community
models (9–67% tool-hallucination rates):

  (d) Out-of-scope query → expect refusal + redirect (no tool call, names limit)
  (c) Under-specified query → expect follow-up question (no tool call, asks back)
  (b) Well-specified in-scope query → expect correct tool call

Pre-W1+W2: v1 model fails (d) + (c) by hallucinating tools or auto-filling defaults.
Post-D5 (v2 model trained with `_UNABLE_TO_ANSWER` + `_FOLLOW_UP_QUESTIONS`):
expected 3/3 + zero regression on Run-6 8/8.
"""

# === CELL 11b: When2Call probes ================================================

WHEN2CALL_PROBES = [
    {
        "category": "(d) Unable to answer",
        "question": "What's the current air quality index in Ann Arbor?",
        "expected_tool": None,
        "must_contain_any": ["don't have", "do not have", "no tool", "AQI", "airnow", "air-quality", "air quality"],
        "rationale": "No AQI tool — model should name the limit and redirect (e.g., airnow.gov)",
    },
    {
        "category": "(c) Follow-up question",
        "question": "How much will a 10 kW array produce today?",
        "expected_tool": None,
        "must_contain_any": ["which", "what", "where", "location", "city", "?"],
        "rationale": "Location parameter missing — model should ask back, not auto-fill Ann Arbor",
    },
    {
        "category": "(b) Tool call",
        "question": "What's the current grid rate?",
        "expected_tool": "get_grid_status",
        "must_contain_any": None,
        "rationale": "Well-specified in-scope query — confirms no over-correction toward conservatism",
    },
]


def _run_when2call_probes():
    """Run the 3 When2Call probes against the loaded model."""
    results = []
    for probe in WHEN2CALL_PROBES:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": probe["question"]},
        ]
        text = processor.apply_chat_template(
            msgs, tools=TOOLS, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512,
                temperature=1.0, top_p=0.95, top_k=64,
            )
        raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        calls = re.findall(r'call:(\w+)\{([^}]*)\}', raw)
        called_tools = [c[0] for c in calls] if calls else []
        parsed = processor.parse_response(raw)
        text_response = parsed["content"] if isinstance(parsed, dict) else parsed

        # Tool-call check
        if probe["expected_tool"] is None:
            tool_match = len(called_tools) == 0
        else:
            tool_match = probe["expected_tool"] in called_tools

        # Content check (if specified)
        if probe["must_contain_any"] is None:
            content_match = True
        else:
            text_lower = (text_response or "").lower()
            content_match = any(kw.lower() in text_lower for kw in probe["must_contain_any"])

        passed = tool_match and content_match
        results.append({
            "category": probe["category"],
            "question": probe["question"],
            "expected_tool": probe["expected_tool"],
            "called_tools": called_tools,
            "response": text_response,
            "tool_match": tool_match,
            "content_match": content_match,
            "passed": passed,
            "rationale": probe["rationale"],
        })
    return results


print("\n" + "=" * 60)
print("When2Call Probes (Ross et al. 2025, arXiv:2504.18851)")
print("=" * 60)

_w2c_results = _run_when2call_probes()
_w2c_correct = sum(1 for r in _w2c_results if r["passed"])

for r in _w2c_results:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n  [{status}] {r['category']}")
    print(f"         Q: {r['question']}")
    print(f"         Expected tool: {r['expected_tool'] or 'no tool call'} | Got: {r['called_tools'] or 'no tool call'}")
    print(f"         Tool match: {r['tool_match']} | Content match: {r['content_match']}")
    print(f"         Rationale: {r['rationale']}")
    if r["response"]:
        print(f"         Response: {r['response'][:200]}")

print(f"\n  When2Call accuracy: {_w2c_correct}/{len(WHEN2CALL_PROBES)}")
print("\n✅ When2Call probes complete")

"""## 12: Interactive Demo (for video capture)"""

# Simple input loop for live demo / video recording.
# Type a question, optionally upload an image, get full agent trace.

print("=" * 60)
print("SolarHive Interactive Demo")
print("Type 'quit' to exit. Upload image when prompted (or press Enter to skip).")
print("=" * 60)

for _demo_i in range(5):  # safety limit: 5 queries max
    print(f"\n--- Query {_demo_i + 1}/5 ---")
    try:
        _q = input("Question: ").strip()
    except EOFError:
        break
    if not _q or _q.lower() in ("quit", "exit", "q"):
        break

    # Optional image upload
    _demo_img = None
    try:
        _upload = input("Upload image? (y/N): ").strip().lower()
        if _upload == "y":
            from google.colab import files as _colab_files
            _uploaded = _colab_files.upload()
            if _uploaded:
                _fname = list(_uploaded.keys())[0]
                _demo_img = Image.open(_fname)
                print(f"   Loaded: {_fname}")
    except (EOFError, ImportError):
        pass

    print("   Running agent...")
    _r = solarhive_agent(_q, _demo_img)
    _called = [c["name"] for c in _r.get("tool_calls", [])]
    print(f"   Tools called: {_called or 'none (direct answer)'}")
    print(f"   Rounds: {_r['rounds']}")
    print(f"   Answer: {_r['response']}")
    print("-" * 60)

print("\n✅ Interactive demo complete")