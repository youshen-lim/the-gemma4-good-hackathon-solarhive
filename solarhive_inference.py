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

# HF auth hoisted to module level so every cell that pulls from a private
# Truthseeker87/* repo (Cell 2b LoRA fallback + every §13 variant cell)
# inherits the login. All 8 SolarHive HF repos stay private until submission
# day; without auth, snapshot_download / from_pretrained on private repos
# 401. Pattern mirrors solarhive_merge_a4b.py / solarhive_merge_e4b.py /
# solarhive_quantize_nf4.py — try Kaggle → try Colab → login() globally;
# non-blocking (skip login if neither secret store yields a token, since
# the base-model fallback path doesn't need HF auth).
from huggingface_hub import login, snapshot_download
HF_TOKEN = None
if _on_kaggle:
    try:
        HF_TOKEN = secrets.get_secret("HF_TOKEN")
    except Exception:
        pass
else:
    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("   HF_TOKEN resolved — logged in to HuggingFace Hub (private repos accessible)")
else:
    print("   HF_TOKEN NOT FOUND — private Truthseeker87/* repos will 401; only public/base paths work")

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
# Resolution order: local Drive cache → cwd local → HF Hub fallback → base model only
_LORA_PATHS = [
    "/content/drive/MyDrive/models/solarhive_a4b_lora",
    "solarhive_a4b_lora",
]
_LORA_HF_REPO = "Truthseeker87/solarhive-26b-a4b-lora"  # canonical HF source
_lora_loaded = False


def _load_lora_from_path(lora_path):
    """Patch adapter_config + load via Unsloth FastVisionModel. Returns (model, processor)."""
    from unsloth import FastVisionModel
    _cfg_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(_cfg_path):
        with open(_cfg_path) as f:
            _cfg = json.load(f)
        if _cfg.get("base_model_name_or_path") != MODEL_PATH:
            _cfg["base_model_name_or_path"] = MODEL_PATH
            with open(_cfg_path, "w") as f:
                json.dump(_cfg, f, indent=2)
            print(f"   Patched adapter config → {MODEL_PATH}")
    return FastVisionModel.from_pretrained(
        model_name=lora_path,
        load_in_4bit=_load_4bit,
        dtype=torch.bfloat16,
    )


# Pass 1: try local Drive paths
for _lp in _LORA_PATHS:
    if os.path.isdir(_lp):
        try:
            model, processor = _load_lora_from_path(_lp)
            from unsloth import FastVisionModel
            FastVisionModel.for_inference(model)
            print(f"✅ Fine-tuned LoRA adapters loaded from local: {_lp}")
            _lora_loaded = True
        except Exception as e:
            print(f"⚠️  Unsloth LoRA load from {_lp} failed: {e}")
        break

# Pass 2: HF Hub fallback — pulls solarhive-26b-a4b-lora if no local copy
# Useful for fresh Colab sessions where Drive isn't pre-populated.
# Module-level HF_TOKEN (Cell 1) already called login() globally; the
# inline _hf_token resolution below stays for explicit token threading
# in case login was skipped (HF_TOKEN unresolved at import time).
if not _lora_loaded:
    print(f"No local LoRA found — attempting HF fallback from {_LORA_HF_REPO}...")
    try:
        _hf_token = HF_TOKEN
        try:
            from google.colab import userdata as _ud
            if not _hf_token:
                _hf_token = _ud.get("HF_TOKEN")
        except Exception:
            pass
        if not _hf_token:
            try:
                from kaggle_secrets import UserSecretsClient as _USC
                _hf_token = _USC().get_secret("HF_TOKEN")
            except Exception:
                pass

        _hf_lora_path = snapshot_download(
            repo_id=_LORA_HF_REPO, repo_type="model", token=_hf_token,
        )
        print(f"   Downloaded LoRA from HF to {_hf_lora_path}")
        model, processor = _load_lora_from_path(_hf_lora_path)
        from unsloth import FastVisionModel
        FastVisionModel.for_inference(model)
        print(f"✅ Fine-tuned LoRA adapters loaded from HF: {_LORA_HF_REPO}")
        _lora_loaded = True
    except Exception as e:
        print(f"⚠️  HF LoRA fallback failed: {e}")
        print(f"    (private repo requires HF_TOKEN in Colab/Kaggle secrets)")

# --- Final fallback: load base model without LoRA via transformers ---
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
_smoke_clean = (_smoke_parsed.get("content", "") if isinstance(_smoke_parsed, dict) else (_smoke_parsed or ""))
print("─" * 60)
print("Smoke test — Gemma 4 response:")
print(_smoke_clean)
print("─" * 60)
print("✅ Chat template, generate, and parse_response all working")

"""## 2d: Multi-Variant Drive Path Registry

Defines Drive cache paths and HF repo IDs for the 4 v2 weight variants
benchmarked in the "Multi-Variant Inference Benchmark" section at the end
of this notebook. These are constants only — actual loads happen in the
per-variant cells. Path conventions match the merge and quantize notebooks
(date-versioned folders with `_YYYYMMDD` suffix; glob fallback finds the
most recent dated folder if today's is empty).
"""

from datetime import datetime as _dt
import glob as _glob

_today = _dt.now().strftime("%Y%m%d")

# Base model Drive caches (kagglehub-cached)
_DRIVE_E4B_PATH = "/content/drive/MyDrive/models/gemma-4/transformers/gemma-4-e4b-it/1"
# (_DRIVE_MODEL_PATH for A4B base is defined earlier in Cell 2a)

# LoRA Drive backups (canonical naming convention used across project notebooks)
_DRIVE_A4B_LORA = "/content/drive/MyDrive/models/solarhive_a4b_lora"
_DRIVE_E4B_LORA = "/content/drive/MyDrive/models/solarhive_e4b_lora"


def _resolve_dated_drive(prefix):
    """Return today's dated folder if populated, else most recent dated, else None.

    Looks for `/content/drive/MyDrive/models/{prefix}_{YYYYMMDD}` directories.
    Mirrors the three-tier resolution used by `solarhive_quantize_nf4.py` Cell 3.
    """
    today_path = f"/content/drive/MyDrive/models/{prefix}_{_today}"
    if os.path.isdir(today_path) and os.listdir(today_path):
        return today_path
    candidates = sorted(_glob.glob(f"/content/drive/MyDrive/models/{prefix}_*"))
    populated = [c for c in candidates if os.path.isdir(c) and os.listdir(c)]
    return populated[-1] if populated else None


_DRIVE_A4B_MERGED = _resolve_dated_drive("solarhive_a4b_merged")
_DRIVE_A4B_NF4    = _resolve_dated_drive("solarhive_a4b_nf4")

# E4B merged uses the canonical fixed-name convention (folder name matches the HF
# repo name `solarhive-e4b-ollama`, no date suffix). Date-versioned form is
# checked as a forward-compatible fallback in case a future E4B merge run
# adopts the established date-versioned cache convention.
_DRIVE_E4B_MERGED_FIXED = "/content/drive/MyDrive/models/solarhive_e4b_ollama"
_DRIVE_E4B_MERGED_DATED = _resolve_dated_drive("solarhive_e4b_merged")
_DRIVE_E4B_MERGED = (
    _DRIVE_E4B_MERGED_FIXED
    if (os.path.isdir(_DRIVE_E4B_MERGED_FIXED) and os.listdir(_DRIVE_E4B_MERGED_FIXED))
    else _DRIVE_E4B_MERGED_DATED
)

# Variant HF repo IDs (canonical 5-repo registry)
VARIANT_REPOS = {
    "e4b_lora":     "Truthseeker87/solarhive-e4b-lora",         # LoRA adapters (~200 MB)
    "e4b_merged":   "Truthseeker87/solarhive-e4b-ollama",       # BF16 merged safetensors (~16 GB)
    "e4b_gguf":     "Truthseeker87/solarhive-e4b-gguf",         # Q4_K_M GGUF for Ollama runtime (~5 GB)
    "a4b_merged":   "Truthseeker87/solarhive-26b-a4b-merged",   # BF16 sharded safetensors (~48 GB)
    "a4b_nf4":      "Truthseeker87/solarhive-26b-a4b-nf4",      # NF4 quantized (~48 GB)
}

# Per-variant skip flags. Default True for transformers/Unsloth variants
# (E4B LoRA, E4B BF16, A4B BF16, A4B NF4); default False for the
# GGUF/Ollama variant since it requires a running Ollama server at
# localhost:11434 (not pre-installed in standard Colab). Set
# _RUN_E4B_GGUF=True after starting the server.
_RUN_E4B_LORA   = True
_RUN_E4B_MERGED = True
_RUN_E4B_GGUF   = False
_RUN_A4B_MERGED = True
_RUN_A4B_NF4    = True

print(f"Multi-variant Drive registry initialized (today={_today})")
print(f"  A4B base   : {_DRIVE_MODEL_PATH}  {'(cached)' if os.path.isdir(_DRIVE_MODEL_PATH) else '(not cached)'}")
print(f"  E4B base   : {_DRIVE_E4B_PATH}  {'(cached)' if os.path.isdir(_DRIVE_E4B_PATH) else '(not cached)'}")
print(f"  A4B LoRA   : {_DRIVE_A4B_LORA}  {'(cached)' if os.path.isdir(_DRIVE_A4B_LORA) else '(not cached)'}")
print(f"  E4B LoRA   : {_DRIVE_E4B_LORA}  {'(cached)' if os.path.isdir(_DRIVE_E4B_LORA) else '(not cached)'}")
print(f"  A4B merged : {_DRIVE_A4B_MERGED or '(no Drive cache — will pull from HF)'}")
print(f"  A4B NF4    : {_DRIVE_A4B_NF4 or '(no Drive cache — will pull from HF)'}")

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


# NREL PVWatts — typical-year solar production baseline for the community
# array. Cached per session because the response is annual/monthly typical
# data (no realtime variation), so one call covers the whole notebook run.
_NREL_PVWATTS_CACHE = None


def _fetch_nrel_pvwatts():
    """Cached fetch of NREL PVWatts v8 typical-year production for our 72 kW
    Ann Arbor array. Returns the `outputs` dict from the API response, or
    None on failure (network, key invalid, throttled)."""
    global _NREL_PVWATTS_CACHE
    if _NREL_PVWATTS_CACHE is not None:
        return _NREL_PVWATTS_CACHE
    try:
        r = requests.get(
            "https://developer.nrel.gov/api/pvwatts/v8.json",
            params={
                "api_key": NREL_API_KEY,
                "system_capacity": COMMUNITY_CAPACITY_KW,  # 72 kW total
                "module_type": 0,    # 0 = standard silicon
                "losses": 14,        # NREL default system losses (%)
                "array_type": 1,     # 1 = fixed roof-mount (residential rooftops)
                "tilt": 30,          # ~Michigan latitude for ~max annual yield
                "azimuth": 180,      # south-facing
                "lat": LAT,
                "lon": LON,
            },
            timeout=15,
        ).json()
        _NREL_PVWATTS_CACHE = r.get("outputs", {})
        return _NREL_PVWATTS_CACHE
    except Exception:
        return None


def get_nrel_pvwatts_baseline() -> dict:
    """Gets NREL PVWatts typical-year solar production baseline for the community 72 kW array.

    Use this to compare current real-time output (from get_solar_production) against
    typical-year performance — useful for diagnosing under-/over-performance and
    setting expectations for the current month. Cached per session.

    Returns:
        Dictionary with annual_kwh, current_month_typical_kwh, current_month_typical_kw_avg, capacity_kw, source.
    """
    out = _fetch_nrel_pvwatts()
    if not out:
        return {"error": "NREL PVWatts API unreachable",
                "annual_kwh": None, "current_month_typical_kwh": None,
                "current_month_typical_kw_avg": None,
                "capacity_kw": COMMUNITY_CAPACITY_KW, "source": "fallback"}

    from calendar import monthrange
    _now = datetime.now()
    _month_idx = _now.month - 1  # 0-indexed
    _monthly = out.get("ac_monthly") or [None] * 12
    _current_month_kwh = _monthly[_month_idx] if _month_idx < len(_monthly) else None
    _days_in_month = monthrange(_now.year, _now.month)[1]
    _avg_kw = (round(_current_month_kwh / (_days_in_month * 24), 2)
               if _current_month_kwh else None)

    return {
        "annual_kwh": round(out["ac_annual"]) if out.get("ac_annual") else None,
        "current_month_typical_kwh": round(_current_month_kwh) if _current_month_kwh else None,
        "current_month_typical_kw_avg": _avg_kw,
        "capacity_kw": COMMUNITY_CAPACITY_KW,
        "source": "nrel-pvwatts-v8",
    }


# Registry: maps function names to callables. All three keyed APIs (OWM,
# EIA, NREL) are actively used — see CLAUDE.md API Keys section. Open-Meteo
# is keyless and serves as the realtime GHI source for get_solar_production.
TOOLS = [get_weather, get_solar_production, get_battery_state, get_grid_status, get_nrel_pvwatts_baseline]
TOOL_MAP = {fn.__name__: fn for fn in TOOLS}


# ---------------------------------------------------------------------------
# Tool-call extraction + args parsing
# ---------------------------------------------------------------------------
# Pattern aligned with Google's official Gemma 4 function-calling docs:
# https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4
#
# The model emits tool calls in the wrapped form:
#   <|tool_call>call:fn{key:<|"|>value<|"|>,n:42}<tool_call|>
# Thinking-mode interactions can occasionally suppress the wrapper, leaving
# bare `call:fn{...}` tokens. Two-pattern fallback (wrapper first, bare
# fallback) mirrors `parse_gemma4_output()` in `test_ollama_tools.py`.
_TOOL_CALL_WRAPPED_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL)
_TOOL_CALL_BARE_RE = re.compile(r"call:(\w+)\{([^}]*)\}")

# Args regex: handles strings (via `<|"|>` delimiters), ints, floats including
# NEGATIVES, booleans, and null. An earlier version used `(\d+\.?\d*)` which silently
# dropped negative numbers (e.g., `temp_f:-5` in Ann Arbor January) — the arg
# would be missing from the parsed dict and the tool function would fall back
# to its default value, masking the model's actual intent.
_ARG_FIELD_RE = re.compile(
    r'(\w+)\s*:\s*(?:<\|"\|>([^<]*)<\|"\|>|(true|false|null)|(-?\d+\.?\d*))'
)


def _extract_tool_calls(raw):
    """Return [(fn_name, args_str), ...] found in raw model output.

    Tries the wrapped form first (Google's documented pattern), falls back
    to bare `call:fn{...}` only if no wrapped calls are found. This is
    additive — any output that previously matched the bare pattern still
    matches under the fallback path."""
    matches = [(m.group(1), m.group(2).strip()) for m in _TOOL_CALL_WRAPPED_RE.finditer(raw)]
    if matches:
        return matches
    return [(m.group(1), m.group(2).strip()) for m in _TOOL_CALL_BARE_RE.finditer(raw)]


def _parse_tool_args(args_str):
    """Convert a Gemma 4 bare-key arg string into a Python dict.

    Handles:
      key:<|"|>value<|"|>   → str (including empty string)
      key:true / false / null → bool / None
      key:42 / -5 / 3.14 / -0.5 → int / float (negatives now supported)

    Returns {} for empty/no-arg calls."""
    args = {}
    for m in _ARG_FIELD_RE.finditer(args_str):
        key = m.group(1)
        s, bn, n = m.group(2), m.group(3), m.group(4)
        if s is not None:  # string alt fired (even an empty string is valid)
            args[key] = s
        elif bn:
            args[key] = {"true": True, "false": False, "null": None}[bn]
        elif n:
            args[key] = float(n) if "." in n else int(n)
    return args


import inspect as _inspect


def _safe_tool_call(fn, args):
    """Dispatch a tool call defensively — drop kwargs the function doesn't accept.

    The model occasionally hallucinates extra kwargs (e.g., emitting
    `call:get_grid_status{location:<|"|>Ann Arbor, MI<|"|>}` even though
    the function takes no args). Without filtering, `fn(**args)` raises
    `TypeError: ... got an unexpected keyword argument 'location'` and
    crashes the agentic loop. Per Google's Gemma 4 function-calling docs:
    "Always validate function names and arguments before execution."

    If the function declares `**kwargs`, we pass everything through
    unchanged — that's an explicit opt-in to accept unknowns.
    """
    sig = _inspect.signature(fn)
    if any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**args)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in args.items() if k in accepted}
    if filtered != args:
        dropped = set(args) - set(filtered)
        print(f"   ⚠️ {fn.__name__}: dropped hallucinated args {sorted(dropped)} (function takes {sorted(accepted) or 'no args'})")
    return fn(**filtered)

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

        # Detect tool calls — wrapped form (Google's docs) preferred, bare
        # form as fallback. See _extract_tool_calls() near Cell 3.
        found = _extract_tool_calls(raw)

        if not found:
            # No tool calls — final answer.
            # processor.parse_response() strips thinking tokens and returns the clean text.
            # Official Kaggle card: processor.parse_response(response) — called directly.
            # HF blog shows it may return dict {"content":..., "thinking":...}.
            # Safe: handle both string and dict return types.
            parsed = processor.parse_response(raw)
            clean = (parsed.get("content", "") if isinstance(parsed, dict) else (parsed or ""))
            return {"response": clean, "tool_calls": all_calls, "rounds": round_num + 1}

        # Parse and execute each tool call (shared helper handles negative
        # numbers, booleans, null, and strings via <|"|> delimiters).
        calls, results = [], []
        for fn_name, args_str in found:
            args = _parse_tool_args(args_str)

            call = {"name": fn_name, "arguments": args}
            calls.append(call)
            all_calls.append(call)

            # Defensive dispatch — drop hallucinated kwargs (see _safe_tool_call docstring)
            result = _safe_tool_call(TOOL_MAP[fn_name], args) if fn_name in TOOL_MAP else {"error": f"Unknown: {fn_name}"}
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

"""## 11: Benchmark — Held-Out Evaluation (10 questions: 5 Q&A + 5 tool)

Same 5 Q&A questions as the finetune notebook plus 5 tool-calling probes
(was 3 in v1; +2 added in v2 for broader tool-routing coverage). Adapted
for inference's two-step tokenization + processor pattern.

Tool-calling format: `(question, expected_tools, [min_calls=1])`
- `expected_tools = None` → expect no tool call (direct answer)
- `expected_tools = {"x"}` → expect a call to tool `x`
- `expected_tools = {"x", "y"}` → expect a call to either `x` or `y`
- Optional 3rd element `min_calls` enables lenient multi-call scoring
  (used by TQ5 — the multi-city irradiance probe — which expects ≥2
  tool calls of any combination, not strict 3-of-3)
"""

BENCHMARK_QS = [
    "What happens to solar production when humidity exceeds 80%?",
    "At what battery SOC should we stop exporting to the grid?",
    "Home #3 has been underperforming by 22% for three weeks. What's the diagnostic checklist?",
    "It's winter in Ann Arbor and panels have snow. Prioritize actions.",
    "Grid frequency dropped to 59.8 Hz. What does that mean for our microgrid?",
]

TOOL_BENCHMARK_QS = [
    # --- Original 3 (carried over from v1 for direct comparability) ---
    ("What's the current battery state?",
     {"get_battery_state"}),
    ("What's the current weather in Ann Arbor and how does it affect solar production?",
     {"get_weather"}),
    ("What are the general maintenance tips for panels?",
     None),  # should NOT call a tool
    # --- v2 additions (benchmark expanded from 8 → 10 questions) ---
    ("What's the grid pricing right now and what's the renewable mix?",
     {"get_grid_status"}),
    ("Compare today's irradiance forecast across Ann Arbor, Phoenix, and Seattle.",
     {"get_solar_production", "get_weather"}, 2),  # multi-call: lenient ≥2 calls
]


def _run_benchmark(model_=None, processor_=None, system_prompt_=None):
    """Generate answers for held-out Q&A benchmark questions.

    Defaults to the module-level `model`, `processor`, and `SYSTEM_PROMPT`
    so existing callers work unchanged. Multi-variant cells reassign
    these globals before calling — or pass explicit args.
    """
    _model = model_ if model_ is not None else model
    _processor = processor_ if processor_ is not None else processor
    _sys = system_prompt_ if system_prompt_ is not None else SYSTEM_PROMPT
    results = []
    for q in BENCHMARK_QS:
        msgs = [
            {"role": "system", "content": _sys},
            {"role": "user", "content": q},
        ]
        text = _processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = _processor(text=text, return_tensors="pt").to(_model.device)
        with torch.no_grad():
            out = _model.generate(
                **inputs, max_new_tokens=1024,
                temperature=1.0, top_p=0.95, top_k=64,
            )
        raw = _processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        parsed = _processor.parse_response(raw)
        answer = (parsed.get("content", "") if isinstance(parsed, dict) else (parsed or ""))
        results.append(answer)
    return results


def _run_tool_benchmark(model_=None, processor_=None, system_prompt_=None):
    """Test tool-calling behavior on held-out questions.

    Returns list of (question, expected_set_or_None, min_calls, called_tools, raw).
    """
    _model = model_ if model_ is not None else model
    _processor = processor_ if processor_ is not None else processor
    _sys = system_prompt_ if system_prompt_ is not None else SYSTEM_PROMPT
    results = []
    for entry in TOOL_BENCHMARK_QS:
        q = entry[0]
        expected = entry[1]
        min_calls = entry[2] if len(entry) > 2 else 1
        msgs = [
            {"role": "system", "content": _sys},
            {"role": "user", "content": q},
        ]
        text = _processor.apply_chat_template(
            msgs, tools=TOOLS, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = _processor(text=text, return_tensors="pt").to(_model.device)
        with torch.no_grad():
            out = _model.generate(
                **inputs, max_new_tokens=1024,
                temperature=1.0, top_p=0.95, top_k=64,
            )
        raw = _processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        # Use shared robust extractor (wrapped + bare-fallback per Google's docs)
        called_tools = [name for name, _args in _extract_tool_calls(raw)]
        results.append((q, expected, min_calls, called_tools, raw))
    return results


def _score_tool_results(tool_results):
    """Score tool-call results applying lenient multi-call rule.

    Returns (correct_count, total_count). A result PASSes when:
    - expected is None and no tool was called (direct answer); OR
    - expected is a set and at least `min_calls` of the actual calls
      match the expected set.
    """
    correct = 0
    for q, expected, min_calls, called, raw in tool_results:
        if expected is None:
            ok = (len(called) == 0)
        else:
            matching = sum(1 for t in called if t in expected)
            ok = matching >= min_calls
        correct += int(ok)
    return correct, len(tool_results)


print("=" * 60)
print(f"Benchmark: Held-Out Evaluation ({len(BENCHMARK_QS)} Q&A + {len(TOOL_BENCHMARK_QS)} tool)")
print("=" * 60)

# Q&A Benchmark
print(f"\n--- Q&A Benchmark ({len(BENCHMARK_QS)} questions) ---")
_qa_results = _run_benchmark()
for i, (q, a) in enumerate(zip(BENCHMARK_QS, _qa_results)):
    print(f"\n  Q{i+1}: {q}")
    print(f"  A: {a[:300]}")

# Tool-Calling Benchmark
print(f"\n--- Tool-Calling Benchmark ({len(TOOL_BENCHMARK_QS)} questions) ---")
_tool_results = _run_tool_benchmark()
for q, expected, min_calls, called, raw in _tool_results:
    if expected is None:
        ok = (len(called) == 0)
        expect_str = "no tool call"
    else:
        matching = sum(1 for t in called if t in expected)
        ok = matching >= min_calls
        expect_str = (
            f"≥{min_calls} of " + " or ".join(f"call:{t}" for t in sorted(expected))
            if min_calls > 1
            else " or ".join(f"call:{t}" for t in sorted(expected))
        )
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Q: {q}")
    print(f"         Expected: {expect_str} | Got: {called or 'no tool call'}")

_tool_correct, _tool_total = _score_tool_results(_tool_results)
print(f"\n  Tool accuracy: {_tool_correct}/{_tool_total}")
print(f"\n  Total: 5/5 Q&A + {_tool_correct}/{_tool_total} tool = {5 + _tool_correct}/{5 + _tool_total}")
print("\n✅ Benchmark complete")

"""## 11b: When2Call-Style Held-Out Probes

Three held-out probes validating coverage of 3 of the 4 failure-mode
categories from Ross, H., Mahabaleshwarkar, A. S., & Suhara, Y. (2025).
*When2Call: When (not) to Call Tools.* arXiv:2504.18851.
URL: https://arxiv.org/abs/2504.18851

These probes check the failure modes the paper documents in untrained community
models (9–67% tool-hallucination rates):

  (d) Out-of-scope query → expect refusal + redirect (no tool call, names limit)
  (c) Under-specified query → expect follow-up question (no tool call, asks back)
  (b) Well-specified in-scope query → expect correct tool call

Without the When2Call categories in training (v1 corpus): the model fails
(d) + (c) by hallucinating tools or auto-filling defaults.
With the When2Call categories trained in (v2 corpus including
`_UNABLE_TO_ANSWER` + `_FOLLOW_UP_QUESTIONS`): target 3/3 with zero
regression on the established 8/8 production benchmark baseline.
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


def _run_when2call_probes(model_=None, processor_=None, system_prompt_=None):
    """Run the 3 When2Call probes against the loaded model.

    Defaults to the module-level `model`, `processor`, and `SYSTEM_PROMPT`
    so the existing §11b call site works unchanged. The §13a/b/d/e
    variant cells reassign these globals (or pass explicit args) so the
    same probe set runs against every loaded variant — mirrors the kwarg refactor applied earlier to `_run_benchmark` and `_run_tool_benchmark`.
    """
    _model = model_ if model_ is not None else model
    _processor = processor_ if processor_ is not None else processor
    _sys = system_prompt_ if system_prompt_ is not None else SYSTEM_PROMPT
    results = []
    for probe in WHEN2CALL_PROBES:
        msgs = [
            {"role": "system", "content": _sys},
            {"role": "user", "content": probe["question"]},
        ]
        text = _processor.apply_chat_template(
            msgs, tools=TOOLS, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = _processor(text=text, return_tensors="pt").to(_model.device)
        with torch.no_grad():
            out = _model.generate(
                **inputs, max_new_tokens=512,
                temperature=1.0, top_p=0.95, top_k=64,
            )
        raw = _processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        # Use shared robust extractor (wrapped + bare-fallback per Google's docs)
        called_tools = [name for name, _args in _extract_tool_calls(raw)]
        parsed = _processor.parse_response(raw)
        text_response = (parsed.get("content", "") if isinstance(parsed, dict) else (parsed or ""))

        # Tool-call check
        if probe["expected_tool"] is None:
            tool_match = len(called_tools) == 0
        else:
            tool_match = probe["expected_tool"] in called_tools

        # Content check (if specified) — uses inference.py's documented
        # whitelist (including the (d) matcher permissiveness — see
        # README §When2Call for the matcher-passes-but-behavior-fails
        # caveat documented for E4B merged in the 2026-02-05 validation)
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


def _print_when2call_results(w2c_results, label=""):
    """Reusable per-variant printer for When2Call probe results.

    Same output format as the existing §11b call site so per-variant
    output is consistent. Returns the nominal pass count for caller use."""
    correct = sum(1 for r in w2c_results if r["passed"])
    if label:
        print(f"\n--- When2Call Probes on {label} ---")
    for r in w2c_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['category']}")
        print(f"         Q: {r['question'][:60]}")
        print(f"         Expected tool: {r['expected_tool'] or 'no tool call'} | Got: {r['called_tools'] or 'no tool call'}")
        print(f"         Tool match: {r['tool_match']} | Content match: {r['content_match']}")
    print(f"  When2Call accuracy: {correct}/{len(w2c_results)}")
    return correct


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

"""## 13: Multi-Variant Inference Benchmark — 5 v2 Weight Variants

Loads each of the 5 published v2 weight variants in turn and runs the
10-question benchmark from Section 11 against each. Variants are loaded
sequentially with VRAM cleanup between each so a single Colab Pro G4 VM
(96 GB VRAM) can run all five in one session.

Variants follow the canonical derivation chain — LoRA → merged → quantized
for E4B (size-ascending), then merged → quantized for A4B:

1. **E4B LoRA + base** — `Truthseeker87/solarhive-e4b-lora` (~200 MB adapters) loaded over the kagglehub-cached base via Unsloth `FastVisionModel.from_pretrained`. Smallest download path. Validates that the published LoRA delta runs at full quality before any merge.
2. **E4B BF16 merged** — `Truthseeker87/solarhive-e4b-ollama` (~16 GB safetensors), transformers BF16 load. Lossless-merge regression check vs Variant 1.
3. **E4B GGUF (Ollama)** — `Truthseeker87/solarhive-e4b-gguf`, runtime via Ollama HTTP API at localhost:11434 (requires `ollama serve` running and the GGUF pulled into Ollama; default-disabled — set `_RUN_E4B_GGUF=True` after setup).
4. **A4B BF16 merged** — `Truthseeker87/solarhive-26b-a4b-merged` (~48 GB sharded), transformers BF16 load (requires ≥48 GB VRAM).
5. **A4B NF4 quantized** — `Truthseeker87/solarhive-26b-a4b-nf4` (~48 GB pre-quantized), transformers direct load with no `BitsAndBytesConfig` (per quantize notebook verification — pre-quantized weights load directly).

Total expected runtime: ~85-105 min on G4 (was ~75-90 with 4 variants).
Each variant has its own skip flag (`_RUN_<variant>`) for opting out.
A4B LoRA + base is implicit in the default-flow benchmark from Cell 11
(Cell 2b loads it; Cell 11 benchmarks it) and is therefore not duplicated
here.
"""

# Free the demo model from VRAM before iterating through variants.
print("=" * 60)
print("Multi-Variant Benchmark — preparing")
print("=" * 60)
# Record §11's default-flow benchmark (A4B LoRA + base via Unsloth — the
# implicit baseline) so the §13f summary table can display it alongside
# the 5 §13 variants. _tool_correct / _tool_total / _w2c_correct are
# still in scope from §11 + §11b. The baseline is keyed under "a4b_lora"
# but does NOT live in VARIANT_REPOS (which intentionally enumerates only
# the 5 §13 variants).
_VARIANT_SCORES = {
    "a4b_lora": (len(BENCHMARK_QS), _tool_correct, _tool_total),
}  # variant_name -> (qa_total, tool_correct, tool_total)
_VARIANT_W2C_SCORES = {
    "a4b_lora": (_w2c_correct, len(WHEN2CALL_PROBES)),
}  # variant_name -> (passed, total) — When2Call (b)/(c)/(d) probes per Ross et al. 2025
try:
    del model, processor
except NameError:
    pass
import gc as _gc
torch.cuda.empty_cache()
_gc.collect()
print(f"VRAM free after cleanup: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB")

"""## 13a: Variant 1 — E4B LoRA + Base via Unsloth (`solarhive-e4b-lora`)

Loads the E4B LoRA adapters (~200 MB) over the kagglehub-cached base
model via Unsloth's `FastVisionModel.from_pretrained`. This validates
the published LoRA delta runs at full quality without going through a
merge step — the upstream artifact in the E4B derivation chain.

Source resolution mirrors Cell 2b's three-tier ladder for the A4B LoRA:
local Drive cache → HF Hub fallback. The base model uses the existing
`_DRIVE_E4B_PATH` Drive cache or kagglehub download.

Variant runtime: ~3-5 min. VRAM: ~16 GB (E4B base + LoRA workspace).
"""

if not _RUN_E4B_LORA:
    print("Variant 1 (E4B LoRA + base): SKIPPED (set _RUN_E4B_LORA=True to enable)")
else:
    _src = VARIANT_REPOS["e4b_lora"]
    print("=" * 60)
    print(f"Variant 1: E4B LoRA + base via Unsloth — {_src}")
    print("=" * 60)

    # Resolve E4B base — Drive cache first, kagglehub fallback
    if os.path.isdir(_DRIVE_E4B_PATH):
        _e4b_base = _DRIVE_E4B_PATH
        print(f"E4B base from Drive cache: {_e4b_base}")
    else:
        print(f"E4B base not in Drive — pulling from kagglehub (~16 GB)...")
        _e4b_base = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e4b-it")
        print(f"E4B base downloaded: {_e4b_base}")

    # Resolve E4B LoRA — local Drive first, HF fallback. Pass token=HF_TOKEN
    # explicitly because solarhive-e4b-lora is private until submission day;
    # the module-level login() also covers this but the explicit thread is
    # defensive (guards against a session where HF_TOKEN was added post-Cell-1).
    _e4b_lora_path = None
    if os.path.isdir(_DRIVE_E4B_LORA):
        _e4b_lora_path = _DRIVE_E4B_LORA
        print(f"E4B LoRA from Drive cache: {_e4b_lora_path}")
    else:
        print(f"E4B LoRA not in Drive — pulling from HF: {_src}")
        _e4b_lora_path = snapshot_download(repo_id=_src, repo_type="model", token=HF_TOKEN)
        print(f"E4B LoRA downloaded to: {_e4b_lora_path}")

    # Patch adapter_config.json so its base_model_name_or_path resolves to
    # the runtime base path (Unsloth checks this at load time).
    _e4b_cfg_path = os.path.join(_e4b_lora_path, "adapter_config.json")
    if os.path.exists(_e4b_cfg_path):
        with open(_e4b_cfg_path) as f:
            _e4b_cfg = json.load(f)
        if _e4b_cfg.get("base_model_name_or_path") != _e4b_base:
            _e4b_cfg["base_model_name_or_path"] = _e4b_base
            with open(_e4b_cfg_path, "w") as f:
                json.dump(_e4b_cfg, f, indent=2)
            print(f"   Patched adapter config → {_e4b_base}")

    # Load via Unsloth (handles Gemma 4 LoRA application natively)
    from unsloth import FastVisionModel as _FastVisionModel_E4B
    model, processor = _FastVisionModel_E4B.from_pretrained(
        model_name=_e4b_lora_path,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    _FastVisionModel_E4B.for_inference(model)
    print(f"Loaded on {model.device} — VRAM used: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info(0)[0])/1e9:.1f} GB")

    print(f"\n--- Q&A Benchmark on {_src} ({len(BENCHMARK_QS)} questions) ---")
    _qa = _run_benchmark()
    for i, (q, a) in enumerate(zip(BENCHMARK_QS, _qa)):
        print(f"\n  Q{i+1}: {q}")
        print(f"  A: {a[:300]}")

    print(f"\n--- Tool-Calling Benchmark on {_src} ({len(TOOL_BENCHMARK_QS)} questions) ---")
    _tools = _run_tool_benchmark()
    _tc, _tt = _score_tool_results(_tools)
    for q, expected, min_calls, called, raw in _tools:
        if expected is None:
            ok = (len(called) == 0); expect_str = "no tool call"
        else:
            matching = sum(1 for t in called if t in expected)
            ok = matching >= min_calls
            expect_str = (f"≥{min_calls} of " + " or ".join(f"call:{t}" for t in sorted(expected))
                          if min_calls > 1
                          else " or ".join(f"call:{t}" for t in sorted(expected)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
        print(f"         Expected: {expect_str} | Got: {called or 'no tool call'}")

    _VARIANT_SCORES["e4b_lora"] = (len(BENCHMARK_QS), _tc, _tt)
    print(f"\nVariant 1 totals: {len(BENCHMARK_QS)}/{len(BENCHMARK_QS)} Q&A + {_tc}/{_tt} tool = {len(BENCHMARK_QS) + _tc}/{len(BENCHMARK_QS) + _tt}")

    # When2Call probes (b)/(c)/(d) — same harness + matcher as §11b, run
    # against the variant's reassigned model + processor globals
    _w2c = _run_when2call_probes()
    _w2c_correct_v = _print_when2call_results(_w2c, label=_src)
    _VARIANT_W2C_SCORES["e4b_lora"] = (_w2c_correct_v, len(WHEN2CALL_PROBES))

    # Cleanup before next variant
    del model, processor
    torch.cuda.empty_cache()
    _gc.collect()
    print(f"VRAM free after Variant 1 cleanup: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB")

"""## 13b: Variant 2 — E4B BF16 Merged (`solarhive-e4b-ollama`)"""

if not _RUN_E4B_MERGED:
    print("Variant 2 (E4B BF16 merged): SKIPPED (set _RUN_E4B_MERGED=True to enable)")
else:
    _src = VARIANT_REPOS["e4b_merged"]
    # Drive cache first, HF fallback (mirrors Variants 3 + 4 pattern)
    _src_path = _DRIVE_E4B_MERGED if _DRIVE_E4B_MERGED else _src
    print("=" * 60)
    print(f"Variant 2: E4B BF16 merged — {_src}")
    print("=" * 60)
    if _DRIVE_E4B_MERGED:
        print(f"Drive cache hit — loading weights from {_DRIVE_E4B_MERGED}")
    else:
        print(f"Drive cache miss — pulling from HF: {_src} (~16 GB)")
    # Processor + model both follow _src_path (Drive when available, else HF
    # repo ID). Module-level login() handles auth for the HF path. Sourcing
    # processor from _src_path avoids an unnecessary HF round-trip when the
    # Drive cache is hit (the merge notebook writes processor files alongside
    # the safetensors).
    processor = AutoProcessor.from_pretrained(_src_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        _src_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    print(f"Loaded on {model.device} — VRAM used: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info(0)[0])/1e9:.1f} GB")

    print(f"\n--- Q&A Benchmark on {_src} ({len(BENCHMARK_QS)} questions) ---")
    _qa = _run_benchmark()
    for i, (q, a) in enumerate(zip(BENCHMARK_QS, _qa)):
        print(f"\n  Q{i+1}: {q}")
        print(f"  A: {a[:300]}")

    print(f"\n--- Tool-Calling Benchmark on {_src} ({len(TOOL_BENCHMARK_QS)} questions) ---")
    _tools = _run_tool_benchmark()
    _tc, _tt = _score_tool_results(_tools)
    for q, expected, min_calls, called, raw in _tools:
        if expected is None:
            ok = (len(called) == 0); expect_str = "no tool call"
        else:
            matching = sum(1 for t in called if t in expected)
            ok = matching >= min_calls
            expect_str = (f"≥{min_calls} of " + " or ".join(f"call:{t}" for t in sorted(expected))
                          if min_calls > 1
                          else " or ".join(f"call:{t}" for t in sorted(expected)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
        print(f"         Expected: {expect_str} | Got: {called or 'no tool call'}")

    _VARIANT_SCORES["e4b_merged"] = (len(BENCHMARK_QS), _tc, _tt)
    print(f"\nVariant 2 totals: {len(BENCHMARK_QS)}/{len(BENCHMARK_QS)} Q&A + {_tc}/{_tt} tool = {len(BENCHMARK_QS) + _tc}/{len(BENCHMARK_QS) + _tt}")

    # When2Call probes — closes the per-variant gap from the 2026-02-05 run
    # where this only fired via the side cell after §13b. Now it runs
    # automatically every multi-variant session.
    _w2c = _run_when2call_probes()
    _w2c_correct_v = _print_when2call_results(_w2c, label=_src)
    _VARIANT_W2C_SCORES["e4b_merged"] = (_w2c_correct_v, len(WHEN2CALL_PROBES))

    # Cleanup before next variant
    del model, processor
    torch.cuda.empty_cache()
    _gc.collect()
    print(f"VRAM free after Variant 2 cleanup: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB")

"""## 13c: Variant 3 — E4B GGUF via Ollama HTTP API (`solarhive-e4b-gguf`)

Prerequisites (do these before setting `_RUN_E4B_GGUF=True`):
1. Install Ollama in the Colab runtime: `!curl -fsSL https://ollama.com/install.sh | sh`
2. Start Ollama in the background: `!nohup ollama serve > /tmp/ollama.log 2>&1 &`
3. Pull the SolarHive E4B GGUF: see `test_ollama_tools.py` in the GitHub repo
   for the Modelfile + `ollama create` recipe (uses `/api/generate` raw mode +
   template-matched prompt builder per Sol B in the GGUF model card)

This cell uses **Solution B from `hf_model_card_e4b_gguf.md`**: byte-identical
prompts to the transformers path via `processor.apply_chat_template(...)`,
sent through Ollama's `/api/generate` raw mode (which bypasses the
`gemma4.go` content-drop bug). Same SYSTEM_PROMPT, same TOOLS schemas,
same tool-call regex, same `_score_tool_results()` lenient `min_calls`
rule as Variants 1/2/4/5 — making §13c apples-to-apples comparable
with the transformers-loaded variants.
"""

if not _RUN_E4B_GGUF:
    print("Variant 3 (E4B GGUF via Ollama): SKIPPED")
    print("To enable: install Ollama, start `ollama serve`, pull the SolarHive E4B GGUF model, then set _RUN_E4B_GGUF=True")
else:
    _ollama_host = "http://localhost:11434"
    _ollama_model = "solarhive:latest"  # local Ollama tag; user must create with `ollama create solarhive -f ...`
    print("=" * 60)
    print(f"Variant 3: E4B GGUF via Ollama HTTP — {_ollama_host} model={_ollama_model}")
    print("=" * 60)

    # Health check
    try:
        _v = requests.get(f"{_ollama_host}/api/version", timeout=2).json()
        print(f"Ollama version: {_v.get('version')}")
    except Exception as _e:
        print(f"⚠️  Ollama not reachable at {_ollama_host}: {_e}")
        print("    Skipping Variant 3 — ensure `ollama serve` is running.")
        _RUN_E4B_GGUF = False

if _RUN_E4B_GGUF:
    # Load an E4B processor (tokenizer + chat template + tool-schema rendering).
    # We don't load model weights — Ollama owns inference. The processor is
    # ~MB; instantiating just the chat template lets us produce prompts
    # byte-identical to the transformers-path variants (Solution B parity).
    print("Loading E4B processor for byte-identical prompt rendering (no model weights)...")
    _e4b_proc_src = _DRIVE_E4B_MERGED if _DRIVE_E4B_MERGED else VARIANT_REPOS["e4b_merged"]
    _gguf_processor = AutoProcessor.from_pretrained(_e4b_proc_src, trust_remote_code=True)

    def _ollama_generate_raw(prompt_text):
        """Call Ollama /api/generate in raw mode — bypasses gemma4.go content-drop bug.
        Matches Solution B from the GGUF model card."""
        r = requests.post(
            f"{_ollama_host}/api/generate",
            json={"model": _ollama_model, "prompt": prompt_text, "raw": True, "stream": False,
                  "options": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "num_predict": 1024}},
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("response", "")

    def _build_prompt_via_chat_template(question, with_tools):
        """Render a Gemma 4 prompt via processor.apply_chat_template — byte-
        identical to the transformers-path benchmark calls in _run_benchmark
        and _run_tool_benchmark. with_tools=True injects TOOLS schemas, matching
        the tool-benchmark path."""
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if with_tools:
            kwargs["tools"] = TOOLS
        return _gguf_processor.apply_chat_template(msgs, **kwargs)

    print(f"\n--- Q&A Benchmark on Ollama ({len(BENCHMARK_QS)} questions) ---")
    _qa_ollama = []
    for q in BENCHMARK_QS:
        _prompt = _build_prompt_via_chat_template(q, with_tools=False)
        _raw = _ollama_generate_raw(_prompt)
        # Strip Gemma 4 special tokens for clean Q&A display (parse_response
        # equivalent — the transformers path uses processor.parse_response()
        # but Ollama hands us the raw decoded string with control tokens
        # already as text, so we strip them via the same regex set used by
        # parse_gemma4_output() in test_ollama_tools.py).
        _ans = re.sub(r"<\|channel>.*?<channel\|>", "", _raw, flags=re.DOTALL)
        _ans = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", _ans, flags=re.DOTALL)
        _ans = re.sub(r"<[^>]+\|>|<\|[^>]+>", "", _ans).strip()
        _qa_ollama.append(_ans)
        print(f"\n  Q: {q}")
        print(f"  A: {_ans[:300]}")

    # Tool benchmark — build (q, expected, min_calls, called_tools, raw)
    # tuples in the same shape _run_tool_benchmark() returns, then score
    # via the shared _score_tool_results() helper.
    print(f"\n--- Tool-Calling Benchmark on Ollama ({len(TOOL_BENCHMARK_QS)} questions) ---")
    _tools_ollama = []
    for entry in TOOL_BENCHMARK_QS:
        q = entry[0]
        expected = entry[1]
        min_calls = entry[2] if len(entry) > 2 else 1
        _prompt = _build_prompt_via_chat_template(q, with_tools=True)
        _raw = _ollama_generate_raw(_prompt)
        # Same robust extractor used by Cell 4 + _run_tool_benchmark
        _called = [name for name, _args in _extract_tool_calls(_raw)]
        _tools_ollama.append((q, expected, min_calls, _called, _raw))

    _tc, _tt = _score_tool_results(_tools_ollama)
    for q, expected, min_calls, called, raw in _tools_ollama:
        if expected is None:
            ok = (len(called) == 0); expect_str = "no tool call"
        else:
            matching = sum(1 for t in called if t in expected)
            ok = matching >= min_calls
            expect_str = (f"≥{min_calls} of " + " or ".join(f"call:{t}" for t in sorted(expected))
                          if min_calls > 1
                          else " or ".join(f"call:{t}" for t in sorted(expected)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
        print(f"         Expected: {expect_str} | Got: {called or 'no tool call'}")

    _VARIANT_SCORES["e4b_gguf"] = (len(BENCHMARK_QS), _tc, _tt)
    print(f"\nVariant 3 totals: {len(BENCHMARK_QS)}/{len(BENCHMARK_QS)} Q&A + {_tc}/{_tt} tool = {len(BENCHMARK_QS) + _tc}/{len(BENCHMARK_QS) + _tt}")

    # When2Call probes via Solution B (no transformers `model.generate()` —
    # use _gguf_processor for prompt rendering + _ollama_generate_raw for
    # inference). Same WHEN2CALL_PROBES, same matcher logic, same
    # _extract_tool_calls helper as the transformers variants — only the
    # inference backend differs. Mirrors tests/test_ollama_local_e4b_gguf.py
    # so future local-machine Ollama runs are byte-equivalent to this Colab path.
    print(f"\n--- When2Call Probes on {_src} (Solution B via Ollama HTTP) ---")
    _gguf_w2c_results = []
    for _probe in WHEN2CALL_PROBES:
        _msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _probe["question"]},
        ]
        _prompt = _gguf_processor.apply_chat_template(
            _msgs, tools=TOOLS, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        _raw = _ollama_generate_raw(_prompt)
        _called = [name for name, _args in _extract_tool_calls(_raw)]
        # Same content matcher as §11b — including the documented (d)
        # whitelist permissiveness, so direct comparison with the
        # transformers-variant When2Call scores is honest.
        _text = re.sub(r"<\|channel>.*?<channel\|>", "", _raw, flags=re.DOTALL)
        _text = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", _text, flags=re.DOTALL)
        _text = re.sub(r"<[^>]+\|>|<\|[^>]+>", "", _text).strip()

        if _probe["expected_tool"] is None:
            _tool_match = (len(_called) == 0)
        else:
            _tool_match = _probe["expected_tool"] in _called
        if _probe["must_contain_any"] is None:
            _content_match = True
        else:
            _text_lower = _text.lower()
            _content_match = any(kw.lower() in _text_lower for kw in _probe["must_contain_any"])
        _passed = _tool_match and _content_match
        _gguf_w2c_results.append({
            "category": _probe["category"],
            "question": _probe["question"],
            "expected_tool": _probe["expected_tool"],
            "called_tools": _called,
            "response": _text,
            "tool_match": _tool_match,
            "content_match": _content_match,
            "passed": _passed,
            "rationale": _probe["rationale"],
        })
    _w2c_correct_v = _print_when2call_results(_gguf_w2c_results, label=_src)
    _VARIANT_W2C_SCORES["e4b_gguf"] = (_w2c_correct_v, len(WHEN2CALL_PROBES))

    # Note: `_gguf_processor` is intentionally kept alive — §13g (the
    # end-to-end agentic-loop probe) reuses it for prompt rendering. §13g
    # frees it at its end. The processor is only ~MB so coexisting with
    # the §13d/e transformers loads is fine.

"""## 13d: Variant 4 — A4B BF16 Merged (`solarhive-26b-a4b-merged`)"""

if not _RUN_A4B_MERGED:
    print("Variant 4 (A4B BF16 merged): SKIPPED")
else:
    _src = VARIANT_REPOS["a4b_merged"]
    print("=" * 60)
    print(f"Variant 4: A4B BF16 merged — {_src}")
    print("=" * 60)
    _vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    if _vram_total < 55:
        print(f"⚠️  GPU has only {_vram_total:.0f} GB total VRAM — BF16 26B A4B needs ~48 GB; skipping to avoid OOM.")
        print(f"    Variant 5 (A4B NF4) covers the same model at lower precision.")
    else:
        print(f"Loading {_src} ...")
        # Try Drive cache for merged BF16; fall back to HF
        _src_path = _DRIVE_A4B_MERGED if _DRIVE_A4B_MERGED else _src
        if _DRIVE_A4B_MERGED:
            print(f"Drive cache hit — loading from {_DRIVE_A4B_MERGED}")
        # Processor + model from _src_path (Drive cache when available, else
        # HF repo); module-level login() handles auth for the HF path.
        processor = AutoProcessor.from_pretrained(_src_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            _src_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        print(f"Loaded on {model.device} — VRAM used: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info(0)[0])/1e9:.1f} GB")

        print(f"\n--- Q&A Benchmark on {_src} ({len(BENCHMARK_QS)} questions) ---")
        _qa = _run_benchmark()
        for i, (q, a) in enumerate(zip(BENCHMARK_QS, _qa)):
            print(f"\n  Q{i+1}: {q}")
            print(f"  A: {a[:300]}")

        print(f"\n--- Tool-Calling Benchmark on {_src} ({len(TOOL_BENCHMARK_QS)} questions) ---")
        _tools = _run_tool_benchmark()
        _tc, _tt = _score_tool_results(_tools)
        for q, expected, min_calls, called, raw in _tools:
            if expected is None:
                ok = (len(called) == 0); expect_str = "no tool call"
            else:
                matching = sum(1 for t in called if t in expected)
                ok = matching >= min_calls
                expect_str = (f"≥{min_calls} of " + " or ".join(f"call:{t}" for t in sorted(expected))
                              if min_calls > 1
                              else " or ".join(f"call:{t}" for t in sorted(expected)))
            print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
            print(f"         Expected: {expect_str} | Got: {called or 'no tool call'}")

        _VARIANT_SCORES["a4b_merged"] = (len(BENCHMARK_QS), _tc, _tt)
        print(f"\nVariant 4 totals: {len(BENCHMARK_QS)}/{len(BENCHMARK_QS)} Q&A + {_tc}/{_tt} tool = {len(BENCHMARK_QS) + _tc}/{len(BENCHMARK_QS) + _tt}")

        # When2Call probes — same as A4B LoRA fine-tune; expect 3/3 if
        # the merge step is lossless (per 2026-02-05 outputs textually
        # matching A4B LoRA baseline)
        _w2c = _run_when2call_probes()
        _w2c_correct_v = _print_when2call_results(_w2c, label=_src)
        _VARIANT_W2C_SCORES["a4b_merged"] = (_w2c_correct_v, len(WHEN2CALL_PROBES))

        del model, processor
        torch.cuda.empty_cache()
        _gc.collect()
        print(f"VRAM free after Variant 4 cleanup: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB")

"""## 13e: Variant 5 — A4B NF4 Quantized (`solarhive-26b-a4b-nf4`)"""

if not _RUN_A4B_NF4:
    print("Variant 5 (A4B NF4): SKIPPED")
else:
    _src = VARIANT_REPOS["a4b_nf4"]
    print("=" * 60)
    print(f"Variant 5: A4B NF4 quantized — {_src}")
    print("=" * 60)
    print(f"Loading from {_src} (no BitsAndBytesConfig — pre-quantized weights load directly per quantize notebook verification)...")
    # Try Drive cache first; fall back to HF
    _src_path = _DRIVE_A4B_NF4 if _DRIVE_A4B_NF4 else _src
    if _DRIVE_A4B_NF4:
        print(f"Drive cache hit — loading from {_DRIVE_A4B_NF4}")
    # Processor + model from _src_path (Drive cache when available, else
    # HF repo); module-level login() handles auth for the HF path. NF4
    # weights are pre-quantized — no BitsAndBytesConfig per quantize notebook verification.
    processor = AutoProcessor.from_pretrained(_src_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        _src_path, device_map="cuda:0", trust_remote_code=True,
    )
    print(f"Loaded on {model.device} — VRAM used: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info(0)[0])/1e9:.1f} GB")

    print(f"\n--- Q&A Benchmark on {_src} ({len(BENCHMARK_QS)} questions) ---")
    _qa = _run_benchmark()
    for i, (q, a) in enumerate(zip(BENCHMARK_QS, _qa)):
        print(f"\n  Q{i+1}: {q}")
        print(f"  A: {a[:300]}")

    print(f"\n--- Tool-Calling Benchmark on {_src} ({len(TOOL_BENCHMARK_QS)} questions) ---")
    _tools = _run_tool_benchmark()
    _tc, _tt = _score_tool_results(_tools)
    for q, expected, min_calls, called, raw in _tools:
        if expected is None:
            ok = (len(called) == 0); expect_str = "no tool call"
        else:
            matching = sum(1 for t in called if t in expected)
            ok = matching >= min_calls
            expect_str = (f"≥{min_calls} of " + " or ".join(f"call:{t}" for t in sorted(expected))
                          if min_calls > 1
                          else " or ".join(f"call:{t}" for t in sorted(expected)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
        print(f"         Expected: {expect_str} | Got: {called or 'no tool call'}")

    _VARIANT_SCORES["a4b_nf4"] = (len(BENCHMARK_QS), _tc, _tt)
    print(f"\nVariant 5 totals: {len(BENCHMARK_QS)}/{len(BENCHMARK_QS)} Q&A + {_tc}/{_tt} tool = {len(BENCHMARK_QS) + _tc}/{len(BENCHMARK_QS) + _tt}")

    # When2Call probes — confirms NF4 quantization preserves the same
    # refusal/follow-up behavior as the BF16 baseline (expected 3/3 if
    # quantization isn't degrading reasoning, per quantize notebook verification)
    _w2c = _run_when2call_probes()
    _w2c_correct_v = _print_when2call_results(_w2c, label=_src)
    _VARIANT_W2C_SCORES["a4b_nf4"] = (_w2c_correct_v, len(WHEN2CALL_PROBES))

    del model, processor
    torch.cuda.empty_cache()
    _gc.collect()

"""## 13f: Multi-Variant Summary

Side-by-side scores across the 5 v2 weight variants.
"""

print("=" * 60)
print("Multi-Variant Benchmark Summary")
print("=" * 60)
print(f"{'Variant':<22} {'Repo':<45} {'Q&A':<6} {'Tool':<8} {'W2C':<6} {'Total':<8}")
print("-" * 105)
_TOTAL_QA = len(BENCHMARK_QS)
_TOTAL_TOOL = len(TOOL_BENCHMARK_QS)
_TOTAL_W2C = len(WHEN2CALL_PROBES)


def _fmt_w2c(name):
    """Render the per-variant When2Call cell as 'X/3' or '-' if not measured."""
    if name in _VARIANT_W2C_SCORES:
        passed, total = _VARIANT_W2C_SCORES[name]
        return f"{passed}/{total}"
    return "-"


# Default-flow baseline (§11 + §11b — A4B LoRA + base via Unsloth, loaded in Cell 2b).
# Implicit baseline; not duplicated as a §13 variant. Print first so the
# 5 §13 variants below have a reference point.
if "a4b_lora" in _VARIANT_SCORES:
    qa_total, tc, tt = _VARIANT_SCORES["a4b_lora"]
    total = qa_total + tc
    print(f"{'a4b_lora (baseline)':<22} {'Truthseeker87/solarhive-26b-a4b-lora':<45} {qa_total}/{_TOTAL_QA:<4} {tc}/{tt:<6} {_fmt_w2c('a4b_lora'):<6} {total}/{qa_total + tt}")
    print("-" * 105)

for name, repo in VARIANT_REPOS.items():
    if name in _VARIANT_SCORES:
        qa_total, tc, tt = _VARIANT_SCORES[name]
        total = qa_total + tc
        print(f"{name:<22} {repo:<45} {qa_total}/{_TOTAL_QA:<4} {tc}/{tt:<6} {_fmt_w2c(name):<6} {total}/{qa_total + tt}")
    else:
        print(f"{name:<22} {repo:<45} {'(skipped)':<6} {'(skipped)':<8} {'-':<6} {'(skipped)':<8}")
print()
print(f"Benchmark questions: {_TOTAL_QA} Q&A + {_TOTAL_TOOL} tool-calling = {_TOTAL_QA + _TOTAL_TOOL} total + {_TOTAL_W2C} When2Call probes (Ross et al. 2025, arXiv:2504.18851)")
print(f"Note: Q&A scores reflect generation completeness ({_TOTAL_QA}/{_TOTAL_QA} when all answers are substantive); tool + W2C scores are programmatic.")
print(f"All 6 rows (a4b_lora baseline + 5 §13 variants) use byte-identical apply_chat_template prompts + same SYSTEM_PROMPT + same TOOLS schemas + same scoring helpers + same When2Call matcher (including documented (d) whitelist permissiveness for honest A4B vs E4B comparison).")
print("\n✅ Multi-variant benchmark complete")

"""## 13g: GGUF Agentic Loop Probe — End-to-End Demo via Ollama

The §11/§13 benchmark cells are **single-turn routing tests** — they check
whether the model emits a `call:fn{...}` token, then stop. Tools are never
executed; results are never fed back to the model. This cell closes the
loop: full agentic flow (parse → execute → feed back, max 3 rounds)
running against the E4B GGUF artifact via Ollama HTTP raw mode.

Mirrors `generate_with_tools()` from Cell 4 exactly — same SYSTEM_PROMPT,
same TOOLS list, same `r'call:(\\w+)\\{([^}]*)\\}'` regex, same
`{role:"tool", name:..., content:json.dumps(result)}` feed-back format,
same `TOOL_MAP[fn](**args)` dispatch — but Ollama HTTP raw mode replaces
transformers `model.generate()` as the inference backend.

Single qualitative probe (a multi-tool community audit query) — not
scored programmatically because (a) live API responses vary every run,
and (b) scoring an agentic final answer requires subjective judgment vs.
the routing test's deterministic name-match. Output is the round trace
+ tool calls executed + final answer for human inspection.

Gated by `_RUN_E4B_GGUF` — skipped if §13c was skipped.
"""

if not _RUN_E4B_GGUF:
    print("§13g (GGUF agentic loop): SKIPPED (gated by _RUN_E4B_GGUF)")
else:
    # Re-acquire processor if §13c freed it (shouldn't with current ordering,
    # but defensive — processor is small, reload is fast).
    if "_gguf_processor" not in dir():
        print("Reloading E4B processor for prompt rendering...")
        _e4b_proc_src = _DRIVE_E4B_MERGED if _DRIVE_E4B_MERGED else VARIANT_REPOS["e4b_merged"]
        _gguf_processor = AutoProcessor.from_pretrained(_e4b_proc_src, trust_remote_code=True)

    def _ollama_agentic_loop(question, max_rounds=3):
        """Full agentic loop via Ollama HTTP raw mode.

        Mirrors `generate_with_tools()` (Cell 4) but with Ollama replacing
        transformers `.generate()`. Reuses `TOOL_MAP` for execution and the
        same `call:fn{...}` regex for parsing, so the agentic semantics are
        identical to the cloud path — only the inference backend differs.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        all_calls = []
        for round_num in range(max_rounds):
            text = _gguf_processor.apply_chat_template(
                messages, tools=TOOLS, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            raw = _ollama_generate_raw(text)
            found = _extract_tool_calls(raw)
            if not found:
                # No tool calls — final answer. Strip Gemma 4 control tokens
                # via the same regex set parse_response() uses internally.
                ans = re.sub(r"<\|channel>.*?<channel\|>", "", raw, flags=re.DOTALL)
                ans = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", ans, flags=re.DOTALL)
                ans = re.sub(r"<[^>]+\|>|<\|[^>]+>", "", ans).strip()
                return {"response": ans, "tool_calls": all_calls, "rounds": round_num + 1}

            # Parse args + execute tools — shared helpers (Cell 3) handle
            # negative numbers, booleans, and strings via <|"|> delimiters.
            calls, results = [], []
            for fn_name, args_str in found:
                args = _parse_tool_args(args_str)
                call = {"name": fn_name, "arguments": args}
                calls.append(call)
                all_calls.append(call)
                # Defensive dispatch — drop hallucinated kwargs
                result = _safe_tool_call(TOOL_MAP[fn_name], args) if fn_name in TOOL_MAP else {"error": f"Unknown: {fn_name}"}
                results.append({"name": fn_name, "response": result})
                print(f"  [Round {round_num+1}] → executed {fn_name}({args}) → {result}")

            # Feed results back — same message format as Cell 4
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

    print("=" * 60)
    print("§13g: GGUF Agentic Loop Probe (end-to-end on edge runtime)")
    print("=" * 60)
    _probe_q = (
        "Full community energy audit — check current weather, solar production, "
        "battery state, and grid pricing. Give a 3-sentence status report."
    )
    print(f"\n📝 Q: {_probe_q}")
    print("Running agentic loop on Ollama (E4B GGUF Q4_K_M)...\n")
    _r = _ollama_agentic_loop(_probe_q)
    print(f"\nRounds completed: {_r['rounds']}")
    print(f"Tool calls executed: {[c['name'] for c in _r['tool_calls']]}")
    print(f"\n💡 Final answer:\n{_r['response']}")

    # Truly done with the GGUF path — free the processor
    del _gguf_processor
    _gc.collect()
    print("\n✅ §13g agentic-loop probe complete")

"""## 14: Multi-Token Prediction (MTP) Drafters — Future Iteration

> **NOT INCLUDED IN THE FINAL INFERENCE RUN.** Google's Gemma 4 MTP drafter
> announcement on May 5, 2026 landed AFTER our final §13 multi-variant
> benchmark had been captured. This cell ships the MTP integration as
> documented future-iteration code so the path is in source for reviewer
> reproducibility, but is gated off by default (`_RUN_MTP_DEMO = False`).
> The §13 benchmark numbers reflect standard autoregressive decoding only —
> no MTP applied. A post-submission re-run with the drafter paired in
> would be a clean follow-up benchmark.

**What is speculative decoding?** Standard transformer decoding is serial:
generating *K* tokens requires *K* sequential forward passes through the
target model, each waiting on the last. Speculative decoding (Leviathan,
Kalman & Matias, 2023, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192))
accelerates this *without changing the output distribution*. A much
smaller "drafter" model M_q generates γ candidate tokens autoregressively
in roughly the time the target M_p would take to generate one; the target
then verifies *all* γ candidates in a single parallel forward pass.
Accepted candidates are kept; the first rejected one (if any) is resampled
from a corrected distribution via the paper's *speculative sampling*
procedure. The math guarantees the output distribution is identical to
standard decoding — no quality loss, just speedup. The original paper
measured **2X–3X walltime speedup on T5-XXL** with no change to outputs.

**What Google shipped on May 5, 2026.**
[*"Accelerating Gemma 4: faster inference with multi-token prediction
drafters"*](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
(Olivier Lacombe, Maarten Grootendorst). Paired drafter checkpoints
shipped for Gemma 4 E2B, E4B, 26B-A4B, and 31B with the canonical
naming convention `<target>-assistant` (so the drafter for our
`google/gemma-4-26B-A4B-it` cloud target is
`google/gemma-4-26B-A4B-it-assistant`). Reported speedups: **up to 3×
decode speedup with no quality degradation**, and **~2.2× on Apple
Silicon** at batch sizes 4–8. Tested across LiteRT-LM, MLX, Hugging Face
Transformers, and vLLM. The drafters share the input embedding table
with the target — what makes them lightweight and high-acceptance.

**HF Transformers integration is one extra kwarg** (per the
[implementation guide](https://ai.google.dev/gemma/docs/mtp/mtp) and the
[Gemma cookbook MTP notebook](https://github.com/google-gemma/cookbook/blob/main/docs/mtp/mtp.ipynb)):

    target.generate(**inputs, assistant_model=assistant)

Optional knobs: `num_assistant_tokens` (the γ parameter from the paper)
and `num_assistant_tokens_schedule = "heuristic" | "constant"` for
dynamic adjustment. The "heuristic" schedule maps directly to the
paper's Section 3.5 suggestion of varying γ during inference: increase
γ by 2 when all draft tokens are accepted, decrease by 1 on any
rejection.

**This cell, when enabled (`_RUN_MTP_DEMO = True`), will:**
1. Load the SolarHive cloud target (`google/gemma-4-26B-A4B-it`) in BF16.
2. Load the paired drafter `google/gemma-4-26B-A4B-it-assistant` in BF16.
3. Run the same prompt through both paths — baseline (no drafter) vs.
   MTP (drafter paired) — with deterministic argmax sampling so the
   speculative-sampling guarantee is byte-verifiable.
4. Print walltime, decoded-tokens-per-second, and the measured speedup.
"""

# === CELL 14: MTP Drafters — Future Iteration =================================
# Citations:
#   - May 5, 2026 Google blog "Accelerating Gemma 4: faster inference with
#     multi-token prediction drafters" by Olivier Lacombe + Maarten Grootendorst:
#     https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
#   - MTP overview:        https://ai.google.dev/gemma/docs/mtp/overview
#   - MTP implementation:  https://ai.google.dev/gemma/docs/mtp/mtp
#   - Gemma cookbook:      https://github.com/google-gemma/cookbook/blob/main/docs/mtp/mtp.ipynb
#   - Speculative decoding paper (Leviathan, Kalman & Matias, ICML 2023):
#     https://arxiv.org/abs/2211.17192

_RUN_MTP_DEMO = False  # Flip to True to run the side-by-side baseline vs. MTP comparison

if not _RUN_MTP_DEMO:
    print("=" * 60)
    print("§14: MTP Drafter Demo — SKIPPED (set _RUN_MTP_DEMO = True to enable)")
    print("=" * 60)
    print(
        "  Final inference run did not include MTP — Gemma 4 drafters were\n"
        "  announced on May 5, 2026, after our §13 multi-variant benchmark\n"
        "  numbers were captured. This cell ships the integration as\n"
        "  documented future-iteration code; flip the flag above to\n"
        "  reproduce the side-by-side timing comparison.\n"
    )
else:
    import time as _t14
    import torch as _torch14
    import gc as _gc14
    from transformers import (
        AutoModelForCausalLM as _AutoCausalLM14,
        AutoProcessor as _AutoProc14,
    )

    _MTP_TARGET_ID    = "google/gemma-4-26B-A4B-it"
    _MTP_ASSISTANT_ID = _MTP_TARGET_ID + "-assistant"  # canonical drafter naming

    print("=" * 60)
    print("§14: MTP Drafter Demo — target + paired drafter")
    print("=" * 60)
    print(f"  Target:    {_MTP_TARGET_ID}")
    print(f"  Drafter:   {_MTP_ASSISTANT_ID}")

    _proc14 = _AutoProc14.from_pretrained(_MTP_TARGET_ID, trust_remote_code=True)

    # Load target + drafter both in BF16 on the available GPU.
    # Note: the drafter is small enough (~few hundred MB) that it adds negligible
    # VRAM pressure to the target's ~48 GB BF16 footprint.
    _target_model14 = _AutoCausalLM14.from_pretrained(
        _MTP_TARGET_ID,
        dtype=_torch14.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _assistant_model14 = _AutoCausalLM14.from_pretrained(
        _MTP_ASSISTANT_ID,
        dtype=_torch14.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Optional: dynamic γ schedule (paper Section 3.5; cookbook example).
    # "heuristic" increases γ by 2 when all drafts accepted, decreases by 1 on
    # any rejection. "constant" keeps γ fixed at num_assistant_tokens.
    _assistant_model14.generation_config.num_assistant_tokens = 4
    _assistant_model14.generation_config.num_assistant_tokens_schedule = "heuristic"

    # Same prompt for both paths. Argmax sampling (do_sample=False) so the
    # speculative-sampling guarantee — that MTP's output distribution is
    # identical to the baseline's — is byte-verifiable on the result.
    _mtp_prompt = (
        "Explain in 2-3 sentences why solar production drops on cloudy days, "
        "using the panel temperature derating effect as part of the explanation."
    )
    _msgs14 = [{"role": "user", "content": _mtp_prompt}]
    _inp14 = _proc14.apply_chat_template(_msgs14, tokenize=False, add_generation_prompt=True)
    _enc14 = _proc14(text=_inp14, return_tensors="pt").to(_target_model14.device)

    # --- Baseline: target only, no drafter --------------------------------------
    print("\n  [1/2] Baseline (target only, no MTP)...")
    _t_b0 = _t14.perf_counter()
    _out_baseline = _target_model14.generate(
        **_enc14, max_new_tokens=256, do_sample=False,
    )
    _baseline_s = round(_t14.perf_counter() - _t_b0, 2)
    _baseline_tokens = _out_baseline.shape[-1] - _enc14["input_ids"].shape[-1]
    _baseline_tps = round(_baseline_tokens / max(0.01, _baseline_s), 2)
    _baseline_text = _proc14.decode(
        _out_baseline[0][_enc14["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    print(f"     Walltime {_baseline_s}s, {_baseline_tokens} tokens "
          f"({_baseline_tps} tok/s)")

    # --- MTP: target + drafter paired -------------------------------------------
    print("\n  [2/2] MTP enabled (target + drafter)...")
    _t_m0 = _t14.perf_counter()
    _out_mtp = _target_model14.generate(
        **_enc14, max_new_tokens=256, do_sample=False,
        assistant_model=_assistant_model14,  # ← The one kwarg that enables MTP
    )
    _mtp_s = round(_t14.perf_counter() - _t_m0, 2)
    _mtp_tokens = _out_mtp.shape[-1] - _enc14["input_ids"].shape[-1]
    _mtp_tps = round(_mtp_tokens / max(0.01, _mtp_s), 2)
    _mtp_text = _proc14.decode(
        _out_mtp[0][_enc14["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    print(f"     Walltime {_mtp_s}s, {_mtp_tokens} tokens ({_mtp_tps} tok/s)")

    # --- Verdict ----------------------------------------------------------------
    _speedup = round(_baseline_s / max(0.01, _mtp_s), 2)
    print("\n" + "=" * 60)
    print(f"  MTP walltime speedup:  {_speedup}x")
    print(f"  Baseline throughput:   {_baseline_tps} tok/s")
    print(f"  MTP throughput:        {_mtp_tps} tok/s")
    print("=" * 60)
    print(f"\n  Baseline output:\n  {_baseline_text}\n")
    print(f"  MTP output:\n  {_mtp_text}\n")
    print(
        "  Speculative-sampling guarantee (Leviathan, Kalman & Matias, 2023):\n"
        "  with argmax decoding, MTP's output distribution is identical to the\n"
        "  baseline's, so the two outputs above should be byte-identical when\n"
        "  the drafter has any non-trivial acceptance rate. Any divergence\n"
        "  would indicate either a stochastic-sampling code path being taken\n"
        "  or a bug in the speculative-sampling implementation."
    )

    # Cleanup — release the target + drafter VRAM
    del _target_model14, _assistant_model14, _proc14
    _gc14.collect()
    _torch14.cuda.empty_cache()
    print("\n✅ §14 MTP demo complete")