# -*- coding: utf-8 -*-
"""SolarHive — LiteRT-LM Mobile-Edge Runtime Notebook
======================================================
SolarHive is an open-source intelligence layer designed to coordinate
community microgrids & community-based storage via fuel cells, pool
midday energy surplus across these microgrids, and eliminate stranded
capacity. It also helps forecast solar irradiance and cloud cover to
plan ahead.

PURPOSE: Demonstrate Google's LiteRT-LM as the runtime for SolarHive's
agentic loop on Gemma 4 E4B, validated on a CPU-only Linux notebook as a
proxy for the mobile-edge deployment matrix LiteRT-LM ships SDKs for —
Android (Kotlin), iOS / macOS (C++; Swift coming), Linux Python on edge
hardware (Raspberry Pi 5, NVIDIA Jetson Orin Nano Super, laptops), and
Windows Python (upcoming). The same `.litertlm` bundle, the same agentic
loop shape, and the same SolarHive system prompt deploy across every
target — that is the cross-platform contract this notebook validates.

ALTERNATIVE TO CACTUS: This notebook is the LiteRT-LM-based alternative
to SolarHive's Cactus mobile deployment. Both target the same edge-
inference use case (a phone, microgrid hub, or single-board computer
running Gemma 4 E4B locally), and both pair with the SolarHive cloud tier
(Gemma 4 26B A4B) and microgrid-hub tier (Ollama on E4B GGUF) via emoji-
routed task handoff. The Cactus implementation lives at `mobile-cactus/`
in this repository (Flutter Android app on fine-tuned E4B); the LiteRT-LM
browser companion lives at `web-litert/` (`.task` bundle on WebGPU). This
notebook covers the third leg — LiteRT-LM running natively on edge Linux
Python — completing the mobile / browser / native edge runtime trio.

Features:
  1. LiteRT-LM Python SDK (`litert-lm-api-nightly`) — Engine + Conversation
     context-managed lifecycle, `send_message_async` streaming
  2. Upstream pre-converted Gemma 4 E4B `.litertlm` bundle (3.66 GB) from
     `litert-community/gemma-4-E4B-it-litert-lm` — same bundle deploys to
     Android Kotlin / iOS C++ / Linux Python / Windows Python via the
     LiteRT-LM SDK family
  3. SolarHive agentic loop with native Gemma 4 function calling — five
     tools (weather, solar production, battery state, grid status, NREL
     PVWatts baseline), 2-message tool-result reply shape, regex-based
     tool-call extraction tolerant of both Gemma 4 native (curly-brace)
     and Python-style (parens) emit forms
  4. Multi-Token Prediction (MTP) speedup probe — runtime API discovery
  5. Multi-modal VQA probe — sky-photo analysis on the Ann Arbor, MI
     community-solar deployment
  6. When2Call sub-bench — three held-out probes from Ross et al. 2025
     (arXiv:2504.18851) for tool-call routing failure modes (b/c/d)
  7. Resilient benchmark loop — per-prompt JSON checkpoint to
     `/content/v3_1_checkpoints/bench_results.json` so a mid-run kernel
     interrupt leaves recoverable evidence for the verdict
  8. Verdict block tabulating measured Colab x86_64 CPU decode tok/s
     against the verbatim mobile-edge benchmarks published in the
     `litert-community/gemma-4-E4B-it-litert-lm` model card (12 published
     hardware targets — Pi 5 16GB CPU, Linux ARM 2.3-2.8GHz, macOS M4 Max
     CPU/GPU, Windows Intel LunarLake CPU/GPU, Android Samsung S26 Ultra
     CPU/GPU, iPhone 17 Pro CPU/GPU, Web Chrome on M4 Max GPU). SolarHive
     UX latency proxy (100-token short imperative answer) per target.

SETUP: Google Colab Pro **CPU + High-RAM** (Linux x86_64). The CPU choice
is intentional — it matches the Pi 5 / Jetson / Android edge story; GPU
on Colab would demonstrate cloud, not edge. The 3.66 GB `.litertlm` plus
Python overhead plus tool-call working memory needs High-RAM (default
Colab ~12 GB is tight). Output ports unchanged to ARM Linux on Pi 5 16GB
or Jetson Orin Nano Super.

Gemma is a trademark of Google LLC.
PRIZE TARGET: LiteRT Special Technology Track

References:
  - LiteRT-LM landing page:
      https://ai.google.dev/edge/litert-lm
  - LiteRT-LM repository:
      https://github.com/google-ai-edge/LiteRT-LM
  - "Bring state-of-the-art agentic skills to the edge with Gemma 4"
    (Google Developers Blog, April 2, 2026):
      https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/
  - "Accelerating Gemma 4: faster inference with multi-token prediction
    drafters" (Google blog, May 5, 2026 — describes the MTP drafter model
    architecture: a heavy target model paired with a lightweight drafter
    artifact for up to 3x decode speedup, validated on LiteRT-LM, MLX,
    HF Transformers, and vLLM):
      https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
  - Multi-Token Prediction overview (Google AI for Developers):
      https://ai.google.dev/gemma/docs/mtp/overview
  - Multi-Token Prediction implementation guide (HF Transformers code
    pattern; pins the `<target_id>-assistant` drafter naming convention,
    e.g., `google/gemma-4-E4B-it-assistant` is the drafter for
    `google/gemma-4-E4B-it`):
      https://ai.google.dev/gemma/docs/mtp/mtp
  - Gemma 4 E4B `.litertlm` model card (verified hardware benchmarks
    cited verbatim in Cell 11 verdict):
      https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
  - SolarHive Cactus mobile alternative — Flutter Android app on
    fine-tuned Gemma 4 E4B:
      `mobile-cactus/` in this repository
  - SolarHive LiteRT browser companion — `.task` bundle on WebGPU via
    MediaPipe Tasks Web:
      `web-litert/` in this repository

Pipeline: Install LiteRT-LM SDK -> resolve `.litertlm` bundle -> Engine
         warmup -> tool-equipped agentic loop -> 8-prompt benchmark ->
         MTP probe -> VQA probe -> When2Call sub-bench -> verdict with
         mobile-edge benchmark tabulation
"""

"""## 0: Install LiteRT-LM Python SDK + supporting deps

Pin-then-fallback pattern — `litert-lm-api-nightly` is a nightly distribution
and individual snapshots can be pruned from PyPI between runs. Capture the
actually-installed version in the verdict block so any drift is recorded.

The SDK is the **only** install we strictly need for Phase 1. Everything else
(`huggingface_hub` for bundle download, `pillow` + `requests` for the tool
functions, `numpy` for benchmark math) is peripheral and stable.
"""

import subprocess as _sp
import sys as _sys
import importlib

print("=" * 72)
print("  Cell 0 — installing LiteRT-LM Python SDK + peripheral deps")
print("=" * 72)

# Tier 1 — peripheral deps (mature, stable)
_sp.check_call([
    _sys.executable, "-m", "pip", "install", "-q", "-U",
    "huggingface_hub",
    "pillow",
    "requests",
    "numpy",
])

# Tier 2 — LiteRT-LM Python SDK with try-pin-then-fallback
_LITERT_LM_AVAILABLE = False
_LITERT_LM_VERSION = None
_LITERT_LM_INSTALL_LOG = []

for _pkg in ["litert-lm-api-nightly", "litert-lm-api"]:
    try:
        _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "-U", _pkg])
        _LITERT_LM_INSTALL_LOG.append(f"installed: {_pkg}")
        # Force fresh import in case a previous attempt cached
        for _mod in list(_sys.modules):
            if _mod.startswith("litert_lm"):
                del _sys.modules[_mod]
        import litert_lm  # noqa: F401
        _LITERT_LM_AVAILABLE = True
        _LITERT_LM_VERSION = getattr(litert_lm, "__version__", "unknown")
        break
    except Exception as _e:
        _LITERT_LM_INSTALL_LOG.append(f"failed: {_pkg} ({type(_e).__name__})")
        continue

print()
for _line in _LITERT_LM_INSTALL_LOG:
    print(f"  {_line}")
print(f"\n  litert_lm import: {'✅' if _LITERT_LM_AVAILABLE else '❌'}")
if _LITERT_LM_AVAILABLE:
    print(f"  litert_lm version: {_LITERT_LM_VERSION}")
print("=" * 72)

if not _LITERT_LM_AVAILABLE:
    print(
        "\n  ⚠️ litert_lm import failed. Phase 1 cannot proceed. Cells below "
        "will graceful-skip. Likely causes: (1) PyPI prune of the nightly; "
        "(2) Linux/macOS-only platform constraint not met; (3) Python "
        "version mismatch. Capture the install log to the verdict block.\n"
    )

"""## 1: Imports + secrets + SolarHive constants

Mirrors `solarhive_inference.py` Cell 1 — same lat/lon, same community
capacity, same secrets-resolution path (Kaggle Secrets first, Colab userdata
fallback). All five SolarHive tools use the same numeric constants so the
agentic-loop output is byte-comparable across runtimes.
"""

import json
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import numpy as np

# --- Secrets resolution: Kaggle Secrets → Colab userdata → environment -------
OWM_API_KEY = NREL_API_KEY = EIA_API_KEY = HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    _us = UserSecretsClient()
    OWM_API_KEY  = _us.get_secret("OWM_API_KEY")
    NREL_API_KEY = _us.get_secret("NREL_API_KEY")
    EIA_API_KEY  = _us.get_secret("EIA_API_KEY")
    try:
        HF_TOKEN = _us.get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = None
    print("   Keys loaded from Kaggle Secrets")
except Exception:
    try:
        from google.colab import userdata
        OWM_API_KEY  = userdata.get("OWM_API_KEY")
        NREL_API_KEY = userdata.get("NREL_API_KEY")
        EIA_API_KEY  = userdata.get("EIA_API_KEY")
        try:
            HF_TOKEN = userdata.get("HF_TOKEN")
        except Exception:
            HF_TOKEN = None
        print("   Keys loaded from Colab userdata")
    except Exception:
        OWM_API_KEY  = os.environ.get("OWM_API_KEY")
        NREL_API_KEY = os.environ.get("NREL_API_KEY")
        EIA_API_KEY  = os.environ.get("EIA_API_KEY")
        HF_TOKEN     = os.environ.get("HF_TOKEN")
        print("   Keys loaded from environment")

# Optional HF auth (read-only suffices for litert-community public repo)
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("   HF auth: ✅")
    except Exception as _e:
        print(f"   HF auth: skipped ({type(_e).__name__})")

# --- SolarHive community constants (Ann Arbor, Michigan) ---------------------
LAT, LON = 42.2808, -83.7430
COMMUNITY_CAPACITY_KW = 72
BATTERY_CAPACITY_KWH = 100
SYSTEM_EFF = 0.85  # inverter 97% × wiring 98% × soiling 97% × mismatch 98%

# --- LiteRT bundle constants -------------------------------------------------
HF_REPO_ID = "litert-community/gemma-4-E4B-it-litert-lm"
LITERTLM_FILENAME_HINT = ".litertlm"  # resolved at runtime in Cell 3
LOCAL_CACHE_DIR = "/content/litert_e4b_bundle"

# --- Per-prompt benchmark checkpoint dir -------------------------------------
# Each iteration of the 8-prompt benchmark writes the running `bench_results`
# list to a JSON file here, so a kernel OOM or SIGTERM mid-run leaves usable
# evidence for the verdict. Cell 11 reads from the checkpoint if any partial
# state survived a crash.
CHECKPOINT_DIR = "/content/v3_1_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Phase 2 conversion-probe gate (default OFF — Phase 1 is the deliverable) -
# Runtime demo is the deliverable; the fine-tuned-Gemma-4 → `.litertlm`
# conversion probe is documented due-diligence kept in this file as gated
# code, but executed only when explicitly opted in.
_PHASE2_ENABLED = False

print()
print(f"   LAT, LON                = {LAT}, {LON}")
print(f"   COMMUNITY_CAPACITY_KW   = {COMMUNITY_CAPACITY_KW}")
print(f"   BATTERY_CAPACITY_KWH    = {BATTERY_CAPACITY_KWH}")
print(f"   HF_REPO_ID              = {HF_REPO_ID}")
print(f"   CHECKPOINT_DIR          = {CHECKPOINT_DIR}")
print(f"   _PHASE2_ENABLED         = {_PHASE2_ENABLED}")
print(f"   _LITERT_LM_AVAILABLE    = {_LITERT_LM_AVAILABLE}")

"""## Phase 1 — LiteRT-LM Python edge runtime

Engine init → tool-equipped agentic loop → 8-prompt benchmark → W2C-3 sub-bench
→ multi-modal VQA → MTP delta → numeric verdict.
"""

"""## 3: Resolve the upstream pre-converted .litertlm bundle

`litert_lm.Engine()` requires the **explicit `.litertlm` file path**, not the
`snapshot_download()` directory. We use `snapshot_download(allow_patterns)`
to fetch only the `.litertlm` and tokenizer companions — skipping the 2.96 GB
`.task` browser bundle the same repo also ships.
"""

from huggingface_hub import snapshot_download

if _LITERT_LM_AVAILABLE:
    _t0 = time.perf_counter()
    LOCAL_BUNDLE_DIR = snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=LOCAL_CACHE_DIR,
        allow_patterns=["*.litertlm", "*.json", "*.txt", "*.md"],
        token=HF_TOKEN,
    )
    _bundle_dl_s = round(time.perf_counter() - _t0, 2)

    # Resolve to explicit .litertlm file path
    _candidates = sorted(Path(LOCAL_BUNDLE_DIR).glob("*.litertlm"))
    if not _candidates:
        print("   ❌ no .litertlm file found in snapshot — graceful skip")
        LITERTLM_PATH = None
    else:
        LITERTLM_PATH = str(_candidates[0])
        _size_gb = Path(LITERTLM_PATH).stat().st_size / 1e9
        print(f"   ✅ resolved: {LITERTLM_PATH}")
        print(f"   ✅ size:     {_size_gb:.2f} GB")
        print(f"   ✅ download: {_bundle_dl_s}s")
else:
    LITERTLM_PATH = None
    print("   ⏭️ skipped (litert_lm unavailable)")

"""## 4: Engine init + warm-up smoke test

Validates the cold-start path and one-shot inference. Captures TTFT
(time-to-first-token) and total-decode time for the verdict block.

The LiteRT-LM Python API uses two nested context managers — `Engine` for the
model lifecycle and `Conversation` for stateful multi-turn interactions
(per `ai.google.dev/edge/litert-lm/python` "Getting Started" + "Initialize the
Engine" + "Create a Conversation" sections).
"""

import litert_lm

_warmup = {"ok": False, "cold_start_s": None, "ttft_s": None, "total_s": None,
           "decode_tokens": 0, "response": None, "error": None}

if _LITERT_LM_AVAILABLE and LITERTLM_PATH:
    try:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

        _t_engine = time.perf_counter()
        with litert_lm.Engine(LITERTLM_PATH) as _engine:
            _warmup["cold_start_s"] = round(time.perf_counter() - _t_engine, 2)
            print(f"   Engine cold-start: {_warmup['cold_start_s']}s")

            with _engine.create_conversation() as _conv:
                _prompt = (
                    "What is solar GHI (Global Horizontal Irradiance)? "
                    "Answer in one sentence."
                )
                _t_send = time.perf_counter()
                _first_chunk_t = None
                _chunks = []
                for _chunk in _conv.send_message_async(_prompt):
                    if _first_chunk_t is None:
                        _first_chunk_t = time.perf_counter()
                    # Defensive: chunk shape may be {"content": [{"text": ...}]} or str
                    if isinstance(_chunk, dict):
                        _content = _chunk.get("content", [])
                        if _content and isinstance(_content, list):
                            _txt = _content[0].get("text", "") if isinstance(_content[0], dict) else str(_content[0])
                        else:
                            _txt = str(_chunk)
                    else:
                        _txt = str(_chunk)
                    _chunks.append(_txt)
                _t_done = time.perf_counter()

                _warmup["ttft_s"] = round((_first_chunk_t or _t_done) - _t_send, 2)
                _warmup["total_s"] = round(_t_done - _t_send, 2)
                _warmup["decode_tokens"] = sum(len(c.split()) for c in _chunks)
                _warmup["response"] = "".join(_chunks).strip()
                _warmup["ok"] = bool(_warmup["response"])

        print(f"   TTFT:              {_warmup['ttft_s']}s")
        print(f"   Total decode:      {_warmup['total_s']}s")
        print(f"   Approx tokens:     {_warmup['decode_tokens']}")
        print(f"   Response preview:  {(_warmup['response'] or '')[:200]}")
    except Exception as _e:
        _warmup["error"] = f"{type(_e).__name__}: {_e}"
        print(f"   ❌ engine init failed: {_warmup['error']}")
else:
    print("   ⏭️ skipped (Cell 0 or Cell 3 prerequisite missing)")

"""## 5: SolarHive tool definitions (5 tools, byte-equivalent to inference.py)

Ported verbatim from `solarhive_inference.py` Cell 4. All numeric constants
match (SYSTEM_EFF=0.85, Fahrenheit-77 derating threshold, NREL PVWatts v8
defaults). The agentic loop in Cell 6 dispatches tool calls through `TOOL_MAP`
exactly the way the cloud and Ollama runtimes do.

Two tools execute fully on-device with no API key (`get_solar_production`
via Open-Meteo keyless GHI; `get_battery_state` via session simulator). The
three keyed tools (OWM, EIA, NREL) execute via the same Python helper that
the LiteRT browser tier escalates to the microgrid hub for — but in the v3
notebook all five tools run in-process for a clean apples-to-apples bench
against the cloud variants.
"""

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
    """Fetch current GHI (W/m²) from Open-Meteo (keyless, NOAA GFS+HRRR backed)."""
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
        clouds_pct: Current cloud cover percentage (0-100).
        temp_f: Current temperature in Fahrenheit.

    Returns:
        Dictionary with production_kw, capacity_kw, efficiency_pct, ghi_wm2, temp_derate_pct, source.
    """
    clouds_pct = max(0, min(100, int(clouds_pct)))
    temp_f = max(-40, min(130, float(temp_f)))
    temp_derate = max(0.75, 1.0 - 0.004 * max(0, temp_f - 77))

    ghi = _get_current_ghi()
    if ghi is not None:
        production = round(max(0, COMMUNITY_CAPACITY_KW * (ghi / 1000) * SYSTEM_EFF * temp_derate), 1)
        return {
            "production_kw": production,
            "capacity_kw": COMMUNITY_CAPACITY_KW,
            "efficiency_pct": round(production / COMMUNITY_CAPACITY_KW * 100, 1),
            "ghi_wm2": round(ghi, 1),
            "temp_derate_pct": round(temp_derate * 100, 1),
            "source": "open-meteo",
        }

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
    """Maintains consistent SOC across tool calls within a session."""
    def __init__(self, capacity_kwh=BATTERY_CAPACITY_KWH):
        self.capacity = capacity_kwh
        self.soc = round(random.uniform(55, 85), 1)

    def get_state(self):
        return {
            "soc_pct": self.soc,
            "kwh_stored": round(self.soc / 100 * self.capacity),
            "capacity_kwh": self.capacity,
            "charging": self.soc < 50,
        }

_battery = _BatterySimulator()


def get_battery_state() -> dict:
    """Gets the current state of the community shared battery storage.

    Returns:
        Dictionary with soc_pct, kwh_stored, capacity_kwh, charging.
    """
    return _battery.get_state()


_FALLBACK_GRID = {"MISO": {"renewable_pct": 12.5, "co2_intensity": 520}}


def _fetch_eia_grid_mix():
    """Fetch MISO grid mix from EIA v2 API. Returns (renewable_pct, co2_intensity)."""
    try:
        end = datetime.now(timezone.utc) - timedelta(days=1)
        start = end - timedelta(days=1)
        r = requests.get(
            "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/",
            params={
                "api_key": EIA_API_KEY,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": "MISO",
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
            fb = _FALLBACK_GRID["MISO"]
            return fb["renewable_pct"], fb["co2_intensity"]
        latest_period = rows[0].get("period")
        latest = [row for row in rows if row.get("period") == latest_period]
        total_mw, renewable_mw, co2_total = 0, 0, 0
        _RENEWABLE = {"SUN", "WND", "WAT", "GEO"}
        _FOSSIL_CO2 = {"COL": 1000, "NG": 450, "PET": 900, "OTH": 500}
        for row in latest:
            mw = float(row.get("value") or 0)
            fuel = row.get("fueltype", "")
            total_mw += mw
            if fuel in _RENEWABLE:
                renewable_mw += mw
            co2_total += mw * _FOSSIL_CO2.get(fuel, 0)
        if total_mw > 0:
            return (
                min(100.0, round(renewable_mw / total_mw * 100, 1)),
                max(0, round(co2_total / total_mw, 1)),
            )
    except Exception:
        pass
    fb = _FALLBACK_GRID["MISO"]
    return fb["renewable_pct"], fb["co2_intensity"]


def get_grid_status() -> dict:
    """Gets current electricity grid pricing period, rate, and grid mix.

    Returns:
        Dictionary with period, rate_per_kwh, renewable_pct, co2_intensity.
    """
    hour = datetime.now().hour
    if 14 <= hour < 19:
        period, rate = "peak", 0.28
    elif (7 <= hour < 14) or (19 <= hour < 23):
        period, rate = "mid-peak", 0.18
    else:
        period, rate = "off-peak", 0.10
    renewable_pct, co2_intensity = _fetch_eia_grid_mix()
    return {"period": period, "rate_per_kwh": rate,
            "renewable_pct": renewable_pct, "co2_intensity": co2_intensity}


_NREL_PVWATTS_CACHE = None


def _fetch_nrel_pvwatts():
    """Cached NREL PVWatts v8 typical-year fetch for the 72 kW Ann Arbor array."""
    global _NREL_PVWATTS_CACHE
    if _NREL_PVWATTS_CACHE is not None:
        return _NREL_PVWATTS_CACHE
    try:
        r = requests.get(
            "https://developer.nrel.gov/api/pvwatts/v8.json",
            params={"api_key": NREL_API_KEY, "system_capacity": COMMUNITY_CAPACITY_KW,
                    "module_type": 0, "losses": 14, "array_type": 1,
                    "tilt": 30, "azimuth": 180, "lat": LAT, "lon": LON},
            timeout=15,
        ).json()
        _NREL_PVWATTS_CACHE = r.get("outputs", {})
        return _NREL_PVWATTS_CACHE
    except Exception:
        return None


def get_nrel_pvwatts_baseline() -> dict:
    """Gets NREL PVWatts typical-year solar baseline for the 72 kW community array.

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
    _month_idx = _now.month - 1
    _monthly = out.get("ac_monthly") or [None] * 12
    _current_month_kwh = _monthly[_month_idx] if _month_idx < len(_monthly) else None
    _days = monthrange(_now.year, _now.month)[1]
    _avg_kw = round(_current_month_kwh / (_days * 24), 2) if _current_month_kwh else None
    return {
        "annual_kwh": round(out["ac_annual"]) if out.get("ac_annual") else None,
        "current_month_typical_kwh": round(_current_month_kwh) if _current_month_kwh else None,
        "current_month_typical_kw_avg": _avg_kw,
        "capacity_kw": COMMUNITY_CAPACITY_KW,
        "source": "nrel-pvwatts-v8",
    }


TOOLS = [get_weather, get_solar_production, get_battery_state, get_grid_status, get_nrel_pvwatts_baseline]
TOOL_MAP = {fn.__name__: fn for fn in TOOLS}

# --- Tool-call extraction (byte-equivalent to inference.py Cell 4) -----------
# Base Gemma 4 E4B (without SolarHive's When2Call fine-tune corpus) sometimes
# drifts into Python-syntax tool calls — `call:fn(arg=val)` instead of the
# native `<|tool_call>call:fn{arg:val}<tool_call|>`. The Cactus / Ollama /
# llama.cpp / cloud tiers don't drift because they run fine-tuned weights;
# the LiteRT-LM tier runs the upstream pre-converted base bundle, so we accept
# either format here and let the prompt nudge the model toward the native form.
_TOOL_CALL_WRAPPED_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL)
_TOOL_CALL_BARE_RE = re.compile(r"call:(\w+)\{([^}]*)\}")
# Python-syntax variants (curly missing, parens used)
_TOOL_CALL_PYPAREN_WRAPPED_RE = re.compile(r"<\|tool_call>call:(\w+)\((.*?)\)<tool_call\|>", re.DOTALL)
_TOOL_CALL_PYPAREN_BARE_RE = re.compile(r"call:(\w+)\(([^)]*)\)")

# Pattern-name lookup so the agentic loop can record which regex fired per
# round. Surfaces curly-vs-parens drift distribution in the verdict block —
# direct evidence of base E4B's tool-call-format consistency.
_PATTERN_NAMES = {
    id(_TOOL_CALL_WRAPPED_RE): "curly-wrapped",
    id(_TOOL_CALL_BARE_RE): "curly-bare",
    id(_TOOL_CALL_PYPAREN_WRAPPED_RE): "parens-wrapped",
    id(_TOOL_CALL_PYPAREN_BARE_RE): "parens-bare",
}

# Gemma native arg format: key:<|"|>str<|"|> | key:true|false|null | key:-?\d+\.?\d*
_ARG_FIELD_RE = re.compile(
    r'(\w+)\s*:\s*(?:<\|"\|>([^<]*)<\|"\|>|(true|false|null)|(-?\d+\.?\d*))'
)
# Python-paren arg format: key="str" | key='str' | key=True|False|None | key=-?\d+\.?\d*
_ARG_FIELD_PYPAREN_RE = re.compile(
    r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(True|False|None)|(-?\d+\.?\d*))"""
)


def _extract_tool_calls(raw):
    """Return (matches, pattern_name) where matches = [(fn_name, args_str), ...].

    Tries 4 patterns in priority order:
      1. <|tool_call>call:fn{...}<tool_call|>     — Gemma 4 native wrapped (preferred)
      2. call:fn{...}                              — Gemma 4 bare (thinking-mode-stripped)
      3. <|tool_call>call:fn(...)<tool_call|>     — Python-paren wrapped (base-E4B drift)
      4. call:fn(...)                              — Python-paren bare (base-E4B drift)

    Returns the matches from the first pattern that finds anything. Patterns
    3 and 4 exist to absorb base Gemma 4's occasional Python-syntax drift on
    the LiteRT-LM tier (where we run base, not fine-tuned, weights — see
    `_TOOL_SCHEMA_HINT` and the April 2, 2026 Google Developers Blog post).

    The function also returns the pattern name (`curly-wrapped`, `curly-bare`,
    `parens-wrapped`, `parens-bare`, or `None` if no pattern matched). The
    benchmark loop and the verdict block use this to log curly-vs-parens
    drift distribution — direct evidence of base E4B's tool-call-format
    consistency on the LiteRT-LM runtime.
    """
    for pat in (_TOOL_CALL_WRAPPED_RE, _TOOL_CALL_BARE_RE,
                _TOOL_CALL_PYPAREN_WRAPPED_RE, _TOOL_CALL_PYPAREN_BARE_RE):
        matches = [(m.group(1), m.group(2).strip()) for m in pat.finditer(raw)]
        if matches:
            return matches, _PATTERN_NAMES[id(pat)]
    return [], None


def _parse_tool_args(args_str):
    """Convert a tool-call arg string into a Python dict.

    Auto-detects format: tries Gemma native (`key:value` with `<|"|>` string
    delimiters and lowercase bools) first; falls back to Python-paren
    (`key=value` with regular quotes and capitalized bools) if Gemma yields
    nothing. The two regexes are mutually exclusive on separator (`:` vs `=`)
    so there's no double-counting risk."""
    # Try Gemma native
    args = {}
    for m in _ARG_FIELD_RE.finditer(args_str):
        key = m.group(1)
        s, bn, n = m.group(2), m.group(3), m.group(4)
        if s is not None:
            args[key] = s
        elif bn:
            args[key] = {"true": True, "false": False, "null": None}[bn]
        elif n:
            args[key] = float(n) if "." in n else int(n)
    if args:
        return args
    # Fallback: Python-paren style
    for m in _ARG_FIELD_PYPAREN_RE.finditer(args_str):
        key = m.group(1)
        s_dq, s_sq, bn, n = m.group(2), m.group(3), m.group(4), m.group(5)
        if s_dq is not None:
            args[key] = s_dq
        elif s_sq is not None:
            args[key] = s_sq
        elif bn:
            args[key] = {"True": True, "False": False, "None": None}[bn]
        elif n:
            args[key] = float(n) if "." in n else int(n)
    return args


import inspect as _inspect


def _safe_tool_call(fn, args):
    """Drop hallucinated kwargs the function doesn't accept; never crash the loop."""
    sig = _inspect.signature(fn)
    if any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**args)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in args.items() if k in accepted}
    return fn(**filtered)


# --- Tool-bench smoke test ---------------------------------------------------
print("   Smoke-testing 5 tools:")
for fn in TOOLS:
    try:
        result = fn() if fn.__name__ != "get_solar_production" else fn(30, 75)
        print(f"     {fn.__name__:30s} → ok")
    except Exception as e:
        print(f"     {fn.__name__:30s} → ❌ {type(e).__name__}")

"""## 6: Agentic loop on LiteRT-LM Conversation

Runtime-agnostic agentic loop using prompt-engineered tool definitions
embedded in the system prompt. This mirrors the Ollama track's
`_ollama_agentic_loop` pattern (already 10/10 on Surface Pro 8) — Gemma 4's
native function calling is pre-trained, not fine-tune-dependent, so the
same tool-call format works on any runtime that exposes the model's raw
output.

The loop:
1. Build a system prompt including the 5 tool signatures and one-shot example
2. Open `with engine.create_conversation() as convo:`
3. send_message_async(user_query) — collect raw tokens
4. _extract_tool_calls(raw); if any, execute via _safe_tool_call and append
   results as the next conversation turn
5. Repeat up to max_rounds=3; emit final answer when no more calls
"""

# System prompt mirroring `solarhive_inference.py` SYSTEM_PROMPT (doubled per
# Leviathan/Kalman/Matias 2024 — repeat to improve causal-LM instruction
# following, +47/70 on Google's internal benchmark with no latency cost).
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

# Explicit tool-call format with anti-pattern callout. Base E4B sometimes
# emits `call:fn(arg=val)` (Python paren syntax) instead of the native
# `<|tool_call>call:fn{arg:val}<tool_call|>` curly form. The fine-tuned
# variants (Cactus / Ollama / llama.cpp / cloud) don't drift because the
# When2Call corpus conditions on the curly format; LiteRT-LM runs base
# weights, so we nudge the model explicitly here AND accept either format
# in the parser as belt-and-suspenders.
_TOOL_SCHEMA_HINT = (
    "\n\n## Tool calling — STRICT FORMAT REQUIRED"
    "\n\nWhen real-time data is needed, emit a tool call EXACTLY in this format:"
    "\n  <|tool_call>call:function_name{arg1:value1,arg2:value2}<tool_call|>"
    "\nUse curly braces { }, NOT parentheses ( )."
    "\nUse colon : between key and value, NOT equals =."
    "\nUse <|\"|> ... <|\"|> for string values, NOT regular quotes."
    "\n"
    "\n### Correct format examples"
    "\n  <|tool_call>call:get_grid_status{}<tool_call|>"
    "\n  <|tool_call>call:get_solar_production{clouds_pct:30,temp_f:75}<tool_call|>"
    "\n  <|tool_call>call:get_weather{location:<|\"|>Ann Arbor, MI<|\"|>}<tool_call|>"
    "\n  <|tool_call>call:get_battery_state{}<tool_call|>"
    "\n  <|tool_call>call:get_nrel_pvwatts_baseline{}<tool_call|>"
    "\n"
    "\n### INCORRECT — do NOT emit any of these"
    "\n  call:get_solar_production(clouds_pct=0, temp_f=70)   ← WRONG: parens, equals"
    "\n  get_solar_production(0, 70)                           ← WRONG: missing call: prefix"
    "\n  {\"name\": \"get_solar_production\", \"args\": {...}}   ← WRONG: JSON format"
    "\n  get_grid_status                                       ← WRONG: missing curly braces"
    "\n"
    "\n### Available tools"
    "\n- get_weather() → temp_f, clouds_pct, description, wind_mph, humidity_pct, sunrise, sunset"
    "\n- get_solar_production(clouds_pct, temp_f) → production_kw, capacity_kw, efficiency_pct, ghi_wm2"
    "\n- get_battery_state() → soc_pct, kwh_stored, capacity_kwh, charging"
    "\n- get_grid_status() → period, rate_per_kwh, renewable_pct, co2_intensity"
    "\n- get_nrel_pvwatts_baseline() → annual_kwh, current_month_typical_kwh, current_month_typical_kw_avg"
    "\n"
    "\nAfter receiving tool results, summarize the answer in 2-4 sentences for the user."
)

SYSTEM_PROMPT = (_UNIFIED_SYSTEM_BODY + _TOOL_SCHEMA_HINT) * 2


def _send_collect(conv, message, timeout_s=180):
    """Send `message` to a LiteRT-LM Conversation, collect all chunks, return raw text + timing.

    180s is the upper bound for SolarHive UX (anything slower isn't usable on
    a phone) — a safe ceiling for the longer benchmark prompts (renewable-mix
    summary, one-paragraph community briefing) without masking real problems.
    """
    t_send = time.perf_counter()
    first_t = None
    chunks = []
    for chunk in conv.send_message_async(message):
        if first_t is None:
            first_t = time.perf_counter()
        if isinstance(chunk, dict):
            content = chunk.get("content", [])
            if content and isinstance(content, list) and isinstance(content[0], dict):
                chunks.append(content[0].get("text", ""))
            else:
                chunks.append(str(chunk))
        else:
            chunks.append(str(chunk))
        if time.perf_counter() - t_send > timeout_s:
            break
    t_end = time.perf_counter()
    return {
        "raw": "".join(chunks),
        "ttft_s": round((first_t or t_end) - t_send, 2),
        "total_s": round(t_end - t_send, 2),
        "tokens_approx": sum(len(c.split()) for c in chunks),
    }


def litertlm_agentic_loop(engine, question, max_rounds=3):
    """Run a SolarHive agentic loop on a LiteRT-LM Engine. Returns transcript dict.

    Transcript shape:
      - `rounds`: per-round timing + raw output
      - `tool_calls`: list of executed tool calls with their args + results
      - `patterns`: list of pattern names (one per tool-call round) so the
        verdict can show curly-vs-parens distribution.
      - `final`: cleaned final answer
      - `ttft_s`, `total_s`: end-to-end timing
    """
    transcript = {"question": question, "rounds": [], "final": None,
                  "total_s": 0.0, "ttft_s": None, "tool_calls": [], "patterns": []}
    t_start = time.perf_counter()

    with engine.create_conversation() as conv:
        # Round 0: prime with system prompt + question
        primed = f"{SYSTEM_PROMPT}\n\nUser: {question}"
        result = _send_collect(conv, primed)
        if transcript["ttft_s"] is None:
            transcript["ttft_s"] = result["ttft_s"]
        transcript["rounds"].append({"role": "assistant", **result})

        for _round in range(max_rounds):
            calls, pattern = _extract_tool_calls(result["raw"])
            if not calls:
                transcript["final"] = result["raw"].strip()
                break
            transcript["patterns"].append(pattern)

            tool_results = []
            for fn_name, args_str in calls:
                fn = TOOL_MAP.get(fn_name)
                if not fn:
                    tool_results.append((fn_name, {"error": "unknown_tool"}))
                    continue
                args = _parse_tool_args(args_str)
                try:
                    out = _safe_tool_call(fn, args)
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}
                tool_results.append((fn_name, out))
                transcript["tool_calls"].append({"fn": fn_name, "args": args, "result": out})

            # Feed results back as next turn (one message per call, JSON content)
            tool_response_msg = "Tool results:\n" + "\n".join(
                f"- {fn}: {json.dumps(out)}" for fn, out in tool_results
            )
            result = _send_collect(conv, tool_response_msg)
            transcript["rounds"].append({"role": "tool_response", **result})
        else:
            # Max rounds exceeded with calls still pending
            transcript["final"] = result["raw"].strip()

    transcript["total_s"] = round(time.perf_counter() - t_start, 2)
    return transcript


print("   Agentic-loop helper defined.")
print("   System prompt length:", len(SYSTEM_PROMPT), "chars")

"""## 7: 8-prompt SolarHive benchmark

Same prompt set used by the Run 6 cloud benchmark (8/8 26B A4B + 10/10 E4B
Ollama on the canonical Sol B subset). Q&A score, prefill/decode timing,
and tool-call success rate captured for the verdict block.
"""

BENCHMARK_PROMPTS = [
    "What's the current grid pricing period and rate?",
    "How much solar are we generating right now?",
    "What's the battery state of charge?",
    "Should we run laundry now or wait for off-peak?",
    "How does today's production compare to typical?",
    "What's the weather looking like for solar today?",
    "Is the grid mostly renewable right now?",
    "Give me a one-paragraph energy briefing for the community.",
]

bench_results = []
_BENCH_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "bench_results.json")


def _checkpoint_bench(results):
    """Persist `bench_results` after each prompt so a kernel crash leaves evidence."""
    try:
        with open(_BENCH_CHECKPOINT_PATH, "w") as _f:
            json.dump(results, _f, indent=2, default=str)
    except Exception:
        pass  # best-effort; never let checkpointing break the run


if _LITERT_LM_AVAILABLE and LITERTLM_PATH and _warmup["ok"]:
    print(f"   Running {len(BENCHMARK_PROMPTS)} prompts...")
    print(f"   Checkpoint: {_BENCH_CHECKPOINT_PATH}")
    with litert_lm.Engine(LITERTLM_PATH) as _engine_b:
        for i, prompt in enumerate(BENCHMARK_PROMPTS, 1):
            try:
                t = litertlm_agentic_loop(_engine_b, prompt, max_rounds=3)
                # Compute decode-only throughput from the final round (excludes prefill/TTFT)
                _last_round = t["rounds"][-1] if t["rounds"] else {}
                _tok = _last_round.get("tokens_approx", 0)
                _decode_window = max(0.01,
                    (_last_round.get("total_s", 0) or 0) - (_last_round.get("ttft_s", 0) or 0))
                _decode_tps = (round(max(0, _tok - 1) / _decode_window, 2)
                               if _tok > 1 else None)
                _patterns = t.get("patterns", [])
                bench_results.append({
                    "idx": i, "prompt": prompt, "ok": bool(t["final"]),
                    "ttft_s": t["ttft_s"], "total_s": t["total_s"],
                    "tool_calls": [tc["fn"] for tc in t["tool_calls"]],
                    "patterns": _patterns,
                    "decode_tps": _decode_tps,
                    "final_preview": (t["final"] or "")[:240],
                })
                print(f"     [{i}/{len(BENCHMARK_PROMPTS)}] {t['total_s']}s "
                      f"({len(t['tool_calls'])} tools, "
                      f"{','.join(_patterns) or 'no-tools'}, "
                      f"~{_decode_tps or '—'} dec tps) — {(t['final'] or '')[:80]}")
            except Exception as e:
                bench_results.append({"idx": i, "prompt": prompt, "ok": False,
                                      "error": f"{type(e).__name__}: {e}"})
                print(f"     [{i}/{len(BENCHMARK_PROMPTS)}] ❌ {type(e).__name__}: {e}")
            # Checkpoint after every prompt — survives mid-run kernel death
            _checkpoint_bench(bench_results)
else:
    print("   ⏭️ skipped (prerequisite missing)")

# Aggregate score
_qa_score = sum(1 for r in bench_results if r.get("ok"))
print(f"\n   Q&A score: {_qa_score}/{len(BENCHMARK_PROMPTS)}")
print(f"   Checkpoint written: {_BENCH_CHECKPOINT_PATH}")

"""## 8: MTP demonstration (Multi-Token Prediction)

Multi-Token Prediction is Google's speculative-decoding architecture for
Gemma 4 — a lightweight "drafter" model paired with the target model
emits multiple tokens per forward pass for up to 3x decode speedup
without quality degradation. References:

- "Accelerating Gemma 4: faster inference with multi-token prediction
  drafters" (Google blog, May 5, 2026):
  https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- Multi-Token Prediction overview:
  https://ai.google.dev/gemma/docs/mtp/overview
- Multi-Token Prediction implementation (HF Transformers code pattern):
  https://ai.google.dev/gemma/docs/mtp/mtp
- Apr 2, 2026 Google Developers Blog (originally claimed ">2x faster
  decode on mobile GPUs"):
  https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/

**Architectural note (per the May 5 blog and the MTP implementation
page).** MTP is implemented as a **separate drafter model artifact**
paired with the target — not a runtime kwarg flipped on a single bundle.
The HF Transformers code pattern documented at
`ai.google.dev/gemma/docs/mtp/mtp` is:

    target = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it", ...)
    assistant = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it-assistant", ...)
    target.generate(..., assistant_model=assistant)

The drafter naming convention is pinned: `<target_id>-assistant`. For
LiteRT-LM specifically, the analogous paired-drafter Python API is not
yet documented as of the citations above; once a `litert-community/
gemma-4-E4B-it-assistant-litert-lm` (or similar) artifact appears, the
paired-load path is the right next step.

This cell probes the public LiteRT-LM Python API surface for any MTP
single-call enable kwarg (in case a same-bundle path exists), and
graceful-skips if none is exposed. The graceful-skip is the expected
outcome on the upstream pre-converted target bundle alone; the verdict
records "drafter pairing required per the May 5 Google blog
architecture" rather than the misleading "MTP API shape not exposed."
"""

mtp_results = {"available": False, "enable_path": None, "decode_speedup": None,
               "baseline_decode_s": None, "mtp_decode_s": None, "error": None}

if _LITERT_LM_AVAILABLE and LITERTLM_PATH and _warmup["ok"]:
    # Discover MTP API surface
    _mtp_candidates = [s for s in dir(litert_lm) if "mtp" in s.lower()
                       or "multi" in s.lower() or "speculative" in s.lower()]
    print(f"   MTP-related symbols in litert_lm: {_mtp_candidates}")

    try:
        # Attempt 1: Engine-level kwarg (most likely shape per common SDK conventions)
        _mtp_prompt = BENCHMARK_PROMPTS[4]  # "How does today's production compare to typical?"
        # Baseline (no MTP)
        with litert_lm.Engine(LITERTLM_PATH) as _engine_base:
            with _engine_base.create_conversation() as _c:
                _r_base = _send_collect(_c, _mtp_prompt)
        mtp_results["baseline_decode_s"] = _r_base["total_s"]

        # MTP-enabled: try several plausible API shapes
        _mtp_engine = None
        for _attempt in [
            lambda: litert_lm.Engine(LITERTLM_PATH, enable_mtp=True),
            lambda: litert_lm.Engine(LITERTLM_PATH, multi_token_prediction=True),
            lambda: litert_lm.Engine(LITERTLM_PATH, speculative_decoding=True),
        ]:
            try:
                _mtp_engine = _attempt()
                mtp_results["enable_path"] = _attempt.__qualname__
                break
            except Exception:
                continue

        if _mtp_engine is None:
            mtp_results["error"] = (
                "MTP requires a paired drafter artifact (e.g., "
                "`google/gemma-4-E4B-it-assistant`) alongside the target — "
                "see https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/ "
                "and https://ai.google.dev/gemma/docs/mtp/mtp. No drafter is "
                "loaded by this notebook, and no single-call enable kwarg is "
                "exposed by the public LiteRT-LM Python API for the "
                "same-bundle path."
            )
            print(f"   ⏭️ MTP probe — drafter pairing required, recording in verdict")
        if _mtp_engine is not None:
            with _mtp_engine as _engine_mtp:
                with _engine_mtp.create_conversation() as _c:
                    _r_mtp = _send_collect(_c, _mtp_prompt)
            mtp_results["mtp_decode_s"] = _r_mtp["total_s"]
            mtp_results["decode_speedup"] = round(
                mtp_results["baseline_decode_s"] / max(0.01, mtp_results["mtp_decode_s"]), 2
            )
            mtp_results["available"] = True
            print(f"   Baseline:  {mtp_results['baseline_decode_s']}s")
            print(f"   MTP:       {mtp_results['mtp_decode_s']}s")
            print(f"   Speedup:   {mtp_results['decode_speedup']}×")
        # If `_mtp_engine` is None, the drafter-pairing-required message
        # set above is already in `mtp_results["error"]`.
    except Exception as e:
        mtp_results["error"] = f"{type(e).__name__}: {e}"
        print(f"   ❌ MTP probe failed: {mtp_results['error']}")
else:
    print("   ⏭️ skipped (prerequisite missing)")

"""## 9: Multi-modal VQA probe (sky-photo analysis)

E4B `.litertlm` is documented to support vision per the model card and
[`gemma_4 model_card_4`](https://ai.google.dev/gemma/docs/core/model_card_4).
The LiteRT-LM Python docs reference a "Multi-Modality" section. Defensive:
discover the image-input API at runtime, attempt a single sky-photo VQA,
record the transcript.
"""

vqa_result = {"attempted": False, "ok": False, "response": None, "error": None,
              "image_source": None}

if _LITERT_LM_AVAILABLE and LITERTLM_PATH and _warmup["ok"]:
    # Try to fetch one Ann Arbor sky photo from the multimodal dataset
    try:
        from huggingface_hub import hf_hub_download
        _img_path = hf_hub_download(
            repo_id="Truthseeker87/solarhive-community-solar-multimodal",
            filename="annarbor_sky_03.jpeg",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        vqa_result["image_source"] = _img_path
    except Exception as e:
        # Fallback: any local image, or skip
        _img_path = None
        vqa_result["error"] = f"image-fetch: {type(e).__name__}: {e}"

    if _img_path:
        try:
            from PIL import Image as _Image
            _img = _Image.open(_img_path).convert("RGB")
            _vqa_prompt = (
                "How will this sky affect our community solar production? "
                "Answer in 2-3 sentences with specific cloud-coverage estimate."
            )
            # Attempt LiteRT-LM multimodal API — defensive across plausible shapes
            with litert_lm.Engine(LITERTLM_PATH) as _engine_v:
                with _engine_v.create_conversation() as _c:
                    _vqa_chunks = []
                    _sent = False
                    for _attempt in [
                        lambda: _c.send_message_async(_vqa_prompt, image=_img),
                        lambda: _c.send_message_async([{"text": _vqa_prompt}, {"image": _img}]),
                        lambda: _c.send_message_async({"text": _vqa_prompt, "image": _img}),
                    ]:
                        try:
                            for _chk in _attempt():
                                if isinstance(_chk, dict):
                                    _content = _chk.get("content", [])
                                    if _content and isinstance(_content, list) and isinstance(_content[0], dict):
                                        _vqa_chunks.append(_content[0].get("text", ""))
                                else:
                                    _vqa_chunks.append(str(_chk))
                            _sent = True
                            break
                        except Exception:
                            continue
                    if _sent and _vqa_chunks:
                        vqa_result["response"] = "".join(_vqa_chunks).strip()
                        vqa_result["ok"] = bool(vqa_result["response"])
                        vqa_result["attempted"] = True
                        print(f"   ✅ VQA response: {vqa_result['response'][:200]}")
                    else:
                        vqa_result["error"] = "no multimodal API shape accepted"
                        print(f"   ⚠️ multimodal API shape not discovered")
        except Exception as e:
            vqa_result["error"] = f"{type(e).__name__}: {e}"
            print(f"   ❌ VQA probe failed: {vqa_result['error']}")
    else:
        print(f"   ⏭️ no image available; recording skip in verdict")
else:
    print("   ⏭️ skipped (prerequisite missing)")

"""## 10: When2Call sub-bench (3 probes)

3 held-out probes from Ross et al. 2025 *When2Call: When (not) to Call Tools*
(arXiv:2504.18851), validating coverage of 3 of the 4 failure-mode categories
the paper documents — same probe set as `solarhive_inference.py` §11b.

  (b) well-specified in-scope → expect correct tool call
  (c) under-specified → expect follow-up question, not auto-filled defaults
  (d) out-of-scope → expect refusal + redirect, not hallucinated tool
"""

W2C_PROBES = [
    {"category": "b", "prompt": "What's the current grid rate?",
     "expect_tool": "get_grid_status",
     "expect_behavior": "should call get_grid_status with no args"},
    {"category": "c", "prompt": "How much will a 10 kW array produce today?",
     "expect_tool": None,
     "expect_behavior": "should ask for location (under-specified) — no auto-call"},
    {"category": "d", "prompt": "What's the current air quality?",
     "expect_tool": None,
     "expect_behavior": "should refuse + redirect (no air-quality tool exists)"},
]

w2c_results = []

if _LITERT_LM_AVAILABLE and LITERTLM_PATH and _warmup["ok"]:
    print(f"   Running {len(W2C_PROBES)} When2Call probes...")
    with litert_lm.Engine(LITERTLM_PATH) as _engine_w:
        for probe in W2C_PROBES:
            try:
                t = litertlm_agentic_loop(_engine_w, probe["prompt"], max_rounds=2)
                tools_called = [tc["fn"] for tc in t["tool_calls"]]

                # Score per category
                if probe["category"] == "b":
                    passed = probe["expect_tool"] in tools_called
                elif probe["category"] == "c":
                    # Under-specified: should NOT auto-call solar/weather without asking
                    passed = (not tools_called) and any(
                        kw in (t["final"] or "").lower()
                        for kw in ["where", "location", "city", "which", "specify"]
                    )
                else:  # d
                    # Out-of-scope: should NOT hallucinate a tool
                    passed = not tools_called and any(
                        kw in (t["final"] or "").lower()
                        for kw in ["don't", "cannot", "no tool", "not available", "unable", "outside"]
                    )

                w2c_results.append({
                    "category": probe["category"], "prompt": probe["prompt"],
                    "tools_called": tools_called, "passed": passed,
                    "final_preview": (t["final"] or "")[:240],
                })
                print(f"     ({probe['category']}) {'✅' if passed else '❌'}  "
                      f"tools={tools_called}  final='{(t['final'] or '')[:80]}'")
            except Exception as e:
                w2c_results.append({"category": probe["category"], "prompt": probe["prompt"],
                                    "passed": False, "error": str(e)})
                print(f"     ({probe['category']}) ❌ {type(e).__name__}")
else:
    print("   ⏭️ skipped (prerequisite missing)")

_w2c_score = sum(1 for r in w2c_results if r.get("passed"))
print(f"\n   W2C-3 score: {_w2c_score}/3")

"""## 11: Phase 1 verdict block

Aggregates Cell 4 (warm-up) + Cell 7 (Q&A bench) + Cell 8 (MTP) + Cell 9
(VQA) + Cell 10 (W2C) into a single structured verdict comparable to the
6-variant table in `solarhive_inference.py` §13. JSON-dumped so it can be
copy-pasted into the README's variant-comparison table.
"""

verdict = {
    "track_objective": "LiteRT-LM as the Gemma 4 mobile-edge inference runtime for SolarHive",
    "runtime": "LiteRT-LM Python (CPU)",
    "model": "Gemma 4 E4B base (.litertlm, upstream pre-converted)",
    "validation_platform": "Colab Pro CPU + High-RAM (Linux x86_64)",
    "validation_role": "proxy for mobile-edge deployment matrix below",
    "mobile_edge_portability": {
        "android_kotlin": "same .litertlm bundle, LiteRT-LM Android Kotlin SDK",
        "ios_macos_cpp": "same .litertlm bundle, LiteRT-LM iOS/macOS C++ SDK (Swift APIs upcoming)",
        "browser_mobile": "companion .task bundle via MediaPipe Tasks Web on phone Chrome / Safari (WebGPU) — see web-litert/",
        "rpi5_jetson_python": "same SDK as this validation, native ARM Linux",
        "windows_python": "same SDK, Windows support upcoming per LiteRT-LM landing",
        "contract": "cross-platform LiteRT-LM bundle + agentic-loop code shape; what runs here ports unchanged",
    },
    "litert_lm_version": _LITERT_LM_VERSION,
    "warmup": {
        "ok": _warmup["ok"],
        "cold_start_s": _warmup["cold_start_s"],
        "ttft_s": _warmup["ttft_s"],
        "total_s": _warmup["total_s"],
    },
    "qa_bench": {
        "score": f"{_qa_score}/{len(BENCHMARK_PROMPTS)}",
        "prompts": len(BENCHMARK_PROMPTS),
        "median_ttft_s": round(float(np.median([r["ttft_s"] for r in bench_results if r.get("ok") and r.get("ttft_s")])), 2)
            if any(r.get("ok") for r in bench_results) else None,
        "median_total_s": round(float(np.median([r["total_s"] for r in bench_results if r.get("ok") and r.get("total_s")])), 2)
            if any(r.get("ok") for r in bench_results) else None,
        "tool_call_count": sum(len(r.get("tool_calls", [])) for r in bench_results),
    },
    "mtp": mtp_results,
    "vqa": {
        "attempted": vqa_result["attempted"], "ok": vqa_result["ok"],
        "response_preview": (vqa_result["response"] or "")[:200] if vqa_result.get("response") else None,
        "error": vqa_result["error"],
    },
    "w2c_3": {
        "score": f"{_w2c_score}/3",
        "by_category": {r["category"]: r["passed"] for r in w2c_results},
    },
}

# --- Verified mobile-edge benchmarks (Gemma 4 E4B, .litertlm) ---------------
# Source: https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
# Benchmark protocol: 1024 prefill tokens / 256 decode tokens / 2048 ctx
# All numbers verbatim from the model-card "Performance" section. This is the
# ground truth for the cross-platform contract claim — same .litertlm bundle,
# multi-platform SDK family.
HF_CARD_BENCHMARKS = [
    # (target_label, prefill_tps, decode_tps, ttft_s, ram_mb, model_size_mb)
    ("Pi 5 16GB CPU (ARM)",          51,    3.2,  20.5,   3069, 3654),
    ("Linux ARM 2.3-2.8GHz CPU",     82,   17.5,  12.6,   3139, 3654),
    ("Linux RTX 4090 GPU",         7260,   91.2,   0.2,   1119, 3654),
    ("Windows Intel LunarLake CPU",  173,  16.8,   5.98,  9372, 3654),
    ("Windows Intel LunarLake GPU", 1202,  25.13,  0.89,  7147, 3654),
    ("macOS M4 Max CPU",             277,  27.0,   3.7,    890, 3654),
    ("macOS M4 Max GPU",            2560, 101.1,   0.4,   3217, 3654),
    ("Android S26 Ultra CPU",        195,  17.7,   5.3,   3283, 3654),
    ("Android S26 Ultra GPU",       1293,  22.1,   0.8,    710, 3654),
    ("iPhone 17 Pro CPU",            159,   9.7,   6.5,    961, 3654),
    ("iPhone 17 Pro GPU",           1189,  25.1,   0.9,   3380, 3654),
    ("Web Chrome (M4 Max GPU)",     1598,  44.4,   None,  None, 2964),
]

# Compute our measured decode tps from the warmup (single shot, easy to compare)
# AND the median over the bench (more representative).
_warmup_dec_tps = None
if _warmup.get("ok") and _warmup.get("decode_tokens", 0) > 1:
    _w_decode_window = max(0.01,
        (_warmup.get("total_s") or 0) - (_warmup.get("ttft_s") or 0))
    _warmup_dec_tps = round(
        max(0, _warmup["decode_tokens"] - 1) / _w_decode_window, 2)

_bench_dec_tps_list = [r.get("decode_tps") for r in bench_results
                        if isinstance(r.get("decode_tps"), (int, float))]
_bench_dec_tps_median = (round(float(np.median(_bench_dec_tps_list)), 2)
                          if _bench_dec_tps_list else None)

# Pattern distribution across all benchmark rounds
_all_patterns = [p for r in bench_results for p in (r.get("patterns") or [])]
_pattern_counts = {p: _all_patterns.count(p) for p in set(_all_patterns)}

verdict["mobile_edge_benchmarks"] = {
    "_source": "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm",
    "_protocol": "1024 prefill / 256 decode / 2048 ctx",
    "measured_x86_linux_decode_tps_warmup": _warmup_dec_tps,
    "measured_x86_linux_decode_tps_median_bench": _bench_dec_tps_median,
    "tool_call_pattern_distribution": _pattern_counts,
    "published_targets": [
        {"target": label, "prefill_tps": pf, "decode_tps": dec, "ttft_s": ttft,
         "ram_mb": ram, "model_size_mb": ms}
        for (label, pf, dec, ttft, ram, ms) in HF_CARD_BENCHMARKS
    ],
}

print("=" * 72)
print("  Phase 1 Verdict — LiteRT-LM Python edge runtime on Gemma 4 E4B")
print("=" * 72)
print(json.dumps(verdict, indent=2, default=str))
print("=" * 72)

# Visual cross-platform benchmark table (more readable than the JSON dump)
print()
print("=" * 72)
print("  Mobile-edge benchmarks — Gemma 4 E4B on .litertlm (same bundle)")
print("  Source: huggingface.co/litert-community/gemma-4-E4B-it-litert-lm")
print("  Protocol: 1024 prefill / 256 decode / 2048 ctx")
print("=" * 72)
print(f"  {'Target':<32s}  {'Prefill':>9s}  {'Decode':>9s}  {'TTFT':>7s}")
print(f"  {'':<32s}  {'tok/s':>9s}  {'tok/s':>9s}  {'(s)':>7s}")
print("  " + "-" * 64)
for label, pf, dec, ttft, ram, ms in HF_CARD_BENCHMARKS:
    ttft_str = f"{ttft:.2f}" if isinstance(ttft, (int, float)) else "—"
    print(f"  {label:<32s}  {pf:>9}  {dec:>9}  {ttft_str:>7}")
print("  " + "-" * 64)
if _warmup_dec_tps is not None:
    _ttft_w = _warmup.get("ttft_s")
    print(f"  {'(this run) Colab x86_64 CPU':<32s}  {'—':>9}  "
          f"{_warmup_dec_tps:>9}  {(_ttft_w if _ttft_w is not None else '—'):>7}"
          f"   ← warmup")
if _bench_dec_tps_median is not None:
    print(f"  {'(this run) Colab x86_64 CPU':<32s}  {'—':>9}  "
          f"{_bench_dec_tps_median:>9}  {'—':>7}   ← median over bench")
print("=" * 72)

# SolarHive UX latency proxy — what a 100-token answer feels like per target
print()
print("  SolarHive UX latency proxy — 100-token short imperative answer:")
print("  (TTFT + 100 tok ÷ decode_tps; lower = better mobile UX)")
print("  " + "-" * 64)
for label, pf, dec, ttft, ram, ms in HF_CARD_BENCHMARKS:
    # Web row has no TTFT (separate init time of 1.5s); use init as proxy
    _is_web = not isinstance(ttft, (int, float))
    _ttft_eff = 1.5 if _is_web else ttft
    _ttft_label = "init" if _is_web else "TTFT"
    _total_100 = round(_ttft_eff + (100 / dec), 1)
    print(f"  {label:<32s}  ~{_total_100:>5.1f}s  "
          f"({_ttft_label} {_ttft_eff}s + 100/{dec})")
print("  " + "-" * 64)
print()
print("  Pattern distribution across this run's tool-call rounds:")
print(f"    {_pattern_counts or '(no tool-call rounds matched)'}")
print("=" * 72)

# Headline summary lines for the README variant-comparison table
print()
print("README variant-comparison row (paste-ready):")
print(
    f"  | LiteRT-LM Python (base E4B, CPU) | "
    f"{verdict['qa_bench']['score']} | "
    f"{verdict['w2c_3']['score']} W2C | "
    f"TTFT {verdict['warmup']['ttft_s']}s | "
    f"MTP {mtp_results['decode_speedup']}× | "
    f"VQA {'✅' if vqa_result['ok'] else '⏭️'} |"
)

"""## Phase 2 (optional, time-boxed) — fine-tuned → `.tflite` conversion probe

Default: `_PHASE2_ENABLED = False` (set in Cell 1). Phase 1 (the LiteRT-LM
runtime demo) is the deliverable; Phase 2 is documented due-diligence that
exercises the generic `ai_edge_torch.convert(model, sample_inputs)` code
path on base then fine-tuned Gemma 4 E4B weights. If Phase 2 produces a
`.tflite`, the LiteRT track entry has a stronger fine-tuned-on-LiteRT
narrative. If it fails, the result is a recorded answer to the natural
"did you try this?" question.

**Source for the conversion approach:** https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/discussions/8
"""

"""## 13: Phase 2a — generic ai-edge-torch convert on BASE Gemma 4 E4B

Sentinel-gated by `_PHASE2_ENABLED` (default False). Approximately 30 minutes
wall-clock when enabled (model load + trace + export).
"""

phase2a = {"enabled": _PHASE2_ENABLED, "ok": False, "tflite_size_mb": None,
           "error": None, "duration_s": None}

if _PHASE2_ENABLED:
    try:
        _t0 = time.perf_counter()
        _sp.check_call([_sys.executable, "-m", "pip", "install", "-q",
                        "ai-edge-torch", "torch", "transformers"])
        import torch
        import ai_edge_torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("   Loading base Gemma 4 E4B...")
        _tok = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", token=HF_TOKEN)
        _model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-4-E4B-it", torch_dtype=torch.bfloat16, token=HF_TOKEN
        )
        _sample = _tok("Validate sequence.", return_tensors="pt")["input_ids"]
        print("   Running ai_edge_torch.convert()...")
        _edge = ai_edge_torch.convert(_model, (_sample,))
        _out_path = "/content/gemma-4-E4B-base-phase2a.tflite"
        _edge.export(_out_path)
        phase2a["tflite_size_mb"] = round(Path(_out_path).stat().st_size / 1e6, 1)
        phase2a["ok"] = True
        phase2a["duration_s"] = round(time.perf_counter() - _t0, 1)
        print(f"   ✅ Phase 2a: produced {phase2a['tflite_size_mb']} MB .tflite "
              f"in {phase2a['duration_s']}s")
    except Exception as e:
        phase2a["error"] = f"{type(e).__name__}: {e}"
        phase2a["duration_s"] = round(time.perf_counter() - _t0, 1) if "_t0" in dir() else None
        print(f"   ❌ Phase 2a failed: {phase2a['error']}")
else:
    print("   ⏭️ skipped (_PHASE2_ENABLED=False — Phase 1 is the deliverable)")

"""## 14: Phase 2b — same convert on FINE-TUNED merged safetensors

Only runs if Phase 2a succeeded AND `_PHASE2_ENABLED=True`. Tests whether
the documented working path for base E4B also works for
`Truthseeker87/solarhive-e4b-ollama` merged safetensors. Success = first
working public path from fine-tuned Gemma 4 → `.tflite`. Failure = expected;
documents the next-layer gap.
"""

phase2b = {"enabled": _PHASE2_ENABLED and phase2a.get("ok", False),
           "ok": False, "tflite_size_mb": None, "error": None, "duration_s": None}

if phase2b["enabled"]:
    try:
        _t0 = time.perf_counter()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import ai_edge_torch

        print("   Resolving fine-tuned merged safetensors...")
        _ft_path = snapshot_download(
            repo_id="Truthseeker87/solarhive-e4b-ollama",
            local_dir="/content/solarhive_e4b_ft",
            allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
            token=HF_TOKEN,
        )
        _tok_ft = AutoTokenizer.from_pretrained(_ft_path, token=HF_TOKEN)
        _model_ft = AutoModelForCausalLM.from_pretrained(
            _ft_path, torch_dtype=torch.bfloat16, token=HF_TOKEN
        )
        _sample_ft = _tok_ft("Validate sequence.", return_tensors="pt")["input_ids"]
        print("   Running ai_edge_torch.convert() on fine-tuned weights...")
        _edge_ft = ai_edge_torch.convert(_model_ft, (_sample_ft,))
        _out_path_ft = "/content/solarhive-e4b-ft-phase2b.tflite"
        _edge_ft.export(_out_path_ft)
        phase2b["tflite_size_mb"] = round(Path(_out_path_ft).stat().st_size / 1e6, 1)
        phase2b["ok"] = True
        phase2b["duration_s"] = round(time.perf_counter() - _t0, 1)
        print(f"   🎉 Phase 2b: produced {phase2b['tflite_size_mb']} MB .tflite "
              f"from FINE-TUNED weights in {phase2b['duration_s']}s")
        print(f"   🎯 Strongest possible LiteRT track upgrade — push to "
              f"Truthseeker87/solarhive-e4b-litert post-hackathon.")
    except Exception as e:
        phase2b["error"] = f"{type(e).__name__}: {e}"
        phase2b["duration_s"] = round(time.perf_counter() - _t0, 1) if "_t0" in dir() else None
        print(f"   ❌ Phase 2b failed: {phase2b['error']}")
else:
    print("   ⏭️ skipped (Phase 2a did not succeed or _PHASE2_ENABLED=False)")

# Final verdict block also captures Phase 2 outcome
verdict["phase2"] = {"a_base_e4b": phase2a, "b_finetuned_e4b": phase2b}
print()
print("=" * 72)
print("  Final Verdict (with Phase 2)")
print("=" * 72)
print(json.dumps(verdict, indent=2, default=str))
print("=" * 72)

"""## What this notebook proves for mobile-edge inference

**This is a cross-platform validation, not a Colab demo.** The LiteRT-LM
contract is "one `.litertlm` bundle, one agentic-loop shape, multi-platform
SDK family." What this notebook validates on Colab Linux x86_64 CPU ports
unchanged to every other LiteRT-LM target:

| Mobile-edge target | What carries over | What changes |
|---|---|---|
| **Android phone** (Kotlin SDK) | `.litertlm` bundle, system prompt, 5 tool definitions, `_extract_tool_calls` / `_parse_tool_args` regex (Kotlin port), When2Call probe set | Engine + Conversation API switch from `litert_lm.Engine` (Python) → `LiteRTLMEngine` (Kotlin); native UI |
| **iPhone / iPad** (C++ SDK; Swift APIs coming) | same bundle, same prompt, same tool-call format | Engine + Conversation API in C++; Swift wrapper as it lands |
| **Browser on phone** (Chrome / Safari / Edge mobile) | same SolarHive prompt + tools; `.task` bundle (companion to `.litertlm`) | MediaPipe Tasks Web via `@mediapipe/tasks-genai`; covered by `web-litert/app.js` in this repository |
| **Raspberry Pi 5 microgrid hub** (ARM Linux Python) | identical SDK, identical bundle, identical code | nothing — same Python source runs |
| **NVIDIA Jetson Orin Nano Super** (ARM Linux Python) | identical SDK, identical bundle, identical code | nothing — same Python source runs |
| **Linux laptop** (x86_64 Python) | identical SDK, identical bundle, identical code | nothing — same Python source runs |
| **Windows laptop** (Python — upcoming per LiteRT-LM landing page) | same SDK, same bundle, same code | wait for the Windows wheel |

The agentic loop, tool-call regex, system prompt, and benchmark methodology
defined in Cells 5–10 are **mobile-portable** — moving from Colab Linux to
Android Kotlin or iOS C++ is an SDK rebinding, not a re-architecture. The
Google Developers Blog post "Bring state-of-the-art agentic skills to the
edge with Gemma 4" (April 2, 2026) is the canonical reference for this
cross-platform contract.

**Why the Pi 5 / Jetson story makes the SolarHive use case credible.** The
microgrid hub reference hardware is a $249 Jetson Orin Nano Super running on
solar — the SolarHive brain runs on the energy infrastructure it advises.
The verdict numbers from this notebook (Q&A score, MTP speedup, VQA latency)
are direct proxies for what that hardware will deliver, because LiteRT-LM is
the same SDK across ARM Linux and x86 Linux.

## Citations (all sources verified)

- LiteRT-LM landing page: https://ai.google.dev/edge/litert-lm
- LiteRT-LM GitHub: https://github.com/google-ai-edge/LiteRT-LM
- Google Developers Blog "Bring state-of-the-art agentic skills to the edge
  with Gemma 4" (April 2, 2026):
  https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/
- Gemma 4 E4B `.litertlm` model card (mobile-edge benchmarks tabulated in
  Cell 11 are sourced verbatim from the "Performance" section):
  https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm

## Companion implementations in this repository

- **Cactus mobile alternative** — Flutter Android app on fine-tuned
  Gemma 4 E4B: `mobile-cactus/`
- **LiteRT browser companion** — `.task` bundle on WebGPU via MediaPipe
  Tasks Web: `web-litert/`
- **Cloud tier** — Gemma 4 26B A4B inference + agentic loop:
  `solarhive_inference.py`
- **Fine-tuning pipeline** — Unsloth LoRA → GGUF for Ollama / llama.cpp:
  `solarhive_finetune.py`
"""
