"""
SolarHive — Unsloth Fine-Tuning Notebook
==========================================
SolarHive is an open-source intelligence layer designed to coordinate
community microgrids & community-based storage via fuel cells, pool
midday energy surplus across these microgrids, and eliminate stranded
capacity. It also helps forecast solar irradiance and cloud cover to
plan ahead.

PURPOSE: Fine-tune Gemma 4 E4B into a solar energy domain expert using
Unsloth LoRA (QLoRA when VRAM <55 GB), then export to GGUF for Ollama deployment.

SETUP: Google Colab Pro (developed on G4 VM — RTX PRO 6000 Blackwell 96GB).
       Also runs on A100-40GB (NF4) or A100-80GB / H100 (BF16).
PRIZE TARGET: Unsloth Special Technology Track

Gemma is a trademark of Google LLC.

Pipeline: Load E4B as VLM (BF16 or 4-bit) → LoRA adapters (vision + language)
         → SFT on energy data → GGUF → Ollama
Based on: https://unsloth.ai/docs/basics/vision-fine-tuning
          https://unsloth.ai/docs/models/gemma-4/train
"""

"""## 0: Dependencies (RUN FIRST, THEN RESTART KERNEL IF WARNED)"""

# === CELL 0: Dependencies (RUN FIRST, THEN RESTART KERNEL IF WARNED) ========
# Kaggle model card install command (exact):
#     pip install -U transformers torch accelerate
#
# CRITICAL DEVIATION — we do NOT install `torch`:
#   The card's `-U torch` is written for a generic environment. On Kaggle,
#   torch is pre-installed as a GPU/cuda-pinned build. Running `pip install
#   -U torch` bumps it to 2.11 and pulls in cuda-toolkit 13.0.2, which BREAKS
#   torchaudio/torchvision/cuML/cudf/Unsloth and leaves the GPU stack in an
#   unusable state. The reference notebook "Introduction to Gemma 4" by
#   Jean Fernandes deliberately uses `!pip install -U transformers accelerate`
#   with no `torch` for this reason.
#
# What we actually install:
#   1. `transformers>=5.5.0` — Gemma 4 (model_type `gemma4`) was added in
#      transformers 5.5.0 (Apr 2, 2026). Kaggle's default image ships with
#      5.0.0 (Jan 2026) which has NO `gemma4` and a broken `AutoProcessor`
#      import (`retry` missing from transformers.utils.generic, HF #42352).
#   2. `accelerate` — required by `device_map="auto"` in Cell 2.
#   3. `bitsandbytes` — 4-bit quantization (QLoRA fallback on <55 GB VRAM).
#   4. `unsloth` — Gemma 4 fine-tuning framework (LoRA/QLoRA, ~30% VRAM savings).
#   5. `trl==0.24.0` — SFT trainer used by Unsloth. Pinned to v0.24.0 because
#      `unsloth==2026.4.5` (pinned above) constrains `trl!=0.19.0,<=0.24.0,>=0.18.2`.
#      TRL v1.x (Mar-Apr 2026) is INCOMPATIBLE with this Unsloth pin — pip
#      refuses to install both. Why does TRL 0.24.0 (Oct 2025, pre-Gemma 4)
#      still work? Because Unsloth's `FastVisionModel` + `UnslothVisionData
#      Collator` do all the Gemma 4-specific multimodal handling at the
#      Unsloth layer; TRL is just the SFTTrainer skeleton. The Gemma 4 vision
#      patches in TRL v1.x (image_position_ids, prepare_multimodal_messages)
#      are for non-Unsloth users (e.g., the CARLA GRPO example in
#      https://huggingface.co/blog/gemma4). Pairing Unsloth + TRL 0.24.0 is
#      the supported path. Note: the Unsloth reference notebook
#      (Gemma4_(E4B)-Vision.ipynb) works around this by using
#      `pip install --no-deps`; we keep dep resolution on for safety, hence
#      the explicit pin within Unsloth's allowed range.
#   6. `datasets` — HF dataset loader for the training examples.
#
# We call pip via `subprocess` (not Jupyter `!pip` magic) so this .py file is
# valid Python AND actually runs the install when pasted into a Kaggle cell.
# No `-U` flag: a version-specifier upgrade of transformers is enough; we
# don't want to force-upgrade the other packages if compatible versions are
# already present on Kaggle.
import subprocess as _sp, sys as _sys

_sp.check_call([
    _sys.executable, "-m", "pip", "install", "-q",
    "transformers>=5.5.0",   # upgrades from Kaggle's 5.0.0 to >=5.5.0
    "accelerate",             # installs if missing, skipped if present
    "bitsandbytes",           # installs if missing, skipped if present
    "unsloth==2026.4.5",      # Training provenance pin — matches the Unsloth version used to produce the published HF weights; exact pin ensures bit-reproducibility
    "trl==0.24.0",            # Pinned — Unsloth 2026.4.5 requires trl<=0.24.0 (see comment above)
    "datasets",               # installs if missing, skipped if present
])

# If transformers was already imported in this kernel BEFORE the upgrade,
# Python has cached the old version. A kernel restart is the only fix.
if "transformers" in _sys.modules:
    _cached = getattr(_sys.modules["transformers"], "__version__", "unknown")
    print("=" * 60)
    print(f"⚠️  transformers {_cached} is cached in this kernel's memory.")
    print("⚠️  Click: Runtime → Restart, then run from Cell 0 again.")
    print("=" * 60)
    raise SystemExit("Runtime restart required to load the new transformers.")

print("✓ Cell 0 complete — fine-tuning stack installed. Proceed to Cell 1.")

"""## 1: Verify GPU + transformers version"""

# === CELL 1: Verify GPU + transformers version ==============================
import torch
assert torch.cuda.is_available(), "Enable GPU: Runtime → Change runtime type → GPU"
print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB)")

# Hard precondition check — Gemma 4 requires transformers >= 5.5.0 (released
# Apr 2, 2026). Fail fast here with a clear message instead of letting Cell 2
# blow up with an opaque `ImportError: cannot import name 'retry'`.
import transformers
print(f"transformers version: {transformers.__version__}")

def _ver_tuple(v):
    return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())

if _ver_tuple(transformers.__version__) < (5, 5, 0):
    raise RuntimeError(
        f"transformers {transformers.__version__} is too old for Gemma 4. "
        f"Need >= 5.5.0. Re-run Cell 0, then Runtime → Restart, then start again."
    )

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
_gemma_keys = sorted(k for k in CONFIG_MAPPING.keys() if "gemma" in k.lower())
print(f"Gemma-related model_types registered: {_gemma_keys}")
if "gemma4" not in _gemma_keys:
    raise RuntimeError(
        f"transformers {transformers.__version__} does not register 'gemma4'. "
        f"Found: {_gemma_keys}. Re-run Cell 0, then Runtime → Restart."
    )
print("✅ transformers has Gemma 4 support")

# API keys — load from Kaggle Secrets (Kaggle) or Colab userdata (Colab)
import os, requests
from datetime import datetime

_on_kaggle = os.path.exists("/kaggle/working")
if _on_kaggle:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    OWM_API_KEY  = secrets.get_secret("OWM_API_KEY")
    NREL_API_KEY = secrets.get_secret("NREL_API_KEY")
    EIA_API_KEY  = secrets.get_secret("EIA_API_KEY")
    try:
        HF_TOKEN = secrets.get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = None
    print("   Keys loaded from Kaggle Secrets")
else:
    from google.colab import userdata
    OWM_API_KEY  = userdata.get("OWM_API_KEY")
    NREL_API_KEY = userdata.get("NREL_API_KEY")
    EIA_API_KEY  = userdata.get("EIA_API_KEY")
    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        HF_TOKEN = None
    print("   Keys loaded from Colab Secrets")

if HF_TOKEN:
    print("   HF_TOKEN loaded ✓")
else:
    print("   ⚠️  HF_TOKEN not set — Cell 7 HF push will be skipped (add later)")

# Community constants — same as solarhive_inference.py
LAT, LON = 42.2808, -83.7430
COMMUNITY_CAPACITY_KW = 72
BATTERY_CAPACITY_KWH = 100

# Mount Google Drive for persistent model cache (Colab only)
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass

print("✅ Cell 1 complete — API keys loaded, proceed to Cell 2")

"""## 2: Load Gemma 4 E4B"""

# === CELL 2: Load Gemma 4 E4B ================================================
# MODEL: Gemma 4 E4B — "Any-to-Any" edge model
#   "E" = Effective — uses Per-Layer Embeddings (PLE) for on-device efficiency
#   - Effective params: 4.5B | Total (with embeddings): 8B
#   - Modalities: Text + Image + Audio (max 30s ASR/translation) + Video
#   - Context window: 128K tokens
#   - HuggingFace class: AutoModelForMultimodalLM  (image/audio/video input)
#                        AutoModelForCausalLM       (text-only input)
#   - Unsloth FastVisionModel handles VLM class selection automatically
#
# KAGGLE PATH: google/gemma-4/transformers/gemma-4-e4b-it
#   Unsloth's FastVisionModel accepts a local path from kagglehub.
#   Use kagglehub.model_download() to pre-stage the model, then pass
#   the returned path as model_name.
#
# ROLE IN SOLARHIVE:
#   - Fine-tuned on community solar energy Q&A (text) + sky/panel image pairs
#   - Exported to GGUF for local serving via Ollama
#   - Ollama target: E2B (2.3B effective, 34 quantized variants, runs on laptop CPU)
#   - Qualifies for Unsloth Special Technology Track
#
# NOTE: Gemma 4 26B A4B (in solarhive_inference.py) handles live VQA demo
#   inference — 3.8B active params (MoE), 256K context, image+video support.
#   E2B (not E4B) is the Ollama serving target: lighter, runs anywhere without GPU.
#
# VRAM auto-detect (same pattern as inference.py):
#   ≥55 GB free → BF16 (full precision, no quantization)
#   <55 GB free → 4-bit QLoRA (NF4 via Unsloth)
#
# Google Drive cache: checks Drive first, falls back to kagglehub download.
#   After first download, copy to Drive:
#     !cp -r /root/.cache/kagglehub/models/google/gemma-4 /content/drive/MyDrive/models/
import kagglehub, time as _time
from unsloth import FastVisionModel

MAX_SEQ_LEN = 2048

# Check Google Drive cache first (persists across runtime restarts)
_DRIVE_E4B_PATH = "/content/drive/MyDrive/models/gemma-4/transformers/gemma-4-e4b-it/1"

if os.path.isdir(_DRIVE_E4B_PATH):
    MODEL_PATH = _DRIVE_E4B_PATH
    print(f"✅ Loading from Google Drive cache: {MODEL_PATH}")
else:
    print("Model not in Drive — downloading via kagglehub (this takes 5-10 min)...")
    print("   (output suppressed to prevent Colab display overflow)")
    _dl_start = _time.time()
    # Suppress kagglehub's verbose download output — it floods Colab's display
    # buffer and causes blank output cells (same fix as inference.py Cell 2a).
    from IPython.utils.capture import capture_output as _capture
    with _capture():
        MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e4b-it")
    print(f"✅ Download complete in {_time.time() - _dl_start:.0f}s: {MODEL_PATH}")
    # Auto-cache to Drive so next session skips the download
    import shutil as _shutil
    _cache_src = "/root/.cache/kagglehub/models/google/gemma-4"
    if os.path.isdir(_cache_src) and os.path.isdir("/content/drive"):
        print("Caching model to Google Drive (next session will load instantly)...")
        _shutil.copytree(_cache_src, "/content/drive/MyDrive/models/gemma-4", dirs_exist_ok=True)
        print("✅ Cached to Drive")

# VRAM auto-detect: BF16 on large GPUs, 4-bit QLoRA on smaller GPUs
_free_vram_gb = torch.cuda.mem_get_info(0)[0] / 1e9
_use_4bit = _free_vram_gb < 55

if _use_4bit:
    print(f"GPU has {_free_vram_gb:.0f} GB free — loading in 4-bit QLoRA (NF4)")
else:
    print(f"GPU has {_free_vram_gb:.0f} GB free — loading in BF16 (full precision)")

model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    load_in_4bit=_use_4bit,                # QLoRA on <55GB, full precision on ≥55GB
    use_gradient_checkpointing="unsloth",  # saves VRAM for vision token sequences
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,                            # auto-detect (BF16 on supporting GPUs)
    full_finetuning=False,                 # explicit LoRA mode
)
print(f"✅ Gemma 4 E4B loaded ({'4-bit NF4' if _use_4bit else 'BF16'}) from: {MODEL_PATH}")

# Use the chat template bundled with the Kaggle-distributed Gemma 4 E4B
# model (in tokenizer_config.json), which ships Google's official Gemma 4
# chat template. Per Unsloth Tip #1, E2B/E4B should use the "gemma-4"
# (non-thinking) template — Kaggle's bundle is already that template.
#
# Chat-template flow:
#   Cell 2: load processor → uses Kaggle-bundled chat template
#   Cell 4b: processor.apply_chat_template(...) → renders rows to plain text
#            (pass tools=TOOL_SCHEMAS for tool-calling rows)
#   Cell 5: SFTTrainer reads dataset["text"] → tokenizes → trains
#   Cell 7: processor.save_pretrained(...) → saves the same chat template
#           alongside the LoRA adapters for inference parity
tokenizer = processor.tokenizer  # text-ops alias (tool-calling, benchmarks, GGUF export)

"""## 3: Attach LoRA Adapters"""

# === CELL 3: Attach LoRA Adapters =============================================
# Per the Unsloth Gemma 4 text-only reference notebook
# (Gemma4_(E4B)-Text.ipynb): turn off vision-layer fine-tuning when the
# training data is text-only. The vision encoder receives no input during
# text training, so its LoRA weights would be saved but never updated —
# wasted parameter count and disk size. Disabling here reduces trainable
# params to language + attention + MLP only.
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # text-only training — vision encoder unused
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules="all-linear",
)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

"""## 4: Training Data"""

# === CELL 4: Training Data ===================================================
# Energy domain Q&A and tool-calling pairs come from solarhive_datagen.py:
# algorithmic categories (A–G) grounded on Open-Meteo, NREL PVWatts, EIA grid
# mix, and OWM weather snapshots, plus hand-crafted Q&A and when-to-call
# tool-routing extensions. The DATA / TOOL_CALL_DATA placeholders below are
# filled at runtime from datagen_latest.json (used for category reporting in
# Cell 4c). Actual training data is loaded from the published HF dataset in
# Cell 4b.
#
# Geographically diverse: ~60% location-agnostic, ~30% varied US locations,
# ~10% Ann Arbor demo community.

# Unified system prompt — single source of truth across the project.
# Body must match `_UNIFIED_SYSTEM_BODY` in solarhive_datagen.py and
# the SYSTEM_PROMPT body in solarhive_inference.py byte-for-byte (verified
# by the project test suite).
#
# Body repeated twice per Leviathan et al. (2024), "Repeat to Improve
# Non-Reasoning LLMs", Google Research. https://arxiv.org/abs/2512.14982
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
SYS = _UNIFIED_SYSTEM_BODY + "\n\n" + _UNIFIED_SYSTEM_BODY
SYS_TOOLS = SYS  # Unified — same prompt for Q&A and tool-calling training

DATA = []  # Hand-crafted Q&A entries are produced by solarhive_datagen.py Cell 7a
           # and surfaced here via the multimodal HF dataset
           # (Truthseeker87/solarhive-community-solar-multimodal). This empty
           # list keeps the downstream assembly code paths working as no-ops.

"""## 4a: Tool-Calling Training Data"""

# === CELL 4a: Tool-Calling Training Data ======================================
# Gemma 4 supports native function calling. Fine-tuning on plain Q&A only risks
# overwriting that capability. These examples teach the model WHEN and HOW to
# call tools for dynamic/volatile data vs answering from knowledge for static data.
#
# Design principle (from project plan):
#   - Immutable data (geography, seasons, physics) → plain Q&A (baked in)
#   - Dynamic data (weather, GHI, battery SOC, grid pricing) → tool-calling format
#
# Distribution (~50 examples):
#   10 get_weather only | 8 get_solar_production | 8 get_battery_state
#   6 get_grid_status   | 8 multi-tool chains    | 10 direct answers (no tool)
#
# Geographic diversity: Ann Arbor, Phoenix, Seattle, Miami, Denver, Chicago,
# Portland, Houston, Minneapolis, Boston, San Diego, Atlanta

# Tool schemas — must match solarhive_inference.py tool definitions
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets current weather conditions for the community.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. 'Ann Arbor, MI'",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_solar_production",
            "description": "Estimates current community solar production using live solar irradiance data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clouds_pct": {
                        "type": "integer",
                        "description": "Current cloud cover percentage (0-100). Get this from get_weather first.",
                    },
                    "temp_f": {
                        "type": "number",
                        "description": "Current temperature in Fahrenheit. Get this from get_weather first.",
                    },
                },
                "required": ["clouds_pct", "temp_f"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_battery_state",
            "description": "Gets the current state of the community shared battery storage.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grid_status",
            "description": "Gets current electricity grid pricing period, rate, and grid mix (renewable percentage, CO2 intensity).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Tool-calling training examples — multi-turn message lists
# Each is a full conversation: system → user → assistant (tool_calls + tool_responses) → assistant (answer)
# Some examples have NO tool calls (direct answers) to teach when NOT to call tools.
TOOL_CALL_DATA = []  # Hand-crafted tool-calling examples are produced by
                     # solarhive_datagen.py Cell 7a and surfaced via the
                     # multimodal HF dataset. See note above on DATA.

"""## 4b: API-Grounded Training Data"""

# === CELL 4b: API-Grounded Training Data ======================================
# Fetches real conditions from OWM + Open-Meteo GHI at training time.
# Uses the SAME API services as solarhive_inference.py so training data
# matches inference data sources. API keys loaded in Cell 1.
#
# Generates ~15 examples: live status, cloud cover, temperature, grid strategy,
# wind/humidity, time-of-day, EV charging (5 SOC variants), battery priority (4 SOC).
# Appended to DATA before dataset construction — counted in final training total.


def _get_current_ghi():
    """Fetch current Global Horizontal Irradiance (W/m²) from Open-Meteo.

    Free API, no key required. Same endpoint as solarhive_inference.py.
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


def _fetch_api_examples():
    """Fetch live Ann Arbor conditions and return grounded training tuples.

    Uses same API services as inference.py: OWM (weather) + Open-Meteo (GHI).
    API keys loaded from Cell 1 globals (Kaggle Secrets or Colab userdata).
    """
    # OWM: weather conditions (uses global OWM_API_KEY from Cell 1)
    _owm_live = False
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": LAT, "lon": LON, "appid": OWM_API_KEY, "units": "imperial"},
            timeout=10,
        ).json()
        if "clouds" not in r:
            # API returned error (e.g. invalid key) — log response for debugging
            print(f"  ⚠️  OWM returned error: {r.get('message', r)} — using fallback weather")
        else:
            _owm_live = True
    except Exception as e:
        print(f"  ⚠️  OWM request failed: {e} — using fallback weather")
        r = {}

    if _owm_live:
        clouds   = r["clouds"]["all"]
        temp_f   = round(r["main"]["temp"])
        desc     = r["weather"][0]["description"]
        wind_mph = round(r["wind"]["speed"], 1)
        humidity = r["main"]["humidity"]
        print(f"  ✓ OWM live: {desc}, {clouds}% clouds, {temp_f}°F")
    else:
        # Fallback: realistic Ann Arbor defaults so we still generate examples
        clouds, temp_f, desc, wind_mph, humidity = 40, 72, "partly cloudy", 8.5, 55
        print(f"  ℹ️  Using fallback weather: {desc}, {clouds}% clouds, {temp_f}°F")

    # Open-Meteo: GHI irradiance (free, no key — same as inference.py)
    ghi = _get_current_ghi()

    # Derived metrics — matches solarhive_inference.py production calculation
    SYSTEM_EFF = 0.85  # inverter × wiring × soiling × mismatch
    now     = datetime.now()
    hour    = now.hour
    month   = now.month
    # Time factor (sun angle proxy) — always computed for training text descriptions
    tf = max(0.0, 1.0 - ((hour - 12) / 6) ** 2) if 6 <= hour <= 18 else 0.0
    # Temperature derating: silicon panels lose ~0.4%/°F above 77°F (25°C)
    temp_derate = max(0.75, 1.0 - 0.004 * max(0, temp_f - 77))

    if ghi is not None:
        # GHI-based: satellite-measured irradiance (same formula as inference.py)
        prod_kw = round(max(0, COMMUNITY_CAPACITY_KW * (ghi / 1000) * SYSTEM_EFF * temp_derate), 1)
        eff_pct = round(prod_kw / COMMUNITY_CAPACITY_KW * 100, 1)
        ghi_note = f" GHI: {round(ghi, 1)} W/m²."
        source = "open-meteo"
    else:
        # Fallback: cloud%-based estimate (matches inference.py fallback path)
        eff  = max(0.15, 0.85 - (clouds / 100) * 0.70)
        prod_kw = round(COMMUNITY_CAPACITY_KW * eff * tf * temp_derate, 1)
        eff_pct = round(eff * temp_derate * 100, 1)
        ghi_note = ""
        source = "fallback"

    if 14 <= hour < 19:                            # 2pm-6:59pm
        period, rate_str = "peak", "$0.28/kWh"
    elif (7 <= hour < 14) or (19 <= hour < 23):    # 7am-1:59pm OR 7pm-10:59pm
        period, rate_str = "mid-peak", "$0.18/kWh"
    else:                                           # 11pm-6:59am
        period, rate_str = "off-peak", "$0.10/kWh"

    SEASONS = ["winter","winter","spring","spring","spring","summer",
               "summer","summer","fall","fall","fall","winter"]
    season = SEASONS[month - 1]

    SYS_ = "You are SolarHive, an AI energy advisor for a community solar microgrid."
    ex = []

    # ── 1. Live status ──────────────────────────────────────────────────────────
    if prod_kw > 40:
        status_tip = "Strong generation — run heavy loads now to maximize self-consumption."
    elif prod_kw > 15:
        status_tip = "Moderate generation — routine loads are fine, defer high-draw appliances."
    else:
        status_tip = "Low generation — rely on battery for essentials, defer discretionary loads."

    ex.append((SYS_,
        "What's the current solar production status?",
        f"Live: Ann Arbor shows {desc}, {clouds}% clouds, {temp_f}°F.{ghi_note} "
        f"Output: {prod_kw}kW of 72kW capacity ({eff_pct}% efficiency). "
        f"Grid is {period} at {rate_str}. {status_tip}"
    ))

    # ── 2. Cloud cover impact ───────────────────────────────────────────────────
    if clouds < 25:
        cloud_note = "Clear skies deliver maximum irradiance and panel efficiency."
    elif clouds < 55:
        cloud_note = f"Partial cloud cover ({clouds}%) causes intermittent dips — the battery buffers these automatically."
    else:
        cloud_note = f"Heavy overcast ({clouds}%) is the main production constraint right now."

    cloud_tip = ("Consider shifting big loads (EV, washer, pool heater) until clouds clear."
                 if clouds > 50 else
                 "Good conditions — run loads during solar hours to maximize free energy use.")

    ex.append((SYS_,
        f"We have {clouds}% cloud cover. What does that mean for production?",
        f"At {clouds}% cloud cover, efficiency is {eff_pct}% — producing {prod_kw}kW.{ghi_note} "
        f"{cloud_note} {cloud_tip}"
    ))

    # ── 3. Temperature impact ───────────────────────────────────────────────────
    if temp_f > 90:
        derate = round((temp_f - 77) * 0.4, 1)
        temp_note = (f"At {temp_f}°F, thermal derating is significant — panels lose ~{derate}% efficiency. "
                     "AC demand across 12 homes will also be high; expect tight net production margins.")
    elif temp_f > 77:
        derate = round((temp_f - 77) * 0.35, 1)
        temp_note = f"At {temp_f}°F there's minor derating (~{derate}%), but conditions remain productive."
    elif temp_f >= 32:
        temp_note = (f"At {temp_f}°F panels run cool — slightly better efficiency than the 77°F rating point. "
                     "No thermal losses.")
    else:
        temp_note = (f"At {temp_f}°F, check for ice or snow on panels. Cold improves cell efficiency, "
                     "but physical shading cuts output to zero.")

    ex.append((SYS_,
        f"It's {temp_f}°F outside. How does temperature affect our solar output?",
        f"{temp_note} Current output: {prod_kw}kW. "
        "Crystalline silicon loses ~0.35–0.45% per °F above 77°F (25°C)."
    ))

    # ── 4. Grid pricing strategy ────────────────────────────────────────────────
    if period == "peak":
        grid_strategy = ("Avoid grid imports — battery should discharge to cover home loads. "
                         "Only use grid if battery drops below 20%.")
    elif period == "mid-peak":
        grid_strategy = "Use solar first, battery second. Draw from grid only for loads exceeding both sources."
    else:
        grid_strategy = ("Cheapest grid time ($0.10/kWh). If battery is below 40% and solar is low, "
                         "a grid top-up now is economical.")

    ex.append((SYS_,
        f"It's {hour:02d}:00, {period} pricing. What's the grid strategy?",
        f"Current: {period} at {rate_str}, solar at {prod_kw}kW. {grid_strategy}"
    ))

    # ── 5. Wind and humidity ────────────────────────────────────────────────────
    wind_note = (f"Wind at {wind_mph} mph cools panels slightly — minor 1-2% efficiency gain."
                 if wind_mph > 15 else
                 f"Wind at {wind_mph} mph has negligible effect on photovoltaic output.")

    humid_note = (f"Humidity at {humidity}% creates aerosol haze, reducing direct irradiance by 10-15%."
                  if humidity > 80 else
                  f"Humidity at {humidity}% is in normal range — no meaningful production impact.")

    ex.append((SYS_,
        f"Wind is {wind_mph} mph and humidity is {humidity}%. Does that affect our output?",
        f"{wind_note} {humid_note} "
        f"Combined with {clouds}% clouds and {temp_f}°F, current production is {prod_kw}kW."
    ))

    # ── 6. Time-of-day ──────────────────────────────────────────────────────────
    if tf < 0.05:
        tod_note = "Sun is below the horizon — production is zero. Battery and grid are the only sources."
    elif tf < 0.4:
        tod_note = f"Low sun angle gives only {round(tf*100)}% of midday intensity — will improve as sun rises."
    else:
        tod_note = f"Good sun angle — {round(tf*100)}% of peak midday intensity available."

    ex.append((SYS_,
        f"It's {hour:02d}:00 in {season}. What's our generation potential?",
        f"At {hour:02d}:00 in Ann Arbor ({season}), sun angle provides {round(tf*100)}% of peak intensity. "
        f"With {clouds}% clouds and {temp_f}°F, current output is {prod_kw}kW. {tod_note}"
    ))

    # ── 7–11. EV charging at varying battery levels ─────────────────────────────
    for soc in [15, 35, 55, 75, 90]:
        gap = round(7.2 - prod_kw, 1)
        if prod_kw >= 7.2:
            surplus = round(prod_kw - 7.2, 1)
            batt_state = "building" if soc < 50 else "nearly full"
            ev_ans = (f"Yes — {prod_kw}kW solar covers the 7.2kW EV charge with {surplus}kW spare. "
                      f"Battery at {soc}% is {batt_state} — good time to charge both.")
        elif soc > 30:
            ev_ans = (f"Solar ({prod_kw}kW) is {gap}kW short of the 7.2kW EV draw. "
                      f"Battery at {soc}% can cover the gap. Go ahead, but monitor if charging extends over 2 hours.")
        else:
            ev_ans = (f"Solar ({prod_kw}kW) can't cover 7.2kW EV charging, and battery at {soc}% is too low to supplement safely. "
                      "Wait for production to increase or charge during off-peak ($0.10/kWh).")

        ex.append((SYS_,
            f"Battery at {soc}%, solar producing {prod_kw}kW. Can I charge my EV (7.2kW Level 2)?",
            ev_ans
        ))

    # ── 12–15. Battery priority at different states ─────────────────────────────
    for soc, kwh in [(18, 18), (45, 45), (72, 72), (91, 91)]:
        if soc < 30:
            batt_ans = (f"Battery at {soc}% ({kwh}kWh) is critically low. "
                        f"With {prod_kw}kW solar, prioritize battery recovery — defer all non-essential loads until reaching 50%.")
        elif soc < 60:
            load_note = "comfortably covers normal loads while charging" if prod_kw > 15 else "is low — focus generation on battery recovery first"
            batt_ans = (f"Battery at {soc}% ({kwh}kWh) is moderate. {prod_kw}kW solar {load_note}.")
        elif soc < 85:
            batt_ans = (f"Battery at {soc}% ({kwh}kWh) is healthy — good evening buffer. "
                        f"With {prod_kw}kW solar, normal loads and continued charging are both fine.")
        else:
            batt_ans = (f"Battery at {soc}% is nearly full ({kwh}kWh / 100kWh). "
                        f"Solar surplus will export at {rate_str} ({period}). "
                        "Shift large loads to now to consume solar directly rather than exporting.")

        ex.append((SYS_,
            f"Battery is at {soc}% with {prod_kw}kW solar production. What's the recommended action?",
            batt_ans
        ))

    print(f"  ✅ {len(ex)} API-grounded examples generated "
          f"(live: {desc}, {clouds}% clouds, {prod_kw}kW, {period}, source: {source})")
    return ex


_api_ex = _fetch_api_examples()
DATA.extend(_api_ex)

# Optional: load datagen output (from solarhive_datagen.py) if available
import json as _json
_datagen_path = "/content/drive/MyDrive/models/solarhive_datasets/datagen_latest.json"
_datagen_loaded = False
_dg = {}
if os.path.exists(_datagen_path):
    with open(_datagen_path) as f:
        _dg = _json.load(f)
    DATA.extend([tuple(ex) for ex in _dg["qa_data"]])
    print(f"  Loaded {len(_dg['qa_data'])} Q&A examples from datagen")
    if "tool_call_data" in _dg:
        TOOL_CALL_DATA.extend(_dg["tool_call_data"])
        print(f"  Loaded {len(_dg['tool_call_data'])} tool-calling examples from datagen")
    _datagen_loaded = True

print(f"Q&A training examples: {len(DATA)} ({len(DATA)-len(_api_ex)} static + {len(_api_ex)} API-grounded)")
print(f"Tool-calling examples: {len(TOOL_CALL_DATA)}")


# Convert to dataset — two formats for vision-compatible training:
#   Q&A examples  → messages format (ready for image pairs when added)
#   Tool-calling  → pre-rendered text (needs tools= in apply_chat_template)
from datasets import Dataset


def to_messages(example):
    """Convert a Q&A 3-tuple to VLM messages format.

    Returns {"messages": [message_dicts]} — compatible with UnslothVisionDataCollator.
    Image pairs can be added later by appending {"type": "image", "image": img}
    to the user content list.
    """
    sys, user, asst = example
    return {"messages": [
        {"role": "system", "content": sys},
        {"role": "user",
         "content": [
             {"type": "text", "text": user},
         ]},
        {"role": "assistant",
         "content": [
             {"type": "text", "text": asst},
         ]},
    ]}


def to_chat(example):
    """Format a training example as chat template text (tool-calling + fallback).

    Accepts either:
    - (sys, user, asst) 3-tuple — plain Q&A
    - list of message dicts — multi-turn with tool calls

    .removeprefix('<bos>') prevents double BOS token (Unsloth adds its own).
    """
    if isinstance(example, (list, tuple)) and len(example) == 3 and isinstance(example[0], str):
        sys, user, asst = example
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": sys},
             {"role": "user", "content": user},
             {"role": "assistant", "content": asst}],
            tokenize=False, add_generation_prompt=False,
        ).removeprefix('<bos>')
    else:
        return tokenizer.apply_chat_template(
            example,
            tools=TOOL_SCHEMAS,
            tokenize=False, add_generation_prompt=False,
        ).removeprefix('<bos>')

# Combine all training data
ALL_DATA = list(DATA) + TOOL_CALL_DATA

# --- Load dataset and pre-render to plain text strings ------------------------
#
# Pipeline:
#   • Read the published corpus from HF
#   • Parse each row's `messages` JSON string back to a Python list of dicts
#   • Pre-render to a plain text string via processor.apply_chat_template(...)
#     — for tool-calling examples, pass tools=TOOL_SCHEMAS so the chat template
#     emits the function-calling preamble
#   • Build a `text`-column HuggingFace Dataset that the SFTTrainer can consume
#     with TRL's default text collator
#
# Why pre-render here instead of letting the trainer/collator render at batch
# time: matches the working text-only Gemma 4 fine-tune pattern from Unsloth's
# reference notebook (Gemma4_(E4B)-Text.ipynb), which is the most stable path
# on this corpus. Image rows are skipped at render time; multimodal training
# is deferred to a future cycle with a real image corpus and a held-out VQA
# benchmark.
#
# Inference compatibility: the LoRA adapter trained here loads onto
# FastVisionModel (which solarhive_inference.py uses), so VQA capability of
# the base model is preserved. Only the SFT data path is text-only.

import json as _json_v2
from datasets import load_dataset, Dataset as _Dataset

V2_DATASET_REPO = "Truthseeker87/solarhive-community-solar-multimodal"
dataset = load_dataset(
    V2_DATASET_REPO,
    split="train",
    token=HF_TOKEN if "HF_TOKEN" in dir() else None,
)
print(f"v2 dataset: {len(dataset)} rows from {V2_DATASET_REPO}")
print(f"  Schema: {list(dataset.features.keys())}")
import collections as _collections
_modality_counts = _collections.Counter(dataset["modality"])
print(f"  Modality split: {dict(_modality_counts)}")


def _render_row(row, processor_to_use=None):
    """Return rendered chat text for a v2 dataset row, or None to skip.

    Skip rules:
      • image-modality rows — cannot be rendered as plain text without losing
        the image; deferred to v3 multimodal cycle
      • rows whose chat template fails to render (template-version edge cases)
    """
    proc = processor_to_use or processor
    if row.get("modality") == "image":
        return None
    msgs = row.get("messages")
    if isinstance(msgs, str):
        try:
            msgs = _json_v2.loads(msgs)
        except Exception:
            return None
    if not isinstance(msgs, list) or not msgs:
        return None
    has_tools = any(("tool_calls" in m) or m.get("role") == "tool" for m in msgs)
    try:
        kwargs = {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False}
        if has_tools:
            kwargs["tools"] = TOOL_SCHEMAS
        rendered = proc.apply_chat_template(msgs, **kwargs)
        return rendered.removeprefix("<bos>") if rendered else None
    except Exception:
        return None


_rendered_texts = []
_rendered_skipped_image = 0
_rendered_skipped_tool = 0
_rendered_skipped_other = 0
for _row in dataset:
    if _row.get("modality") == "image":
        _rendered_skipped_image += 1
        continue
    _txt = _render_row(_row)
    if _txt is None:
        # Distinguish between tool_call render failures and other failures
        _msgs_raw = _row.get("messages")
        if isinstance(_msgs_raw, str) and ("tool_calls" in _msgs_raw or '"role":"tool"' in _msgs_raw or '"role": "tool"' in _msgs_raw):
            _rendered_skipped_tool += 1
        else:
            _rendered_skipped_other += 1
        continue
    _rendered_texts.append(_txt)

dataset = _Dataset.from_dict({"text": _rendered_texts})
print(f"  Pre-rendered {len(dataset)} text rows for v2 SFT.")
print(f"  Skipped {_rendered_skipped_image} image-modality rows (deferred to v3).")
if _rendered_skipped_tool:
    print(f"  Skipped {_rendered_skipped_tool} tool_call rows that failed to render "
          f"(template alternation — recoverable in a future iteration).")
if _rendered_skipped_other:
    print(f"  Skipped {_rendered_skipped_other} other rows that failed to render.")

"""## 4c: Data Inspection & Validation"""

# === CELL 4c: Data Inspection & Validation ====================================
# Run BEFORE training to catch data issues early. Shows ML rigor for judges.

print("=" * 60)
print("DATA INSPECTION REPORT")
print("=" * 60)

# 1. Format validation — Q&A tuples must be (system, user, assistant)
_bad = [(i, d) for i, d in enumerate(DATA) if len(d) != 3 or not all(isinstance(s, str) and len(s) > 0 for s in d)]
if _bad:
    for i, d in _bad:
        print(f"  ❌ Q&A Example {i}: malformed — expected (sys, user, asst) non-empty strings")
    raise ValueError(f"{len(_bad)} malformed Q&A examples — fix before training")
print(f"✅ Format: all {len(DATA)} Q&A examples are valid (system, user, assistant) tuples")

# 1b. Tool-calling format validation
_valid_tool_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
_tc_bad = []
_tc_with_tools = 0
_tc_no_tools = 0
for i, ex in enumerate(TOOL_CALL_DATA):
    if not isinstance(ex, list) or len(ex) < 3:
        _tc_bad.append((i, "too few messages"))
        continue
    # Must have system + user + at least one assistant
    has_system = any(m.get("role") == "system" for m in ex)
    has_user = any(m.get("role") == "user" for m in ex)
    has_assistant = any(m.get("role") == "assistant" for m in ex)
    if not (has_system and has_user and has_assistant):
        _tc_bad.append((i, "missing system/user/assistant"))
        continue
    # Check tool names if tool_calls present
    has_tool_calls = any(m.get("tool_calls") for m in ex)
    if has_tool_calls:
        _tc_with_tools += 1
        for m in ex:
            for tc in m.get("tool_calls", []):
                fn_name = tc["function"]["name"]
                if fn_name not in _valid_tool_names:
                    _tc_bad.append((i, f"unknown tool: {fn_name}"))
    else:
        _tc_no_tools += 1
    # Must end with assistant content (final answer)
    if ex[-1].get("role") != "assistant" or not ex[-1].get("content"):
        _tc_bad.append((i, "must end with assistant content"))

if _tc_bad:
    for i, reason in _tc_bad:
        print(f"  ❌ Tool-call example {i}: {reason}")
    raise ValueError(f"{len(_tc_bad)} malformed tool-calling examples — fix before training")
print(f"✅ Tool-calling: all {len(TOOL_CALL_DATA)} examples valid "
      f"({_tc_with_tools} with tool calls, {_tc_no_tools} direct answers)")

# Check ratio: ~20% should be no-tool (direct answer) examples
_no_tool_pct = round(_tc_no_tools / len(TOOL_CALL_DATA) * 100, 1) if TOOL_CALL_DATA else 0
if _no_tool_pct < 10:
    print(f"  ⚠️  Only {_no_tool_pct}% direct-answer examples — consider adding more (target ~20%)")
else:
    print(f"✅ Direct-answer ratio: {_no_tool_pct}% of tool-calling set (teaches when NOT to call tools)")

# 2. Duplicate check (Q&A only — tool-calling questions may overlap intentionally)
_questions = [d[1] for d in DATA]
_dupes = {q for q in _questions if _questions.count(q) > 1}
if _dupes:
    print(f"  ⚠️  {len(_dupes)} duplicate question(s):")
    for q in _dupes:
        print(f"      → {q[:80]}...")
else:
    print(f"✅ Uniqueness: no duplicate questions in Q&A set")

# 3. Answer length — flag any < 50 words (Q&A only)
_short = [(i, len(d[2].split())) for i, d in enumerate(DATA) if len(d[2].split()) < 50]
if _short:
    print(f"  ⚠️  {len(_short)} short answers (< 50 words):")
    for i, wc in _short:
        print(f"      → Example {i}: {wc} words — '{DATA[i][1][:60]}...'")
else:
    print(f"✅ Answer length: all Q&A answers ≥ 50 words")

# 4. Token length distribution (all examples including tool-calling)
# Datasets may carry either a pre-rendered "text" column or a "messages"
# (JSON string) column. Tokenizing the messages JSON is a rough proxy for
# the rendered chat-template length but adequate for the MAX_SEQ_LEN
# overflow gate.
_text_field = "text" if "text" in dataset.column_names else "messages"
_token_lens = []
for text in dataset[_text_field]:
    _toks = tokenizer(text, return_tensors=None)["input_ids"]
    _token_lens.append(len(_toks))

_min_t, _max_t = min(_token_lens), max(_token_lens)
_mean_t = sum(_token_lens) / len(_token_lens)
_sorted_t = sorted(_token_lens)
_median_t = _sorted_t[len(_sorted_t) // 2]
_over = [l for l in _token_lens if l > MAX_SEQ_LEN]

print(f"\n📊 Token lengths (MAX_SEQ_LEN = {MAX_SEQ_LEN}):")
print(f"   min: {_min_t}  median: {_median_t}  mean: {_mean_t:.0f}  max: {_max_t}")
if _over:
    print(f"  ❌ {len(_over)} examples EXCEED max seq length — will be truncated!")
else:
    print(f"   ✅ All examples fit within {MAX_SEQ_LEN} tokens (headroom: {MAX_SEQ_LEN - _max_t})")

# 5. Category distribution — read from datagen metadata (source of truth).
# Falls back to a summary line if datagen_latest.json wasn't loaded this run.
_n_static = len(DATA) - len(_api_ex)
_n_api = len(_api_ex)
print(f"\n📋 Category distribution:")
_datagen_cats = (
    _dg.get("metadata", {}).get("categories", {})
    if _datagen_loaded else {}
)
if _datagen_cats:
    print(f"   Datagen Q&A: {_n_static} examples across {len(_datagen_cats)} category buckets")
    for cat, count in _datagen_cats.items():
        print(f"     • {cat}: {count}")
else:
    print(f"   Datagen Q&A: {_n_static} examples (datagen metadata unavailable)")
print(f"   API-grounded (live, this run): {_n_api}")
print(f"   Tool-calling: {len(TOOL_CALL_DATA)} ({_tc_with_tools} with tools, {_tc_no_tools} direct)")
print(f"   Datagen-reporting total (DATA + tool-calling): {len(ALL_DATA)}")
print(f"   Training corpus (HF dataset): {len(dataset)} rows  ← actual training input")

# 6. Sample preview — show one formatted Q&A and one tool-calling example
print(f"\n📄 Sample example (index 0):")
print("─" * 60)
# Defensive: dataset may carry a pre-rendered "text" column (preferred) or
# a "messages" JSON-string column. Try "text" first, fall back to "messages".
try:
    _sample = dataset["text"][0]
    print(_sample[:500] + ("..." if len(_sample) > 500 else ""))
except (KeyError, ValueError):
    _sample = dataset["messages"][0]
    print(f"[messages JSON]: {_sample[:500] + ('...' if len(_sample) > 500 else '')}")
print("─" * 60)

# Surface a sample image-modality row when present.
# Defensive: only attempt the modality lookup if the dataset actually has
# a `modality` column. The pre-rendered training dataset built in Cell 4b
# has only a `text` column; accessing dataset["modality"] there would raise
# ValueError ("Column 'modality' doesn't exist."), not KeyError.
_cols = getattr(dataset, "column_names", [])
if "modality" in _cols and "messages" in _cols:
    try:
        _img_idx = next(i for i, m in enumerate(dataset["modality"])
                        if m == "image")
        print(f"\n📄 Sample image-grounded example (index {_img_idx}):")
        print("─" * 60)
        _img_msg = dataset["messages"][_img_idx]
        print(_img_msg[:500] + ("..." if len(_img_msg) > 500 else ""))
        print("─" * 60)
    except (StopIteration, KeyError, ValueError):
        # No image rows present (or column lookup race) — non-blocking.
        pass

print(f"\n✅ Cell 4c complete — data validated. Proceed to training.")

"""## 4d: Save Dataset to Google Drive"""

# === CELL 4d: Save Dataset to Google Drive ====================================
# Persists the exact training data used for this run. Enables:
#   - Audit trail: judges can inspect what the model was trained on
#   - Reproducibility: API-grounded examples change per run (live weather)
#   - Comparison: diff datasets across training runs
import json

_ds_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_ds_drive_dir = "/content/drive/MyDrive/models/solarhive_datasets"

try:
    os.makedirs(_ds_drive_dir, exist_ok=True)

    # JSONL — human-readable, one formatted example per line.
    # If the dataset has a pre-rendered "text" column, write that. Otherwise
    # emit messages JSON plus modality tag; image bytes stay in the HF
    # Dataset (preserved by save_to_disk below), the JSONL is path-only for
    # fast inspection.
    _jsonl_path = f"{_ds_drive_dir}/train_{_ds_timestamp}.jsonl"
    with open(_jsonl_path, "w", encoding="utf-8") as f:
        if "text" in dataset.column_names:
            for text in dataset["text"]:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        else:
            for _msgs, _mod in zip(dataset["messages"], dataset["modality"]):
                f.write(json.dumps(
                    {"messages": _msgs, "modality": _mod},
                    ensure_ascii=False,
                ) + "\n")

    # HF Dataset format — fast reload with load_from_disk()
    _hf_path = f"{_ds_drive_dir}/train_{_ds_timestamp}"
    dataset.save_to_disk(_hf_path)

    _jsonl_mb = os.path.getsize(_jsonl_path) / 1e6
    print(f"✅ Dataset saved to Google Drive:")
    print(f"   JSONL: {_jsonl_path} ({_jsonl_mb:.1f} MB)")
    print(f"   HF:    {_hf_path}/")
    print(f"   Reload: Dataset.load_from_disk('{_hf_path}')")
except Exception as e:
    print(f"⚠️  Dataset save skipped (non-blocking): {e}")
    print("   Training continues — dataset is in memory.")

"""## 5: Train"""

# === CELL 5: Train ===========================================================
# Text-only SFT with TRL's default text data collator. The `dataset` built in
# Cell 4b is already pre-rendered into a `text` column via
# processor.apply_chat_template — no batch-time chat-template rendering is
# needed, and no multimodal collator is loaded.
from trl import SFTTrainer, SFTConfig

# Auto-tune batch size by GPU VRAM (effective batch = per_device × accumulation)
_train_vram_gb = torch.cuda.mem_get_info(0)[0] / 1e9
if _train_vram_gb >= 55:
    _batch_size, _grad_accum = 4, 4   # ≥55GB VRAM (BF16): effective 16
elif _train_vram_gb >= 30:
    _batch_size, _grad_accum = 2, 8   # 30-55GB VRAM (NF4): effective 16
else:
    _batch_size, _grad_accum = 1, 8   # T4 / L4: effective 8
print(f"Training: batch_size={_batch_size}, grad_accum={_grad_accum} "
      f"(effective batch {_batch_size * _grad_accum}, VRAM: {_train_vram_gb:.0f} GB free)")
print(f"Training data: {len(dataset)} pre-rendered text examples")

trainer = SFTTrainer(
    model=model,
    processing_class=processor.tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=_batch_size,
        gradient_accumulation_steps=_grad_accum,
        warmup_steps=5,                    # Unsloth Gemma 4 text-only default
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="solarhive_out",
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        # max_grad_norm left at TRL default (1.0); the lower 0.3 value seen in
        # some vision-encoder recipes is unnecessary for text-only training.
        seed=3407,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
    ),
)

# Per the Unsloth Gemma 4 text-only reference notebook
# (Gemma4_(E4B)-Text.ipynb): wrap the trainer with `train_on_responses_only`
# so the loss is computed on assistant outputs only (the
# `<|turn>model\n...<turn|>` spans), not on the user/system input. This
# improves convergence by focusing gradient signal on what we actually want
# the model to learn to produce, rather than what we're teaching it to
# read. Wrapped in try/except so a chat-template token mismatch doesn't
# block training — falls back to full-sequence loss.
try:
    from unsloth.chat_templates import train_on_responses_only as _tor
    trainer = _tor(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )
    print("✅ train_on_responses_only wrapper applied — loss masked to assistant outputs.")
except Exception as _tor_err:
    print(f"⚠️  train_on_responses_only failed ({_tor_err}); falling back to "
          f"full-sequence loss. Training continues.")

stats = trainer.train()
_log = [x["loss"] for x in trainer.state.log_history if "loss" in x]
_last20 = _log[-20:] if len(_log) >= 20 else _log
print(f"✅ E4B done — {stats.metrics['train_runtime']:.0f}s, {len(_log)} steps")
print(f"   Converged loss (last 20 avg): {sum(_last20)/len(_last20):.4f}")
print(f"   Final step: {_log[-1]:.4f} | Min: {min(_log):.4f} | HF avg (all steps): {stats.training_loss:.4f}")

"""## 6: Test Fine-Tuned Model"""

# === CELL 6: Test Fine-Tuned Model ============================================
FastVisionModel.for_inference(model)

# Test 1: General knowledge (should answer directly, no tool call)
print("─" * 60)
print("TEST 1: General knowledge (expect direct answer)")
print("─" * 60)
_t1_text = processor.apply_chat_template(
    [{"role": "system", "content": SYS_TOOLS},
     {"role": "user", "content": "It's snowing and panels are covered. What now?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
_t1_inputs = processor(text=_t1_text, return_tensors="pt").to(model.device)

out = model.generate(**_t1_inputs, max_new_tokens=300,
                     temperature=1.0, top_k=64, top_p=0.95)
print(processor.tokenizer.decode(out[0][_t1_inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# Test 2: Real-time question (should emit tool call)
print("\n" + "─" * 60)
print("TEST 2: Real-time question (expect tool call)")
print("─" * 60)
_tc_text = processor.apply_chat_template(
    [{"role": "system", "content": SYS_TOOLS},
     {"role": "user", "content": "What's the current battery charge level?"}],
    tools=TOOL_SCHEMAS,
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
_tc_inputs = processor(text=_tc_text, return_tensors="pt").to(model.device)
_tc_out = model.generate(**_tc_inputs, max_new_tokens=200,
                         temperature=1.0, top_k=64, top_p=0.95)
_tc_raw = processor.tokenizer.decode(_tc_out[0][_tc_inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(f"Raw output: {_tc_raw[:400]}")
import re
_tc_calls = re.findall(r'call:(\w+)\{([^}]*)\}', _tc_raw)
if _tc_calls:
    print(f"✅ Tool call detected: {_tc_calls}")
else:
    print("⚠️  No tool call detected — model may need more tool-calling examples")

"""## 6b: Benchmark — Base vs Fine-Tuned"""

# === CELL 6b: Benchmark — Base vs Fine-Tuned ==================================
# Competition rules: "publish your benchmarks."
# 5 held-out Q&A questions + 3 tool-calling questions NOT in training data.
# Output is formatted for direct inclusion in Kaggle writeup.

BENCHMARK_QS = [
    "What happens to solar production when humidity exceeds 80%?",
    "At what battery SOC should we stop exporting to the grid?",
    "Home #3 has been underperforming by 22% for three weeks. What's the diagnostic checklist?",
    "It's winter in Ann Arbor and panels have snow. Prioritize actions.",
    "Grid frequency dropped to 59.8 Hz. What does that mean for our microgrid?",
]

# Tool-calling benchmark: held-out questions that SHOULD trigger tool calls
# Each entry: (question, set of acceptable tools OR None for no-tool expected)
# Uses set-based matching: any valid tool in the set counts as correct.
TOOL_BENCHMARK_QS = [
    ("What's the current battery state?", {"get_battery_state"}),
    ("How much solar are we producing right now in Seattle?", {"get_solar_production", "get_weather"}),
    ("What are the general maintenance tips for panels?", None),  # should NOT call a tool
]

def _run_benchmark(model, processor, label):
    """Generate answers for all benchmark questions."""
    results = []
    for q in BENCHMARK_QS:
        text = processor.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=300,
                             temperature=1.0, top_k=64, top_p=0.95)
        answer = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append(answer)
    return results

def _run_tool_benchmark(model, processor):
    """Test tool-calling behavior on held-out questions."""
    results = []
    for q, expected_tool in TOOL_BENCHMARK_QS:
        text = processor.apply_chat_template(
            [{"role": "system", "content": SYS_TOOLS},
             {"role": "user", "content": q}],
            tools=TOOL_SCHEMAS,
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=200,
                             temperature=1.0, top_k=64, top_p=0.95)
        raw = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        calls = re.findall(r'call:(\w+)\{([^}]*)\}', raw)
        called_tools = [c[0] for c in calls] if calls else []
        results.append((q, expected_tool, called_tools, raw))
    return results

# Run fine-tuned model (already in inference mode from Cell 6)
print("Running benchmark on fine-tuned model...")
ft_answers = _run_benchmark(model, processor, "FINE-TUNED")

# Print Q&A benchmark results
print("\n" + "=" * 70)
print("BENCHMARK: Fine-Tuned Model — 5 Held-Out Q&A Questions")
print("=" * 70)
for i, (q, ft) in enumerate(zip(BENCHMARK_QS, ft_answers), 1):
    print(f"\n--- Q{i}: {q}")
    print(f"[FINE-TUNED]: {ft[:500]}")

# Print tool-calling benchmark results
print("\n" + "=" * 70)
print("BENCHMARK: Fine-Tuned Model — Tool-Calling Held-Out Questions")
print("=" * 70)
tc_results = _run_tool_benchmark(model, processor)
_tc_correct = 0
for q, expected, called, raw in tc_results:
    if expected is None:
        passed = len(called) == 0
        expect_str = "no tool call"
    else:
        passed = bool(set(called) & expected)  # any called tool in acceptable set
        expect_str = " or ".join(f"call:{t}" for t in sorted(expected))
    _tc_correct += int(passed)
    status = "✅" if passed else "❌"
    print(f"\n{status} Q: {q}")
    print(f"   Expected: {expect_str}")
    print(f"   Got:      {called if called else 'direct answer'}")
    if not passed:
        print(f"   Raw:      {raw[:200]}")

print(f"\nTool-calling accuracy: {_tc_correct}/{len(TOOL_BENCHMARK_QS)}")
print("\n" + "=" * 70)
print("✅ Benchmark complete — copy above for Kaggle writeup")
print("=" * 70)

"""## 7: Export for Ollama + Push to HuggingFace"""

# === CELL 7: Export for Ollama + Push to HuggingFace ==========================
# Competition rules: "If training a model, publish your weights and benchmarks."
# HF_TOKEN loaded in Cell 1 from Kaggle Secrets or Colab userdata.
#
# Publishes to Truthseeker87/solarhive-e4b-ollama (in-place refresh of the
# safetensors source repo). The companion GGUF deployment repo
# (Truthseeker87/solarhive-e4b-gguf) is updated by Cell 14 below.
HF_REPO = "Truthseeker87/solarhive-e4b-ollama"

# Save LoRA adapters locally
# Per Unsloth Vision reference (Cell 37): pass `processor`, not `tokenizer`.
# The processor includes the image processor + tokenizer; saving only the
# tokenizer would lose the vision preprocessing config required for
# multimodal inference.
model.save_pretrained("solarhive_lora")
processor.save_pretrained("solarhive_lora")
print("✅ LoRA adapters saved to solarhive_lora/")

# Copy E4B LoRA to Drive for persistence across sessions
_drive_e4b_lora_path = "/content/drive/MyDrive/models/solarhive_e4b_lora"
try:
    import shutil as _shutil_e4b
    _shutil_e4b.copytree("solarhive_lora", _drive_e4b_lora_path, dirs_exist_ok=True)
    print(f"✅ E4B LoRA copied to Google Drive: {_drive_e4b_lora_path}")
except Exception as e:
    print(f"⚠️  Drive copy failed (non-blocking): {e}")
    print("   Manually copy: !cp -r solarhive_lora /content/drive/MyDrive/models/")

# Export for Ollama: try GGUF first, fall back to merged 16-bit safetensors.
# Ollama imports safetensors directly — no GGUF conversion needed.
# GGUF export may fail on Gemma 4 VLM (mmproj issue); safetensors fallback is reliable.
# Per Unsloth Vision reference (Cell 41): pass `processor`, not `tokenizer`,
# so vision preprocessing config is preserved in the merged output.
import os as _os
try:
    model.save_pretrained_gguf("solarhive_gguf", processor, quantization_method="q4_k_m")
    print("✅ GGUF export saved to solarhive_gguf/")
except Exception as _gguf_err:
    print(f"⚠️  GGUF export failed: {_gguf_err}")
    print("   Falling back to merged 16-bit HF format (convert to GGUF manually later).")
    model.save_pretrained_merged("solarhive_gguf", processor, save_method="merged_16bit")
    print("✅ Merged 16-bit model saved to solarhive_gguf/")

# List exported files
import glob as _glob
_gguf_files = _glob.glob("solarhive_gguf/*.gguf")
for f in _gguf_files:
    _sz = _os.path.getsize(f) / 1e9
    print(f"   {f} ({_sz:.1f} GB)")

# Copy exported model to Drive for persistence
_drive_gguf_path = "/content/drive/MyDrive/models/solarhive_e4b_ollama"
try:
    import shutil as _shutil_gguf
    _os.makedirs(_drive_gguf_path, exist_ok=True)
    for _f in _os.listdir("solarhive_gguf"):
        _src = f"solarhive_gguf/{_f}"
        if _os.path.isfile(_src):
            _shutil_gguf.copy2(_src, _drive_gguf_path)
    print(f"✅ Model copied to Google Drive: {_drive_gguf_path}")
except Exception as e:
    print(f"⚠️  Drive copy failed (non-blocking): {e}")
    print("   Manually copy: !cp -r solarhive_gguf /content/drive/MyDrive/models/solarhive_e4b_ollama")

# Push to HuggingFace (required for competition: "publish your weights")
# Safe to skip on first run — set HF_REPO and add HF_TOKEN to Colab Secrets when ready.
try:
    if HF_REPO.startswith("YOUR_USERNAME"):
        print("⚠️  HF push skipped — update HF_REPO with your HuggingFace username first")
    elif _gguf_files:
        # Per Unsloth Vision reference: pass `processor` (preserves vision config)
        model.push_to_hub_gguf(HF_REPO, processor,
                                quantization_method="q4_k_m", token=HF_TOKEN)
        print(f"✅ Pushed GGUF to https://huggingface.co/{HF_REPO}")
    else:
        # Fallback: push merged model if GGUF export failed
        from huggingface_hub import HfApi
        _api = HfApi(token=HF_TOKEN)
        _api.create_repo(HF_REPO, exist_ok=True, private=False)
        _api.upload_folder(folder_path="solarhive_gguf", repo_id=HF_REPO, token=HF_TOKEN)
        print(f"✅ Pushed merged model to https://huggingface.co/{HF_REPO}")
except Exception as e:
    print(f"⚠️  HF push failed (non-blocking): {e}")
    print("   Add HF_TOKEN to Colab Secrets and re-run this cell when ready.")

print(f"""
✅ Export complete! {'HF push: pending — update HF_REPO' if HF_REPO.startswith('YOUR_USERNAME') else f'HF: https://huggingface.co/{HF_REPO}'}

Ollama deployment:
  1. Copy solarhive_gguf/ folder to your machine
  2. Create Modelfile:
       FROM ./solarhive_gguf
       SYSTEM "You are SolarHive, an AI energy advisor for a community solar microgrid."
  3. ollama create solarhive -f Modelfile
  4. ollama run solarhive

Note: Ollama imports safetensors directly — no GGUF conversion needed.
""")


"""## Part B: Fine-Tune 26B A4B (Cloud Inference Model)

Same training data, second model. The fine-tuned 26B A4B powers
the live inference demo with SolarHive domain expertise.

Run AFTER Part A (E4B) completes. The E4B model is released from GPU memory
before loading 26B A4B. Both fit on a single GPU sequentially (not concurrently).
"""

"""## 8: Release E4B + Load 26B A4B"""

# === CELL 8: Release E4B + Load 26B A4B ======================================
# Free GPU memory from E4B fine-tuning before loading the larger 26B A4B model.
import gc
for _var in ("model", "trainer"):
    if _var in dir():
        del globals()[_var]
gc.collect()
torch.cuda.empty_cache()
print(f"GPU memory freed: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB available")

# Load 26B A4B from Google Drive cache (same pattern as inference.py)
_DRIVE_A4B_PATH = "/content/drive/MyDrive/models/gemma-4/transformers/gemma-4-26b-a4b-it/1"

if os.path.isdir(_DRIVE_A4B_PATH):
    A4B_MODEL_PATH = _DRIVE_A4B_PATH
    print(f"✅ Loading 26B A4B from Google Drive cache: {A4B_MODEL_PATH}")
else:
    print("26B A4B not in Drive — downloading via kagglehub (this takes 10-20 min)...")
    print("   (output suppressed to prevent Colab display overflow)")
    _dl_start = _time.time()
    from IPython.utils.capture import capture_output as _capture
    with _capture():
        A4B_MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-26b-a4b-it")
    print(f"✅ Download complete in {_time.time() - _dl_start:.0f}s: {A4B_MODEL_PATH}")
    import shutil as _shutil
    _cache_src = "/root/.cache/kagglehub/models/google/gemma-4"
    if os.path.isdir(_cache_src) and os.path.isdir("/content/drive"):
        print("Caching model to Google Drive (next session will load instantly)...")
        _shutil.copytree(_cache_src, "/content/drive/MyDrive/models/gemma-4", dirs_exist_ok=True)
        print("✅ Cached to Drive")

# VRAM auto-detect: 26B A4B needs 4-bit on <55GB VRAM, BF16 on ≥55GB
_free_vram_gb = torch.cuda.mem_get_info(0)[0] / 1e9
_a4b_use_4bit = _free_vram_gb < 55

if _a4b_use_4bit:
    print(f"GPU has {_free_vram_gb:.0f} GB free — loading 26B A4B in 4-bit QLoRA")
else:
    print(f"GPU has {_free_vram_gb:.0f} GB free — loading 26B A4B in BF16")

a4b_model, a4b_processor = FastVisionModel.from_pretrained(
    model_name=A4B_MODEL_PATH,
    load_in_4bit=_a4b_use_4bit,
    use_gradient_checkpointing="unsloth",
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    full_finetuning=False,
)
print(f"✅ Gemma 4 26B A4B loaded ({'4-bit NF4' if _a4b_use_4bit else 'BF16'})")

# 26B A4B uses the chat template bundled with the Kaggle-distributed Gemma 4
# 26B A4B model (in tokenizer_config.json) — same source-of-truth approach
# as Cell 2 for the E4B model.
a4b_tokenizer = a4b_processor.tokenizer  # text-ops alias

"""## 9: Attach LoRA Adapters to 26B A4B"""

# === CELL 9: Attach LoRA Adapters to 26B A4B =================================
# Same as Cell 3 — vision layers off for text-only training.
a4b_model = FastVisionModel.get_peft_model(
    a4b_model,
    finetune_vision_layers=False,      # text-only training — vision encoder unused
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules="all-linear",
)
a4b_trainable, a4b_total = a4b_model.get_nb_trainable_parameters()
print(f"26B A4B — Trainable: {a4b_trainable:,} / {a4b_total:,} ({100*a4b_trainable/a4b_total:.2f}%)")

"""## 10: Train 26B A4B"""

# === CELL 10: Train 26B A4B ===================================================
# Same training shape as Cell 5 but with the A4B tokenizer/processor. The
# dataset is re-loaded and pre-rendered into a `text` column using
# `a4b_processor.apply_chat_template(...)` so the resulting strings carry the
# 26B A4B model's specific chat-template tokens. SFTTrainer then trains with
# TRL's default text collator on that pre-rendered text.
#
# 26B A4B VRAM: ~25-31 GB (NF4) or ~56 GB (BF16) — smaller batches needed
# on tight VRAM.

from datasets import load_dataset as _load_dataset_a4b
a4b_dataset = _load_dataset_a4b(
    V2_DATASET_REPO,
    split="train",
    token=HF_TOKEN if "HF_TOKEN" in dir() else None,
)
print(f"26B A4B training dataset: {len(a4b_dataset)} rows")
print(f"  Schema: {list(a4b_dataset.features.keys())}")

# Pre-render the dataset rows for A4B using the A4B-specific tokenizer.
# Mirror of Cell 4b's pre-render pattern but using `a4b_processor` so the
# template + tokenization produce the exact 26B A4B chat shape (same source
# data, different tokenizer = different output).
a4b_rendered = []
a4b_skipped = 0
for _r in a4b_dataset:
    _t = _render_row(_r, processor_to_use=a4b_processor)
    if _t is None:
        a4b_skipped += 1
        continue
    a4b_rendered.append(_t)
a4b_dataset = _Dataset.from_dict({"text": a4b_rendered})
print(f"  Pre-rendered {len(a4b_dataset)} text rows for A4B SFT.")
print(f"  Skipped {a4b_skipped} rows (image-modality + render failures).")

# Save the pre-rendered A4B training set to Drive for audit + reproducibility.
try:
    _a4b_jsonl = f"{_ds_drive_dir}/train_a4b_{_ds_timestamp}.jsonl"
    with open(_a4b_jsonl, "w", encoding="utf-8") as f:
        for _t in a4b_dataset["text"]:
            f.write(json.dumps({"text": _t}, ensure_ascii=False) + "\n")
    _a4b_hf = f"{_ds_drive_dir}/train_a4b_{_ds_timestamp}"
    a4b_dataset.save_to_disk(_a4b_hf)
    print(f"✅ 26B A4B dataset saved: {_a4b_jsonl}")
except Exception as e:
    print(f"⚠️  26B A4B dataset save skipped (non-blocking): {e}")
    print(f"   Source of truth is the HF dataset repo: {V2_DATASET_REPO}")

# Batch size for 26B A4B (larger model than E4B, smaller batches needed)
_a4b_vram = torch.cuda.mem_get_info(0)[0] / 1e9
if _a4b_vram >= 55:
    _a4b_batch, _a4b_accum = 2, 8   # ≥55GB VRAM (BF16): effective 16
elif _a4b_vram >= 25:
    _a4b_batch, _a4b_accum = 1, 8   # <55GB VRAM (NF4): effective 8 (tight)
else:
    _a4b_batch, _a4b_accum = 1, 4   # Fallback
print(f"26B A4B training: batch={_a4b_batch}, accum={_a4b_accum} "
      f"(effective {_a4b_batch * _a4b_accum}, VRAM: {_a4b_vram:.0f} GB free)")

a4b_trainer = SFTTrainer(
    model=a4b_model,
    processing_class=a4b_processor.tokenizer,
    train_dataset=a4b_dataset,
    args=SFTConfig(
        per_device_train_batch_size=_a4b_batch,
        gradient_accumulation_steps=_a4b_accum,
        warmup_steps=5,                    # Unsloth Gemma 4 text-only default
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="solarhive_a4b_out",
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        # max_grad_norm left at TRL default (1.0)
        seed=3407,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
    ),
)

# Same train_on_responses_only wrapper as Cell 5 (E4B). The wrapper uses
# the same `<|turn>` markers — the 26B A4B chat template uses the same
# turn-token convention as E4B.
try:
    from unsloth.chat_templates import train_on_responses_only as _tor_a4b
    a4b_trainer = _tor_a4b(
        a4b_trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )
    print("✅ train_on_responses_only applied to A4B trainer.")
except Exception as _tor_err:
    print(f"⚠️  A4B train_on_responses_only failed ({_tor_err}); "
          f"falling back to full-sequence loss.")

a4b_stats = a4b_trainer.train()
_a4b_log = [x["loss"] for x in a4b_trainer.state.log_history if "loss" in x]
_a4b_last20 = _a4b_log[-20:] if len(_a4b_log) >= 20 else _a4b_log
print(f"✅ 26B A4B done — {a4b_stats.metrics['train_runtime']:.0f}s, {len(_a4b_log)} steps")
print(f"   Converged loss (last 20 avg): {sum(_a4b_last20)/len(_a4b_last20):.4f}")
print(f"   Final step: {_a4b_log[-1]:.4f} | Min: {min(_a4b_log):.4f} | HF avg (all steps): {a4b_stats.training_loss:.4f}")

"""## 11: Test + Benchmark 26B A4B"""

# === CELL 11: Test + Benchmark 26B A4B ========================================
FastVisionModel.for_inference(a4b_model)

# Test 1: General knowledge
print("─" * 60)
print("26B A4B TEST 1: General knowledge")
print("─" * 60)
_a4b_t1_text = a4b_processor.apply_chat_template(
    [{"role": "system", "content": SYS_TOOLS},
     {"role": "user", "content": "It's snowing and panels are covered. What now?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
_a4b_inputs = a4b_processor(text=_a4b_t1_text, return_tensors="pt").to(a4b_model.device)

_a4b_out = a4b_model.generate(**_a4b_inputs, max_new_tokens=300,
                               temperature=1.0, top_k=64, top_p=0.95)
print(a4b_processor.tokenizer.decode(_a4b_out[0][_a4b_inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# Test 2: Tool call
print("\n" + "─" * 60)
print("26B A4B TEST 2: Real-time question (expect tool call)")
print("─" * 60)
_a4b_tc_text = a4b_processor.apply_chat_template(
    [{"role": "system", "content": SYS_TOOLS},
     {"role": "user", "content": "What's the current battery charge level?"}],
    tools=TOOL_SCHEMAS,
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
_a4b_tc_inputs = a4b_processor(text=_a4b_tc_text, return_tensors="pt").to(a4b_model.device)
_a4b_tc_out = a4b_model.generate(**_a4b_tc_inputs, max_new_tokens=200,
                                  temperature=1.0, top_k=64, top_p=0.95)
_a4b_tc_raw = a4b_processor.tokenizer.decode(_a4b_tc_out[0][_a4b_tc_inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(f"Raw output: {_a4b_tc_raw[:400]}")
_a4b_tc_calls = re.findall(r'call:(\w+)\{([^}]*)\}', _a4b_tc_raw)
if _a4b_tc_calls:
    print(f"✅ Tool call detected: {_a4b_tc_calls}")
else:
    print("⚠️  No tool call detected")

# Q&A Benchmark
print("\n" + "=" * 70)
print("BENCHMARK: 26B A4B Fine-Tuned — 5 Held-Out Q&A Questions")
print("=" * 70)

a4b_ft_answers = _run_benchmark(a4b_model, a4b_processor, "26B-A4B-FT")
for i, (q, ft) in enumerate(zip(BENCHMARK_QS, a4b_ft_answers), 1):
    print(f"\n--- Q{i}: {q}")
    print(f"[26B A4B FINE-TUNED]: {ft[:500]}")

# Tool-calling benchmark
print("\n" + "=" * 70)
print("BENCHMARK: 26B A4B Fine-Tuned — Tool-Calling Held-Out Questions")
print("=" * 70)
a4b_tc_results = _run_tool_benchmark(a4b_model, a4b_processor)
_a4b_tc_correct = 0
for q, expected, called, raw in a4b_tc_results:
    if expected is None:
        passed = len(called) == 0
        expect_str = "no tool call"
    else:
        passed = bool(set(called) & expected)
        expect_str = " or ".join(f"call:{t}" for t in sorted(expected))
    _a4b_tc_correct += int(passed)
    status = "✅" if passed else "❌"
    print(f"\n{status} Q: {q}")
    print(f"   Expected: {expect_str}")
    print(f"   Got:      {called if called else 'direct answer'}")
    if not passed:
        print(f"   Raw:      {raw[:200]}")

print(f"\n26B A4B tool-calling accuracy: {_a4b_tc_correct}/{len(TOOL_BENCHMARK_QS)}")
print("\n" + "=" * 70)
print("✅ 26B A4B benchmark complete")
print("=" * 70)

"""## 12: Save 26B A4B LoRA Adapters"""

# === CELL 12: Save 26B A4B LoRA Adapters ======================================
# Save adapters for inference.py to load. Copy to Google Drive for persistence.
# Per Unsloth Vision reference: pass `processor`, not `tokenizer` — the
# processor includes the image processor + tokenizer, so saving it preserves
# the multimodal preprocessing config required by inference.py at load time.
a4b_model.save_pretrained("solarhive_a4b_lora")
a4b_processor.save_pretrained("solarhive_a4b_lora")
print("✅ 26B A4B LoRA adapters saved to solarhive_a4b_lora/")

# Copy to Drive for persistence across sessions
_drive_lora_path = "/content/drive/MyDrive/models/solarhive_a4b_lora"
try:
    import shutil
    shutil.copytree("solarhive_a4b_lora", _drive_lora_path, dirs_exist_ok=True)
    print(f"✅ Copied to Google Drive: {_drive_lora_path}")
    print("   inference.py can load these adapters for the fine-tuned demo.")
except Exception as e:
    print(f"⚠️  Drive copy failed (non-blocking): {e}")
    print("   Manually copy: !cp -r solarhive_a4b_lora /content/drive/MyDrive/models/")

print("""
✅ Dual fine-tune complete!

Part A: E4B fine-tuned → GGUF → Ollama (edge deployment)
Part B: 26B A4B fine-tuned → LoRA adapters (cloud inference demo)

To use in inference.py, load the base 26B A4B model then apply LoRA:
  from peft import PeftModel
  model = PeftModel.from_pretrained(model, "solarhive_a4b_lora")
""")

"""## 12b: Push 26B A4B LoRA Adapters to HuggingFace"""

# === CELL 12b: Push 26B A4B LoRA Adapters to HuggingFace ======================
# Pushes the LoRA adapters saved by Cell 12 to HuggingFace as a LoRA-only
# repo. No model reload, no merge — `solarhive_inference.py` loads base + LoRA
# at inference time via FastVisionModel.from_pretrained(lora_path).
#
# Prerequisites:
#   - Cell 12 has run and `solarhive_a4b_lora/` exists locally (or on Drive)
#   - HF_TOKEN set in Colab secrets with write access
#
# Target: Truthseeker87/solarhive-26b-a4b-lora (the published LoRA repo).
# By default this preserves the existing curated model card and only updates
# the weight files. Set PRESERVE_EXISTING_CARD = False to also overwrite the
# README with the auto-generated one (after metadata patching below).

import os, re, shutil

_a4b_lora_path = "solarhive_a4b_lora"
_DRIVE_LORA = "/content/drive/MyDrive/models/solarhive_a4b_lora"
HF_REPO_A4B = "Truthseeker87/solarhive-26b-a4b-lora"

# Set False to overwrite the existing curated model card with the auto-
# generated Unsloth README. Default True keeps the curated card intact and
# only refreshes the weight files.
PRESERVE_EXISTING_CARD = True

# Locate the LoRA folder. Prefer local (Cell 12 just wrote it). Fall back to
# Drive (lets this cell run standalone on a fresh runtime if Cell 12 already
# copied to Drive in a prior session).
if not os.path.isdir(_a4b_lora_path) and os.path.isdir(_DRIVE_LORA):
    print(f"Local {_a4b_lora_path}/ not found — copying from Drive")
    shutil.copytree(_DRIVE_LORA, _a4b_lora_path)

assert os.path.isdir(_a4b_lora_path), (
    f"LoRA adapters not found at {_a4b_lora_path} or {_DRIVE_LORA}.\n"
    "  Run Cell 12 first to save adapters."
)
_lora_size_mb = sum(
    os.path.getsize(os.path.join(_a4b_lora_path, f))
    for f in os.listdir(_a4b_lora_path)
    if os.path.isfile(os.path.join(_a4b_lora_path, f))
) / 1e6
print(f"✅ LoRA adapters found: {_a4b_lora_path} ({_lora_size_mb:.0f} MB)")

# Patch the auto-generated README so HF accepts it (only matters if uploaded).
# Unsloth writes `base_model: /content/drive/...` (a local path) which HF
# rejects with "not a valid model id". Rewrite to the canonical HF model id.
_readme_path = os.path.join(_a4b_lora_path, "README.md")
if os.path.exists(_readme_path):
    with open(_readme_path, "r", encoding="utf-8") as f:
        _readme = f.read()
    _patched = re.sub(
        r'^base_model:\s*.*$',
        'base_model: google/gemma-4-26b-a4b-it',
        _readme, count=1, flags=re.MULTILINE,
    )
    if _patched != _readme:
        with open(_readme_path, "w", encoding="utf-8") as f:
            f.write(_patched)
        print("✅ Patched README.md base_model → google/gemma-4-26b-a4b-it")

# Resolve HF_TOKEN. Cell 1 already loaded HF_TOKEN into globals on the active
# runtime; this fallback chain re-resolves it for the standalone-runtime case.
_hf_token = HF_TOKEN if "HF_TOKEN" in dir() and HF_TOKEN else None
if not _hf_token:
    try:
        from google.colab import userdata
        _hf_token = userdata.get("HF_TOKEN")
    except Exception:
        _hf_token = os.environ.get("HF_TOKEN")

if not _hf_token:
    print("⚠️  HF_TOKEN not found — skipping upload.")
    print(f"   Manual upload: huggingface-cli upload {HF_REPO_A4B} {_a4b_lora_path}")
else:
    try:
        from huggingface_hub import HfApi
        _api = HfApi(token=_hf_token)
        _api.create_repo(HF_REPO_A4B, exist_ok=True, private=False, repo_type="model")
        _ignore = ["README.md"] if PRESERVE_EXISTING_CARD else None
        _msg_card = "preserving existing card" if PRESERVE_EXISTING_CARD else "OVERWRITING card"
        print(f"Uploading LoRA adapters to {HF_REPO_A4B} ({_msg_card})...")
        _api.upload_folder(
            folder_path=_a4b_lora_path,
            repo_id=HF_REPO_A4B,
            token=_hf_token,
            ignore_patterns=_ignore,
        )
        print(f"✅ Pushed to https://huggingface.co/{HF_REPO_A4B}")
        print("   solarhive_inference.py loads this via FastVisionModel.from_pretrained(lora_path).")
    except Exception as e:
        print(f"⚠️  Upload failed (non-blocking): {e}")
        print(f"   Manual upload: huggingface-cli upload {HF_REPO_A4B} {_a4b_lora_path}")

"""## 13: Ollama Deployment"""

# === CELL 13: Ollama Deployment ===============================================
# Unsloth's save_pretrained_gguf may fail on Gemma 4 VLM models because its
# llama.cpp tries to convert the vision projector (--mmproj). Cell 7's fallback
# saves a merged 16-bit HF model (safetensors) instead.
#
# Two Ollama import paths (https://docs.ollama.com/import):
#   Path A: GGUF — smaller file, faster Ollama inference (preferred for distribution)
#   Path B: Safetensors — Ollama imports HF models directly (guaranteed fallback)
#
# Gemma is explicitly supported for both import methods.

import os as _os, glob as _glob, subprocess, shutil

_gguf_dir = "solarhive_gguf"
_drive_gguf = "/content/drive/MyDrive/models/solarhive_e4b_ollama"

# Check if GGUF already exists from Cell 7
_existing_gguf = _glob.glob(f"{_gguf_dir}/*.gguf")
if _existing_gguf:
    print("✅ GGUF files already exist from Cell 7 — skipping conversion.")
    for f in _existing_gguf:
        print(f"   {f} ({_os.path.getsize(f)/1e9:.1f} GB)")
else:
    # --- Path A: Try manual GGUF conversion via llama.cpp ---
    # Unsloth installs llama.cpp during Cell 7 (even if GGUF export fails)
    _LLAMA_CPP = "/root/.unsloth/llama.cpp"
    _CONVERTER = f"{_LLAMA_CPP}/convert_hf_to_gguf.py"
    _QUANTIZE = f"{_LLAMA_CPP}/llama-quantize"

    _gguf_ok = False
    if _os.path.exists(_CONVERTER) and _os.path.exists(_QUANTIZE):
        print("Attempting GGUF conversion (text-only, skipping vision projector)...")
        _bf16_gguf = f"{_gguf_dir}/solarhive-e4b-bf16.gguf"
        _q4km_gguf = f"{_gguf_dir}/solarhive-e4b-q4_k_m.gguf"

        try:
            # Step 1: HF safetensors → BF16 GGUF (no --mmproj flag)
            print("  Step 1/2: HF → BF16 GGUF...")
            subprocess.run(
                ["python", _CONVERTER, "--outfile", _bf16_gguf,
                 "--outtype", "bf16", "--split-max-size", "50G", _gguf_dir],
                capture_output=True, text=True, timeout=600, check=True)
            print(f"  ✅ BF16: {_bf16_gguf} ({_os.path.getsize(_bf16_gguf)/1e9:.1f} GB)")

            # Step 2: Quantize BF16 → Q4_K_M
            print("  Step 2/2: BF16 → Q4_K_M...")
            subprocess.run(
                [_QUANTIZE, _bf16_gguf, _q4km_gguf, "Q4_K_M"],
                capture_output=True, text=True, timeout=600, check=True)
            print(f"  ✅ Q4_K_M: {_q4km_gguf} ({_os.path.getsize(_q4km_gguf)/1e9:.1f} GB)")

            # Clean up BF16 intermediate
            _os.remove(_bf16_gguf)
            _gguf_ok = True
        except (subprocess.CalledProcessError, Exception) as e:
            print(f"  ⚠️  GGUF conversion failed: {e}")
            # Clean up partial files
            for _f in [_bf16_gguf, _q4km_gguf]:
                if _os.path.exists(_f):
                    _os.remove(_f)
    else:
        print("llama.cpp not found — Cell 7 may not have run yet.")

    # --- Path B: Safetensors direct import (guaranteed fallback) ---
    if not _gguf_ok:
        print("\n📦 Using Ollama safetensors import (no GGUF conversion needed).")
        print("   Ollama natively imports HF safetensors models for Gemma architectures.")
        print(f"   Merged model location: /content/{_gguf_dir}/")

# Copy to Drive for persistence
try:
    _os.makedirs(_drive_gguf, exist_ok=True)
    # Copy all model files (gguf or safetensors)
    for _f in _os.listdir(_gguf_dir):
        _src = f"{_gguf_dir}/{_f}"
        if _os.path.isfile(_src):
            shutil.copy2(_src, _drive_gguf)
    print(f"✅ Model files copied to Google Drive: {_drive_gguf}")
except Exception as e:
    print(f"⚠️  Drive copy failed (non-blocking): {e}")

# Deployment instructions
_has_gguf = bool(_glob.glob(f"{_gguf_dir}/*.gguf"))
if _has_gguf:
    _gguf_file = _glob.glob(f"{_gguf_dir}/*.gguf")[0]
    print(f"""
✅ GGUF export complete!

Ollama deployment (from GGUF):
  1. Copy {_gguf_file} to your machine
  2. Create Modelfile:
       FROM ./{_os.path.basename(_gguf_file)}
       SYSTEM "You are SolarHive, an AI energy advisor for a community of 12 homes with rooftop solar and shared battery storage in Ann Arbor, Michigan."
  3. ollama create solarhive -f Modelfile
  4. ollama run solarhive
""")
else:
    print(f"""
✅ Merged model ready for Ollama!

Ollama deployment (from safetensors — https://docs.ollama.com/import):
  1. Copy {_gguf_dir}/ folder to your machine
  2. Create Modelfile in the model folder:
       FROM .
       SYSTEM "You are SolarHive, an AI energy advisor for a community of 12 homes with rooftop solar and shared battery storage in Ann Arbor, Michigan."
  3. ollama create solarhive -f Modelfile
  4. ollama run solarhive
""")

"""## 14: Push E4B GGUF to HuggingFace"""

# === CELL 14: Push E4B GGUF to HuggingFace ====================================
# Pushes the Q4_K_M GGUF produced by Cell 13 to the deployment repo
# Truthseeker87/solarhive-e4b-gguf. No GPU work, no merge — just an upload.
#
# Prerequisites:
#   - Cell 13 has run and `solarhive_gguf/solarhive-e4b-q4_k_m.gguf` exists
#     locally (or in the Drive copy at /content/drive/MyDrive/models/solarhive_e4b_ollama)
#   - HF_TOKEN set in Colab secrets with write access
#
# Target: Truthseeker87/solarhive-e4b-gguf (the GGUF deployment repo).
# By default this preserves the existing curated model card and only uploads
# the GGUF file. Set PRESERVE_EXISTING_CARD = False to also upload a README.

import os, glob

HF_REPO_GGUF = "Truthseeker87/solarhive-e4b-gguf"
PRESERVE_EXISTING_CARD = True

# Locate the Q4_K_M GGUF. Prefer local (Cell 13 just produced it). Fall back
# to the Drive copy at /content/drive/MyDrive/models/solarhive_e4b_ollama
# (where Cell 13 mirrored the GGUF for cross-session persistence).
_local_dir = "solarhive_gguf"
_drive_dir = "/content/drive/MyDrive/models/solarhive_e4b_ollama"
_gguf_filename = "solarhive-e4b-q4_k_m.gguf"

_gguf_path = None
for _candidate in (
    os.path.join(_local_dir, _gguf_filename),
    os.path.join(_drive_dir, _gguf_filename),
):
    if os.path.isfile(_candidate):
        _gguf_path = _candidate
        break

if _gguf_path is None:
    # Last-ditch: glob for any *.gguf in the local dir
    _glob_hits = glob.glob(os.path.join(_local_dir, "*.gguf"))
    if _glob_hits:
        _gguf_path = _glob_hits[0]

assert _gguf_path is not None, (
    f"GGUF not found at {_local_dir}/{_gguf_filename} or {_drive_dir}/{_gguf_filename}.\n"
    "  Run Cell 13 first to produce the Q4_K_M GGUF."
)
_gguf_size_gb = os.path.getsize(_gguf_path) / 1e9
print(f"✅ GGUF found: {_gguf_path} ({_gguf_size_gb:.2f} GB)")

# Resolve HF_TOKEN. Cell 1 already loaded HF_TOKEN into globals on the active
# runtime; this fallback chain re-resolves it for the standalone-runtime case.
_hf_token = HF_TOKEN if "HF_TOKEN" in dir() and HF_TOKEN else None
if not _hf_token:
    try:
        from google.colab import userdata
        _hf_token = userdata.get("HF_TOKEN")
    except Exception:
        _hf_token = os.environ.get("HF_TOKEN")

if not _hf_token:
    print("⚠️  HF_TOKEN not found — skipping upload.")
    print(f"   Manual upload: huggingface-cli upload {HF_REPO_GGUF} {_gguf_path}")
else:
    try:
        from huggingface_hub import HfApi
        _api = HfApi(token=_hf_token)
        _api.create_repo(HF_REPO_GGUF, exist_ok=True, private=False, repo_type="model")
        _msg_card = "preserving existing card" if PRESERVE_EXISTING_CARD else "OVERWRITING card"
        print(f"Uploading GGUF to {HF_REPO_GGUF} ({_msg_card})...")
        _api.upload_file(
            path_or_fileobj=_gguf_path,
            path_in_repo=_gguf_filename,
            repo_id=HF_REPO_GGUF,
            token=_hf_token,
        )
        print(f"✅ Pushed {_gguf_filename} to https://huggingface.co/{HF_REPO_GGUF}")
        print("   Companion files (mmproj, Modelfiles, alternate-quant variants) on the repo are unchanged.")
    except Exception as e:
        print(f"⚠️  Upload failed (non-blocking): {e}")
        print(f"   Manual upload: huggingface-cli upload {HF_REPO_GGUF} {_gguf_path}")
