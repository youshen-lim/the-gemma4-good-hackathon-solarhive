# -*- coding: utf-8 -*-
"""SolarHive — Fine-tuned Gemma 4 E4B → Cactus Mobile Deployment Notebook
========================================================================
SolarHive is an open-source intelligence layer designed to coordinate
community microgrids & community-based storage via fuel cells, pool
midday energy surplus across these microgrids, and eliminate stranded
capacity. It also helps forecast solar irradiance and cloud cover to
plan ahead.

PURPOSE: Convert the fine-tuned Gemma 4 E4B merged safetensors to Cactus's
INT4 mobile deployment format, then smoke-test the artifact via the Cactus
Python SDK with SolarHive emoji-format prompts. The output artifact is
consumed by the companion Flutter Android app for on-device inference.

Pipeline:
    Truthseeker87/solarhive-e4b-ollama (BF16 safetensors, ~16 GB)
                ↓ cactus convert --precision INT4
    /content/solarhive-e4b-cactus (INT4 multimodal artifact, ~7 GB)
                ↓ Cactus Python SDK + 10 SolarHive prompts (ARM hosts)
    Three-outcome verdict — ship / iterate / fall back to base E4B

SETUP: Google Colab Pro CPU + High-RAM runtime (~25 GB system RAM
       recommended). No GPU required — Cactus convert is CPU-bound.
       Convert step takes ~5-15 min on Colab CPU+High-RAM.

       The Cactus runtime targets ARM platforms (Apple Silicon, iOS,
       Android, ARM64 Linux) — see the official Cactus Gemma 4 deployment
       blog at https://docs.cactuscompute.com/v1.14/blog/gemma4/. On x86
       development hosts the convert step still produces a deployable
       artifact that the mobile app loads on-device; the Python SDK smoke
       test runs on ARM hosts (Android emulator, Apple Silicon Mac,
       ARM cloud, Pi 5). The notebook detects the host architecture and
       gracefully skips ARM-only steps on x86.

PRIZE TARGET: Cactus Special Technology Track

Gemma is a trademark of Google LLC.

References:
- Cactus repository + supported-models table:
  https://github.com/cactus-compute/cactus
- Cactus convert CLI: `cactus convert <model> [dir] --precision INT4|INT8|FP16`
- Cactus Flutter SDK: https://pub.dev/packages/cactus
- Companion notebooks:
    solarhive_finetune.py     — produces the LoRA adapters
    solarhive_merge_e4b.py    — merges LoRA + base to BF16 safetensors
                                 published as Truthseeker87/solarhive-e4b-ollama
                                 (the source artifact this notebook consumes)
    solarhive_inference.py    — defines the SolarHive prompt set reused
                                 for the Class A / Class B smoke-test verdict

## 0: Dependencies & cactus-compute install
"""

"""## 0: Install cactus-compute from GitHub + verification gate

cactus-compute (the LLM framework targeted by this notebook) is distributed
via GitHub clone + editable install of its `python/` subdirectory. It is NOT
published to PyPI under any name; the PyPI package `cactus` belongs to an
unrelated static-site generator and a stray `pip install cactus` would
silently install the wrong tool.

This cell:
1. Sweeps any prior `cactus` / `cactus-compute` installs to clear name-
   collision risk and partial-install residue.
2. Clones the cactus-compute repository so its data files (notably
   `models.json`, which `cactus/python/src/cli.py` resolves at import time)
   sit alongside the editable install. A pip-from-subdirectory install
   would orphan those data files and cause runtime errors on first
   `cactus --help`.
3. Installs the `python/` subdirectory in editable mode (mirrors what
   the upstream `source ./setup` script does internally).
4. Patches the running kernel's `sys.path` so subsequent cells can import
   `from src.cactus import ...` without a kernel restart.
5. Verifies the resulting `cactus` CLI exposes the `convert` subcommand —
   guards against the name-collision regression returning silently.
6. Detects host architecture and runs `cactus build --python` only on ARM
   hosts (the Cactus C++ engine is ARM-only by design — see the host-arch
   gate at the end of Cell 0).
"""

import subprocess as _sp
import sys as _sys
import os
import shutil as _shutil

# Step 1 — install peripheral dependencies
_sp.check_call([
    _sys.executable, "-m", "pip", "install", "--upgrade", "-q",
    "huggingface_hub",
])

# Step 2 — sweep any prior cactus / cactus-compute installs
# Cleans up: (a) the static-site-generator landed by a `pip install cactus`
# name-collision, (b) any partial cactus-compute install that may have left
# a broken /usr/local/bin/cactus stub. Either uninstall is allowed to fail
# (returncode != 0 just means the package wasn't installed).
print("Removing any prior cactus packages (name-collision + partial-install cleanup)...")
for _pkg in ("cactus", "cactus-compute"):
    _r = _sp.run(
        [_sys.executable, "-m", "pip", "uninstall", "-y", _pkg],
        capture_output=True, text=True,
    )
    print(f"  pip uninstall {_pkg}: exit {_r.returncode}")

# Step 3 — clone the cactus-compute repository, then editable-install the
# python/ subdirectory. This mirrors what upstream's `source ./setup` script
# does internally. The clone-then-install pattern (rather than pip's
# `git+<url>#subdirectory=...` shortcut) is required because
# `cactus/python/src/cli.py` computes `PROJECT_ROOT = <repo>` and reads
# `<repo>/models.json` at import time; an install that only ships the
# python/ subtree to site-packages would leave that data file orphaned.
CACTUS_SRC_DIR = "/content/cactus-compute-src"
if os.path.isdir(CACTUS_SRC_DIR):
    print(f"\nRemoving stale clone at {CACTUS_SRC_DIR} ...")
    _shutil.rmtree(CACTUS_SRC_DIR)

print(f"\nCloning cactus-compute → {CACTUS_SRC_DIR} ...")
print("(progress streams below — should take 10-60s on Colab. If it hangs >3min,")
print(" click ⏹ Stop on this cell, Runtime → Restart session, and retry.)")
print()
# Clone design choices:
# - NO `capture_output=True` — git's progress streams to the cell so the
#   user sees Receiving/Resolving lines and can confirm the clone is alive
# - `--depth 1` shallow clone — only fetches the latest commit (~5-10× faster
#   than a full history clone)
# - `--single-branch` — only fetches main, skips other branch refs
# - `GIT_LFS_SKIP_SMUDGE=1` — skips LFS binary content; only the Python
#   package source + `models.json` text are needed
_clone_env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
_clone = _sp.run(
    ["git", "clone", "--depth", "1", "--single-branch",
     "https://github.com/cactus-compute/cactus.git", CACTUS_SRC_DIR],
    timeout=300,
    env=_clone_env,
)
if _clone.returncode != 0:
    raise SystemExit(
        f"\ngit clone failed (exit {_clone.returncode}). Possible causes:\n"
        "- Colab network blip — retry the cell\n"
        "- Repo URL changed — verify https://github.com/cactus-compute/cactus.git\n"
        "- LFS smudge re-enabled by .gitattributes despite env override\n"
        "Check the git output above for the actual error message."
    )
print(f"\n✅ Cloned ({len(os.listdir(CACTUS_SRC_DIR))} top-level entries in {CACTUS_SRC_DIR})")

print(f"\nRunning pip install -e {CACTUS_SRC_DIR}/python ...")
_install = _sp.run(
    [_sys.executable, "-m", "pip", "install", "-e", f"{CACTUS_SRC_DIR}/python"],
    capture_output=True, text=True, timeout=900,
)
if _install.returncode != 0:
    print(f"❌ Editable install failed (exit {_install.returncode}):")
    print(_install.stderr[-2000:])
    raise SystemExit(
        "cactus-compute editable install failed. The `python/` subdirectory "
        f"may have moved in the repo — inspect {CACTUS_SRC_DIR}/ for the "
        "current package layout. Their `setup` script may also need a different "
        "install target name now."
    )
print("✅ Editable install succeeded")

# Verify models.json sits where cli.py expects it (at the repo root,
# alongside the python/ subdir). Missing-data-file regressions surface here.
if os.path.isfile(f"{CACTUS_SRC_DIR}/models.json"):
    print(f"  ✅ models.json present at {CACTUS_SRC_DIR}/models.json")
else:
    print(f"  ⚠️  models.json NOT at {CACTUS_SRC_DIR}/models.json — package may "
          "expect it elsewhere now. Verification gate below will catch the "
          "actual import-time failure.")

# Patch the running kernel's sys.path so subsequent cells can do
# `from src.cactus import ...` without a kernel restart. `pip install -e`
# creates a .pth file in site-packages, but .pth files are only read at
# Python interpreter startup; since the kernel was already running when
# Cell 0 completed, its sys.path doesn't include the editable-install path
# yet. Adding the path explicitly mirrors what `import site; site.main()`
# would do and is more reliable in Colab's kernel.
_PY_PKG_DIR = f"{CACTUS_SRC_DIR}/python"
if _PY_PKG_DIR not in _sys.path:
    _sys.path.insert(0, _PY_PKG_DIR)
    print(f"  ✅ Added {_PY_PKG_DIR} to sys.path (so `from src.cactus import ...` works in this kernel)")
else:
    print(f"  ✅ {_PY_PKG_DIR} already on sys.path")

# Invalidate import caches so Python re-scans for the freshly-installed module
import importlib as _il
_il.invalidate_caches()

# Step 4 — verification gate: confirm `cactus --help` runs cleanly AND
# mentions the `convert` subcommand we'll need in Cell 3
print()
print("Verifying cactus CLI ...")
_cactus_path = _shutil.which("cactus")
print(f"  cactus binary: {_cactus_path}")

_help = _sp.run(["cactus", "--help"], capture_output=True, text=True, timeout=30)
_help_text = (_help.stdout or "") + (_help.stderr or "")
print(f"  --help exit: {_help.returncode}")
print(f"  --help output (first 800 chars):")
print("    " + _help_text[:800].replace("\n", "\n    "))

# Three failure modes the verification gate must catch:
# (a) Wrong package loaded — the unrelated PyPI `cactus` (static-site
#     generator) lists commands create/build/deploy/serve; absence of the
#     `convert` subcommand signals package collision.
# (b) Runtime ImportError on first CLI invocation — usually a missing data
#     file (e.g. models.json) when install left the package layout broken.
#     Symptom: traceback in stderr; --help exit non-zero.
# (c) Clean help output containing `convert` → install is OK, proceed.
_help_lc = _help_text.lower()
if _help.returncode != 0 or "traceback" in _help_lc or "filenotfounderror" in _help_lc:
    raise SystemExit(
        "FATAL — `cactus --help` raised a runtime error (likely a data-file or "
        "import issue inside the cactus-compute package). Inspect the traceback "
        f"in the output above. Common cause: the package was installed but a "
        f"required data file (e.g. models.json) is missing at the path the "
        f"package expects. Verify {CACTUS_SRC_DIR}/models.json exists and the "
        f"editable-install .pth points at {CACTUS_SRC_DIR}/python. If those look "
        "right, cactus-compute may have shipped a packaging regression — file "
        "an upstream issue with the traceback above."
    )
if "convert" not in _help_lc:
    raise SystemExit(
        "FATAL — `cactus --help` ran cleanly but does not mention 'convert'. "
        "The wrong cactus package is loaded. The eudicots static-site-generator "
        "lists commands create/build/deploy/serve — those would error on "
        "Cell 3's `cactus convert` invocation. Try Runtime → Restart session, "
        "then re-run from Cell 0."
    )

print()
print("✅ cactus-compute LLM framework installed and verified ('convert' in help, "
      "no runtime errors on import)")

# ============================================================================
# Step 5 — Host-arch-gated `cactus build --python` (ARM only)
# ============================================================================
#
# `cactus build --python` compiles libcactus.so, the C++ engine the Python
# SDK FFIs into. The Cactus C++ engine targets ARM platforms by design:
# CMakeLists hardcodes `-march=armv8.2-a+i8mm` and the SIMD kernels (e.g.
# kernel_i8mm.cpp) use ARM intrinsics with no x86 fallback. The official
# Cactus Gemma 4 deployment blog confirms this scope:
#
#   "Cactus targets ARM across platforms: Apple Silicon Macs, iPhones,
#    iPads, Vision Pro, and Android devices with ARM64 chipsets."
#   — https://docs.cactuscompute.com/v1.14/blog/gemma4/
#
# Cactus's documented benchmarks page is also CPU-only — there is no PCIe
# NVIDIA/AMD GPU acceleration path; on-device acceleration uses the
# integrated NPUs on ARM SoCs (Apple Neural Engine, Qualcomm Hexagon,
# MediaTek/Exynos APU).
#
# We retain the build step in source so the GitHub repo shows the full
# deployment pipeline end-to-end, but gate execution on host architecture:
#   - ARM host (Apple Silicon Mac, Pi 5, Android emulator on ARM, ARM cloud):
#     compile libcactus.so, validate the binary, prepare for Cell 5's
#     Python SDK smoke test.
#   - x86 host (Colab CPU/GPU runtimes, x86 servers): skip the build with
#     a clear architectural-rationale message. The convert step in Cell 3
#     still works (it is a pure-Python format converter with no engine
#     dependency); the Cell 5 smoke test also skips on x86.
#
# This means a full end-to-end run requires an ARM host. The convert-only
# path on Colab x86 still produces a deployable artifact for the companion
# Flutter Android app to load on a real ARM device or emulator.

import platform as _platform

_HOST_ARCH = _platform.machine().lower()
_IS_ARM_HOST = _HOST_ARCH in ("aarch64", "arm64", "armv8")

print()
print("=" * 72)
print(f"Host architecture detected: {_HOST_ARCH}  →  "
      f"{'ARM (Cactus build will RUN)' if _IS_ARM_HOST else 'x86 (Cactus build will SKIP gracefully)'}")
print("=" * 72)

if _IS_ARM_HOST:
    print()
    print("Running `cactus build --python` (ARM host detected) ...")
    print("(streaming compiler output below; this is the slow step, ~5-15 min)")
    print()
    _build = _sp.run(
        ["cactus", "build", "--python"],
        cwd=CACTUS_SRC_DIR,
        capture_output=True, text=True, timeout=1800,
    )
    print(f"--- cactus build --python (exit {_build.returncode}) ---")
    print("--- stdout (last 1500 chars) ---")
    print((_build.stdout or "")[-1500:])
    print("--- stderr (last 1500 chars) ---")
    print((_build.stderr or "")[-1500:])
    print("--- end of build output ---")

    if _build.returncode != 0:
        raise SystemExit(
            f"cactus build --python failed on ARM host (exit {_build.returncode}). "
            "ARM build is the supported path; investigate the compiler output above."
        )

    _LIB_SO_PATH = f"{CACTUS_SRC_DIR}/cactus/build/libcactus.so"
    if os.path.isfile(_LIB_SO_PATH):
        print(f"\n✅ libcactus.so built at {_LIB_SO_PATH} "
              f"({os.path.getsize(_LIB_SO_PATH) / 1e6:.1f} MB)")
    else:
        # Fall back to recursive search if the .so landed somewhere else
        _candidates = []
        for _root, _, _files in os.walk(CACTUS_SRC_DIR):
            for _f in _files:
                if _f.startswith("libcactus."):
                    _candidates.append(os.path.join(_root, _f))
        print(f"⚠️  Build returned 0 but expected .so missing. Found:")
        for _c in _candidates:
            print(f"     {_c}  ({os.path.getsize(_c) / 1e6:.1f} MB)")
        if not _candidates:
            raise SystemExit("No libcactus.so produced.")
else:
    # x86 host (e.g. Colab x86 runtime) — gracefully skip the ARM build.
    print()
    print("⏭  ARM build step SKIPPED on x86 host (architectural constraint, not a bug).")
    print()
    print("   What would happen on an ARM host:")
    print("   - `cactus build --python` compiles libcactus.so (~5-15 min)")
    print("   - Cell 5's `from src.cactus import ...` resolves against the .so")
    print("   - Cell 5 runs the 10-prompt smoke test on the converted artifact")
    print("   - Cell 6 emits the 3-outcome quality verdict")
    print()
    print("   Why the build skips on x86:")
    print("   - cactus-compute's CMakeLists hardcodes `-march=armv8.2-a+i8mm`")
    print("     and the SIMD kernels use ARM intrinsics with no x86 fallback")
    print("   - The official Cactus Gemma 4 deployment blog scopes Cactus to")
    print("     ARM platforms: https://docs.cactuscompute.com/v1.14/blog/gemma4/")
    print()
    print("   What still runs on x86:")
    print("   - Cell 3 (`cactus convert`) is a pure-Python format converter,")
    print("     no engine dependency. Produces the deployable INT4 artifact at")
    print("     /content/solarhive-e4b-cactus/ for the companion Flutter app.")
    print()
    print("   How to validate end-to-end:")
    print("   - Run the companion Flutter Android app on an emulator (Android")
    print("     Studio's ARM-via-QEMU emulators) or a physical Android device.")
    print("   - Or re-run this notebook on an ARM host (Apple Silicon Mac,")
    print("     Pi 5, or an ARM cloud VM) — the ARM branch above runs the")
    print("     full pipeline including build + smoke test + quality verdict.")
    _LIB_SO_PATH = None  # sentinel — later cells detect x86 mode via _IS_ARM_HOST

print()
print("=" * 72)
print("⚠️  IF SUBSEQUENT CELLS FAIL with errors like:")
print('       AttributeError: module \'numpy._core._multiarray_umath\' has no')
print("       attribute '_blas_supports_fpe'")
print("    or any *_multiarray_umath / torch / mediapipe AttributeError on first")
print("    import → the Colab kernel has stale C extensions from before Cell 0's")
print("    pip upgrade. Fix: Runtime → Restart session, then re-run from Cell 0.")
print("    Do NOT 'Disconnect and delete runtime' — that wipes /content/ and you")
print("    lose Cell 2's downloaded weights. A plain restart preserves /content/.")
print("=" * 72)

print()
print("System RAM check (cactus convert needs ≥ 16 GB system RAM for E4B INT4):")
try:
    with open("/proc/meminfo") as _mi:
        for _line in _mi:
            if _line.startswith("MemTotal"):
                _kb = int(_line.split()[1])
                _gb = _kb / (1024 * 1024)
                print(f"  MemTotal: {_gb:.1f} GB")
                if _gb < 14:
                    print(f"  WARNING — kernel has only {_gb:.1f} GB RAM. "
                          "E4B INT4 conversion may OOM. Switch to Pro High-RAM.")
                break
except Exception as _meminfo_err:
    print(f"  (skipped — non-Linux env: {_meminfo_err})")

"""## 1: HuggingFace authentication

Resolves an HF token from Colab Secrets first, then Kaggle Secrets, then
the `HF_TOKEN` environment variable — same three-source resolution used
in the companion fine-tune / merge / inference notebooks. A read-only
token is sufficient because this notebook only DOWNLOADS the fine-tuned
safetensors; uploading the converted artifact is a separate step.
"""

import os

HF_TOKEN = None

try:
    from google.colab import userdata as _ud
    HF_TOKEN = _ud.get("HF_TOKEN")
    if HF_TOKEN:
        print("HF token loaded from Colab Secrets.")
except Exception:
    pass

if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF token loaded from Kaggle Secrets.")
    except Exception:
        pass

if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        print("HF token loaded from environment.")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    print("WARNING — no HF token found. If solarhive-e4b-ollama is private at "
          "run time, Cell 2 will fail with 401. Set HF_TOKEN before continuing.")

"""## 2: Resolve the fine-tuned E4B merged safetensors

Two-tier resolution mirroring the companion merge / fine-tune notebooks:
1. Drive cache at `/content/drive/MyDrive/models/solarhive_e4b_merged_*/`
   — fast path for repeat runs that previously saved the safetensors to
   Google Drive.
2. HuggingFace fallback: `snapshot_download` from
   `Truthseeker87/solarhive-e4b-ollama` — the canonical merged-safetensors
   artifact produced by the fine-tune + merge pipeline.

Cactus's `convert` command accepts either an HF model ID or a local path;
this notebook resolves to a local path so the convert subprocess does not
need to re-authenticate against HF for private-repo access.
"""

import glob as _glob
import time as _time
from pathlib import Path as _Path

_DRIVE_PATTERN = "/content/drive/MyDrive/models/solarhive_e4b_merged_*"
_HF_REPO = "Truthseeker87/solarhive-e4b-ollama"
LOCAL_E4B_MERGED = "/content/solarhive-e4b-merged"

try:
    from google.colab import drive as _gdrive
    _gdrive.mount("/content/drive")
except Exception:
    pass

_drive_hits = sorted(_glob.glob(_DRIVE_PATTERN))
if _drive_hits:
    _src = _drive_hits[-1]
    print(f"Drive cache hit — copying from {_src}")
    if _Path(LOCAL_E4B_MERGED).exists():
        _shutil.rmtree(LOCAL_E4B_MERGED)
    _t0 = _time.time()
    _shutil.copytree(_src, LOCAL_E4B_MERGED)
    print(f"  Copied in {_time.time() - _t0:.0f}s")
else:
    print(f"No Drive cache — downloading from HF: {_HF_REPO}")
    from huggingface_hub import snapshot_download as _snap
    _t0 = _time.time()
    LOCAL_E4B_MERGED = _snap(
        repo_id=_HF_REPO,
        local_dir=LOCAL_E4B_MERGED,
        token=HF_TOKEN,
    )
    print(f"  Downloaded in {_time.time() - _t0:.0f}s")

print()
print(f"Source ready at {LOCAL_E4B_MERGED}")
print(f"Files:")
for _f in sorted(os.listdir(LOCAL_E4B_MERGED)):
    _fp = _Path(LOCAL_E4B_MERGED) / _f
    if _fp.is_file():
        _gb = _fp.stat().st_size / 1e9
        print(f"  {_gb:7.3f} GB  {_f}")

"""## 3: Run `cactus convert` with INT4 quantization

The core conversion step. Inputs:
- Source: the locally-resolved merged safetensors from Cell 2 (PEFT-merged
  weights — no `--lora` flag is needed because the LoRA adapters were already
  merged into the base by the upstream merge notebook).
- Quantization: INT4 (default for Cactus E4B mobile deployment per the
  Cactus docs).
- Output directory: `/content/solarhive-e4b-cactus/` — picked up by the
  companion Flutter Android app.

Possible outcomes:
- Conversion succeeds → Cell 4 inspects the artifact.
- Architecture rejection (model not in Cactus's supported list) → fall back
  to base via `cactus run google/gemma-4-E4B-it`.
- OOM during convert → upgrade the runtime to Colab Pro CPU + High-RAM.
- Quant-format error → try INT8 if the Cactus version does not support INT4.
"""

CACTUS_OUT = "/content/solarhive-e4b-cactus"
# Cactus convert CLI per https://github.com/cactus-compute/cactus :
#   cactus convert <model> [dir] --precision INT4|INT8|FP16
#   - <model> is positional (local path or HF id)
#   - [dir] is the positional output directory (NOT a --output flag)
#   - --precision defaults to INT4; specified explicitly here for clarity
CACTUS_CMD = [
    "cactus", "convert",
    LOCAL_E4B_MERGED,         # positional <model>
    CACTUS_OUT,               # positional [dir]
    "--precision", "INT4",    # explicit even though INT4 is the default
]

print(f"Running: {' '.join(CACTUS_CMD)}")
print(f"Expected: ~5-15 min on Colab Pro CPU + High-RAM runtime")
print()

_t0 = _time.time()
_convert = _sp.run(CACTUS_CMD, capture_output=True, text=True)
_elapsed = _time.time() - _t0

print(f"Exit code: {_convert.returncode}  |  Elapsed: {_elapsed:.0f}s")
print()

if _convert.returncode != 0:
    print("CONVERT FAILED")
    print("--- last 30 lines of stdout ---")
    print("\n".join((_convert.stdout or "").splitlines()[-30:]))
    print()
    print("--- last 30 lines of stderr ---")
    _err_tail = "\n".join((_convert.stderr or "").splitlines()[-30:])
    print(_err_tail)
    print()
    _err_lc = (_convert.stderr or "").lower()
    if "not supported" in _err_lc or "unknown model" in _err_lc:
        print("📉 Cactus rejected fine-tuned E4B as an unsupported architecture.")
        print("   Next: try `cactus run google/gemma-4-E4B-it` directly (base model")
        print("   auto-downloaded) and ship base + SolarHive system prompts.")
    elif "memory" in _err_lc or "oom" in _err_lc or "killed" in _err_lc:
        print("📉 OOM during convert. Switch to Colab Pro High-RAM (≥51 GB) and retry.")
    elif "int4" in _err_lc or "quantization" in _err_lc:
        print("📉 INT4 quantization not supported. Try INT8 or check Cactus version.")
    else:
        print("📉 New failure mode — capture stack trace for the upstream issue.")
    raise SystemExit("Aborting — cactus convert failed.")

print("CONVERT SUCCEEDED")
print()
print("--- last 20 lines of stdout ---")
print("\n".join((_convert.stdout or "").splitlines()[-20:]))

"""## 4: Inspect the converted Cactus artifact

Confirms `cactus convert` wrote a valid artifact and lists the per-file
sizes. The Cactus blog cites ~4 GB for an INT4-quantized text-only Gemma 4
E4B; the multimodal Gemma 4 E4B artifact lands at ~7 GB because the audio
Conformer tower and vision encoder are kept in FP16 alongside the
INT4-quantized text weights.
"""

if not _Path(CACTUS_OUT).is_dir():
    print(f"❌ Output directory {CACTUS_OUT} not created.")
    raise SystemExit("Aborting — no output dir.")

print(f"Files in {CACTUS_OUT}:")
_total_gb = 0.0
for _f in sorted(_Path(CACTUS_OUT).rglob("*")):
    if _f.is_file():
        _gb = _f.stat().st_size / 1e9
        _total_gb += _gb
        print(f"  {_gb:7.3f} GB  {_f.relative_to(CACTUS_OUT)}")

print()
print(f"Total artifact size: {_total_gb:.2f} GB")
print("Reference: Cactus blog cites ~4 GB INT4 (text-only).")
print("Multimodal Gemma 4 E4B is expected to be ~7 GB because the audio")
print("Conformer tower and vision encoder remain FP16 alongside the INT4")
print("text weights.")
if _total_gb < 1.5 or _total_gb > 9.0:
    print(f"⚠️  Size off-trend ({_total_gb:.2f} GB) — may indicate truncated convert.")

"""## 5: Smoke test via the Cactus Python SDK

Runs two prompt classes against the converted artifact (5 prompts each,
10 total) using the same prompt structure as the SolarHive inference
parity benchmark:

- Class A — standard Q&A (no emoji format expected). Five domain probes
  that test whether the fine-tuned domain knowledge survived INT4
  quantization.
- Class B — SolarHive emoji format. Five prompts that test whether the
  model can produce the `[1-2 emojis] [imperative <15 words]` output
  format with mode-appropriate emoji vocabulary (suburban vs. rural).
  This is the UX-layer test the companion Flutter app depends on.

Inference uses the Cactus Python SDK (module-level `cactus_init` /
`cactus_complete` / `cactus_destroy` from `src.cactus`), not the
`cactus run` CLI — `cactus run` opens an interactive playground rather
than a one-shot completion endpoint. The cell loads the converted
artifact once, loops the 10 prompts through `cactus_complete()`, then
frees the model handle.

Cactus's Python SDK requires the compiled libcactus.so library, which is
ARM-only. On x86 hosts this cell skips gracefully and Cell 6 emits a
convert-only verdict; on ARM hosts the full smoke test runs.
"""

CLASS_A_PROMPTS = [
    "What happens to solar production when humidity exceeds 80%?",
    "At what battery SOC should we stop exporting to the grid?",
    "Home #3 has been underperforming by 22% for three weeks. What's the diagnostic checklist?",
    "It's winter in Ann Arbor and panels have snow. Prioritize actions.",
    "Grid frequency dropped to 59.8 Hz. What does that mean for our microgrid?",
]

# Class B uses the SolarHive emoji-format system prompt — strict format
# `[1-2 emojis] [imperative <15 words]` with mode-specific vocabulary.
EMOJI_SYSTEM_PROMPT_SUBURBAN = (
    "You are SolarHive, an AI energy advisor for community solar households "
    "in SUBURBAN mode (rooftop-fixed panels). Reply in this exact format:\n"
    "[1-2 emojis] [one short imperative sentence under 15 words]\n"
    "Use these emojis: ☀️ 🌤️ ⛅ ☁️ 🌧️ 🔋 🪫 ⚡ 🟢 🧹 🍂 ❄️ ⚠️ 🔧 📉 🛰️ 📡 🔬"
)
EMOJI_SYSTEM_PROMPT_RURAL = (
    "You are SolarHive, an AI energy advisor for community solar households "
    "in RURAL_OFFGRID mode (deployable panels). Reply in this exact format:\n"
    "[1-2 emojis] [one short imperative sentence under 15 words]\n"
    "Use these emojis: 🌅 ☀️ 🌧️ 💨 ⛈️ 📤 📥 🪛 🧹 💧 ⏰ 🔋 🪫 🛰️ 📡"
)

CLASS_B_PROMPTS = [
    (EMOJI_SYSTEM_PROMPT_SUBURBAN, "How is solar today?"),
    (EMOJI_SYSTEM_PROMPT_SUBURBAN, "Should I run my dishwasher now?"),
    (EMOJI_SYSTEM_PROMPT_SUBURBAN, "Why is output low this week?"),
    (EMOJI_SYSTEM_PROMPT_RURAL, "Storm warning incoming?"),
    (EMOJI_SYSTEM_PROMPT_RURAL, "Should I deploy panels right now?"),
]


import json

# Host-arch-gated execution: Cactus's Python SDK FFIs into libcactus.so,
# which is only compiled on ARM hosts (see the host-arch gate at the end
# of Cell 0). On x86 hosts the smoke test skips gracefully and Cell 6
# falls back to a convert-only verdict; on ARM hosts the full 10-prompt
# validation runs end-to-end.

if _IS_ARM_HOST:
    # The Cactus Python SDK is exposed as module-level functions in the
    # `src.cactus` namespace (sys.path was patched in Cell 0 to make this
    # importable without a kernel restart).
    from src.cactus import cactus_init, cactus_complete, cactus_destroy

    # Load the converted model ONCE (the load is the expensive part).
    print(f"Loading converted model from {CACTUS_OUT} via cactus_init() ...")
    _load_t0 = _time.time()
    _model_handle = cactus_init(str(CACTUS_OUT), None, False)
    _load_elapsed = _time.time() - _load_t0
    print(f"  ✅ Model loaded in {_load_elapsed:.0f}s — handle = {_model_handle}")
    print()


    def _cactus_complete_prompt(system_prompt, user_prompt,
                                 max_tokens=256, temperature=0.7):
        """Call cactus_complete() via the Python SDK. Returns
        (response_text, error_or_none, elapsed_seconds)."""
        if system_prompt:
            _messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            _messages = [{"role": "user", "content": user_prompt}]
        _options = json.dumps({"max_tokens": max_tokens, "temperature": temperature})
        _t0 = _time.time()
        try:
            _result_json = cactus_complete(
                _model_handle,
                json.dumps(_messages),
                _options,
                None,   # tools_json — none for smoke test
                None,   # streaming callback — blocking call
            )
            _result = json.loads(_result_json)
            return _result.get("response", ""), None, _time.time() - _t0
        except Exception as _e:
            return "", str(_e), _time.time() - _t0


    print("=" * 72)
    print("CLASS A — Standard Q&A (no emoji format expected)")
    print("=" * 72)
    results_a = []
    for _i, _q in enumerate(CLASS_A_PROMPTS, 1):
        print(f"\nQ{_i}: {_q}")
        _resp, _err, _elapsed = _cactus_complete_prompt(None, _q)
        if _err:
            print(f"  ❌ cactus_complete failed: {_err[:200]}")
            results_a.append({"prompt": _q, "response": "", "error": _err})
            continue
        print(f"A{_i} ({_elapsed:.1f}s): {_resp[:280]}{'...' if len(_resp) > 280 else ''}")
        results_a.append({"prompt": _q, "response": _resp, "elapsed": _elapsed})

    print()
    print("=" * 72)
    print("CLASS B — SolarHive emoji format (suburban + rural)")
    print("=" * 72)
    results_b = []
    for _i, (_sys_text, _q) in enumerate(CLASS_B_PROMPTS, 1):
        _mode = "🏘️" if "SUBURBAN" in _sys_text else "🌾"
        print(f"\nQ{_i} ({_mode}): {_q}")
        _resp, _err, _elapsed = _cactus_complete_prompt(_sys_text, _q)
        if _err:
            print(f"  ❌ cactus_complete failed: {_err[:200]}")
            results_b.append({"prompt": _q, "mode": _mode, "response": "", "error": _err})
            continue
        print(f"A{_i} ({_elapsed:.1f}s): {_resp[:280]}{'...' if len(_resp) > 280 else ''}")
        results_b.append({"prompt": _q, "mode": _mode, "response": _resp, "elapsed": _elapsed})

    # Free the model handle
    cactus_destroy(_model_handle)
    print("\n✅ Model handle destroyed; smoke test complete.")
else:
    # x86 host — gracefully skip the smoke test; Cell 6 adapts its verdict.
    print("=" * 72)
    print("⏭  Python SDK smoke test SKIPPED on x86 host")
    print("=" * 72)
    print()
    print("`from src.cactus import cactus_init` requires libcactus.so, which")
    print("Cell 0's `cactus build --python` step builds. The build step is")
    print("ARM-only (per Cactus's documented deployment surface), so it")
    print("skipped on this x86 host and libcactus.so is not available.")
    print()
    print("The smoke-test code above runs end-to-end on any ARM host:")
    print("  - Android emulator (e.g. Android Studio Pixel emulators on ARM)")
    print("  - Apple Silicon Mac")
    print("  - Pi 5 / ARM cloud VM (e.g. Ampere A1)")
    print()
    print("Provide stub results so Cell 6's verdict logic remains well-defined:")
    results_a = []
    results_b = []

"""## 6: Verdict (decision gate)

On ARM hosts, applies a three-outcome gate based on Cell 5's smoke-test
counts:

- VERDICT 1 — ≥4/5 Class A coherent AND ≥3/5 Class B emoji-format. The
  fine-tune survived INT4 quantization with both domain knowledge and UX
  format intact. Proceed to the companion Flutter Android app.
- VERDICT 2 — Class A ≥4/5 but Class B <3/5. Domain knowledge intact but
  emoji format weakened. Iterate the system prompt (or relax the parser)
  and re-validate.
- VERDICT 3 — Class A <4/5. The INT4 quantization eroded too much
  fine-tune knowledge. Fall back to base E4B (`cactus run
  google/gemma-4-E4B-it`) plus SolarHive system prompts at runtime.

On x86 hosts (where Cell 5's smoke test skipped), the verdict reduces to
"convert pass / fail" based on Cell 4's artifact inspection. Inference
quality is validated downstream in the companion Flutter Android app.
"""

PAD_SIGNALS = ("<pad>", "<unk>", "")


def _is_coherent(_text: str, min_chars: int = 10) -> bool:
    """Permissive heuristic — coherent if length >= min_chars and not pad-token spam."""
    _t = _text.strip()
    if len(_t) < min_chars:
        return False
    _first5 = _t.split()[:5]
    if all(_w.strip() in PAD_SIGNALS for _w in _first5):
        return False
    if not any(_w.isalpha() or any(_c.isalpha() for _c in _w) for _w in _first5):
        return False
    return True


def _has_leading_emoji(_text: str) -> bool:
    """True if the response starts with at least one emoji character."""
    _t = _text.strip()
    if not _t:
        return False
    # Permissive — any of the documented SolarHive emojis at the start
    _emojis = "☀️🌤️⛅☁️🌧️💨⛈️🔋🪫⚡🟢🧹🍂❄️⚠️🔧📉🛰️📡🔬🌅📤📥🪛💧⏰🏘️🌾"
    _first_chars = _t[:6]  # First few chars (multi-byte emojis can be 1-4 bytes each)
    return any(_e in _first_chars for _e in _emojis)


if not _IS_ARM_HOST:
    # x86 host — Cell 5's smoke test skipped, so the verdict is reduced
    # to convert-only based on Cell 4's artifact inspection. Inference
    # quality validation runs downstream in the companion Flutter app.
    print("=" * 72)
    print("VERDICT (x86 development host) — CONVERT-ONLY ✅")
    print("=" * 72)
    print()
    print(f"Convert artifact: {CACTUS_OUT} ({_total_gb:.2f} GB)")
    print("Reference: Cactus blog cites ~4 GB INT4 (text-only); the multimodal")
    print("E4B artifact retains FP16 audio + vision towers, expected ~7 GB.")
    print()
    print("Convert-pipeline deliverables on x86:")
    print("  ✅ Cactus install + clone + editable install + sys.path patch")
    print("  ✅ Cactus CLI verified (--help recognizes 'convert' subcommand)")
    print("  ⏭  cactus build --python skipped (ARM-only — see Cell 0)")
    print("  ✅ Fine-tuned E4B safetensors resolved from local cache or HF")
    print("  ✅ Cactus convert succeeded — produces deployable INT4 artifact")
    print("     (~5-15 min on Colab Pro CPU + High-RAM)")
    print("  ⏭  Python SDK smoke test skipped (ARM-only — see Cell 5)")
    print()
    print("Next actions:")
    print("  1. Save the converted artifact (push to a new HF repo or Drive).")
    print("  2. Update companion documentation to reference the new artifact.")
    print("  3. Build the companion Flutter Android app and load the artifact")
    print("     via the Cactus Flutter SDK on an emulator or physical device —")
    print("     this is where the Class A / Class B quality gate runs.")
    print("  4. Optional: re-run THIS notebook on an ARM host (Apple Silicon")
    print("     Mac, Pi 5, ARM cloud) — the ARM branches in Cells 0/5/6 will")
    print("     activate and run the full pipeline end-to-end.")
else:
    # ARM host — apply the three-outcome verdict using Cell 5's smoke counts.
    _a_coherent = sum(_is_coherent(_r["response"]) for _r in results_a)
    _b_emoji_format = sum(
        _is_coherent(_r["response"], min_chars=5) and _has_leading_emoji(_r["response"])
        for _r in results_b
    )

    print("=" * 72)
    print(f"CLASS A QUALITY    : {_a_coherent}/5 coherent (target: ≥4/5)")
    print(f"CLASS B EMOJI FORMAT: {_b_emoji_format}/5 well-formed (target: ≥3/5)")
    print(f"ARTIFACT SIZE      : {_total_gb:.2f} GB (multimodal Cactus INT4)")
    print("=" * 72)
    print()

    if _a_coherent >= 4 and _b_emoji_format >= 3:
        print("✅  VERDICT 1 — Cactus convert + smoke test PASS")
        print()
        print("Next actions:")
        print("  1. Save this notebook output for the project audit log.")
        print("  2. Push the converted artifact to a public HF repo for")
        print("     Flutter app distribution.")
        print("  3. Build the companion Flutter Android app and load the")
        print("     artifact via the Cactus Flutter SDK.")
    elif _a_coherent >= 4:
        print(f"⚠️   VERDICT 2 — Convert works but emoji format weak ({_b_emoji_format}/5)")
        print()
        print("Next actions:")
        print("  1. Save the Cell 5 transcript and note which prompts failed format.")
        print("  2. Iterate the emoji system prompt (more few-shot examples) and re-run.")
        print("  3. If format adherence climbs to ≥3/5 with prompt changes → VERDICT 1.")
        print("  4. Otherwise relax the Dart-side parser to tolerate non-leading")
        print("     emojis (extract the emoji from anywhere in the response).")
        print("  5. Document any residual format regression honestly.")
    else:
        print(f"❌  VERDICT 3 — Fine-tune lost too much in INT4 ({_a_coherent}/5 Class A coherent)")
        print()
        print("Next actions:")
        print("  1. Save this Cell 5 transcript as evidence of the INT4 regression.")
        print("  2. Try `cactus run google/gemma-4-E4B-it` (Cactus auto-downloads")
        print("     INT4 of the base) and re-run Cell 5 with the same prompts.")
        print("  3. If base E4B passes Class A → ship the companion app with")
        print("     base E4B + SolarHive system prompts at runtime.")
        print("  4. If even base E4B fails on Class A → drop the Cactus track entirely.")
