# SolarHive — Data Principles

Methodology for building the fine-tuning dataset. Every example traces back to a
real API response or a verified physical model — no synthetic or hallucinated numbers.

## 1. Data Gathering

- **Live API grounding.** Every numeric claim originates from a real API call:
  Open-Meteo (GHI irradiance), OpenWeatherMap (temperature, wind, cloud cover),
  EIA Open Data (regional grid mix, CO₂ intensity), PVWatts (cross-validation).
- **Temporal sampling.** Training examples span real hourly data: ~1,180
  EIA mix snapshots (30-day rolling window per location), ~52,560 Open-Meteo
  meteorological rows per location (hourly × ~3 years).
- **Geographic diversity.** Two distinct climate zones — MISO/Midwest
  (Ann Arbor MI, Detroit MI) and CAISO/Pacific (San Mateo CA, Fresno CA) —
  with per-location metadata: timezone, panel capacity, tilt angle, grid region.
- **No synthetic values.** All temperatures, irradiance, grid percentages, and
  CO₂ numbers are derived directly from API responses, not invented.

## 2. Inspection

- **14 diagnostic charts** generated per datagen run: GHI distribution,
  hourly production curve, month×hour heatmap, temperature derating,
  feature correlation, cloud cover by season, seasonal production, GHI vs
  production scatter, PVWatts cross-validation, OWM snapshot, fuel mix,
  renewable% + CO2 time-series, irradiance triple (GHI/DNI/DHI), and
  vertical cloud-cover stack.
- **Anomaly detection rules:**
  - Nighttime GHI > 50 W/m² → flagged
  - Renewable percentage > 100% (midday solar overproduction) → clamped to [0, 100]
  - Negative CO₂ intensity → clamped to [0, ∞)
- **Format validation:** every example checked for 3-tuple structure,
  non-empty strings, and token length within MAX_SEQ_LEN.

## 3. Analysis

- **Question diversity:** unique questions / total count (target > 70%).
- **Answer skeleton diversity:** unique structural templates / total (target > 30%).
- **Category balance:** harder tasks (tool-calling, multi-step reasoning) get
  proportionally more examples — not equal-sized buckets.
- **Cross-validation:** Open-Meteo GHI vs PVWatts monthly production estimates
  to catch systematic bias in either source.
- **Temporal pattern verification:** production peaks align with local solar noon
  (not shifted by timezone bugs).

## 4. Data Preparation

- **Two training formats:**
  - Q&A 3-tuples `(system_prompt, question, answer)` for domain knowledge
  - Tool-calling message lists `[system, user, assistant+tool_calls, tool, assistant]`
    for function-calling behavior
- **Question-responsive answers:** answer emphasis matches question intent
  (e.g., "how much power?" → lead with kW number, not weather summary).
- **Geographic task filtering:** climate-specific examples tagged to region
  (snow-guard maintenance → MISO only, UV degradation → CAISO only).
- **Exact dedup** via hash on `(question, answer)` pair.
- **Token length validation:** flag examples exceeding MAX_SEQ_LEN.
- **Cross-file dedup policy:** accepted semantic overlap (~15–25%) between
  datagen and static examples at current scale.

## 5. Fine-Tuning Objectives

- **Domain knowledge transfer:** weather → production impact, battery strategy,
  grid economics, maintenance scheduling.
- **Tool routing:** teach WHEN to call tools vs. answer from knowledge.
  - "right now" / "current" → call a tool for live data
  - "typical" / "generally" → answer from internalized knowledge
- **Tool format:** emit `call:fn_name{args}` + synthesize multi-step results
  into coherent natural-language answers.
- **Multi-step reasoning:** chained tool calls (weather + solar + battery),
  weekly energy audits, seasonal planning.
- **Dual target:** 26B A4B (cloud demo with full capabilities) +
  E4B → E2B GGUF (Ollama edge deployment for offline resilience).

## 6. Data Volume & Diversity Tenets

### 6.1 Target Volume

Research literature on parameter-efficient fine-tuning (LoRA/QLoRA) of
instruction-tuned LLMs converges on **500–2,000 examples** as the sweet spot
for domain-specific adaptation:

- Below ~300 examples, QLoRA struggles to reliably shift model behavior away
  from generic base-model responses toward domain-specific patterns.
- Between 500–2,000, the model internalizes domain vocabulary, reasoning
  chains, and output style without catastrophic forgetting of base capabilities.
- Above ~5,000 with QLoRA (low-rank), returns diminish sharply — the adapter
  capacity saturates before the data does. Full fine-tuning benefits from
  larger sets, but QLoRA's rank-16 bottleneck limits absorptive capacity.

SolarHive targets **1,727 total examples** (1,530 unique Q&A + 183 tool-calling
+ 14 image-grounded), placing it squarely in the productive range for LoRA r=16
on instruction-tuned Gemma 4.

### 6.2 Combined Training Data Stack

Three sources feed the training set. Overlap between sources is intentional —
the model sees the same concepts expressed in different authoring styles,
grounded in different data snapshots.

**Q&A Domain Knowledge (1,530 unique post-dedup; 1,535 generated):**

| Skill Domain | Hand-crafted | Cell 7 algorithmic | Combined |
|---|---|---|---|
| Sky / Weather → Production | 51 | 793 (A+D) | **844** |
| Battery & Grid Strategy | 60 (bat+alt) | 80 (E) | **140** |
| Grid Mix / Carbon / TOU | 52 | 80 (C) | **132** |
| PVWatts / Benchmarking | — | 112 (B) | **112** |
| Multi-Step Reasoning | 52 | 57 (F) | **109** |
| Emergency & Resilience | 51 | — | **51** |
| Seasonal & Forecast | 50 | — | **50** |
| Consumption Optimization | 49 | — | **49** |
| Panel Health / Maintenance | 48 | — | **48** |
| **Q&A Total** | **413** | **1,122** | **1,535 (1,530 unique post-dedup)** |

**Tool-Calling Behavior (183 examples)** — distribution follows Ross et al.
(2025) [*When2Call: When (not) to Call Tools*](https://arxiv.org/abs/2504.18851)
to address the 9–67% tool-hallucination rates that public datasets exhibit
because they lack follow-up and unable-to-answer cases:

| Category | Count |
|---|---|
| (b) *should-call* (single + multi-tool chains + full 4-tool community audits) | **106** |
| (a) *should-not-call* (general-knowledge — teaches when NOT to invoke a tool) | **53** |
| (d) *unable-to-answer* (questions outside available tools — name the limit + redirect) | **10** |
| (c) *follow-up clarification* (insufficient input — model asks for the missing detail) | **6** |
| Failure-recovery sequences (graceful handling when a tool returns an error) | **8** |
| **Tool-Calling Total** | **183** |

**Image-Grounded Q&A (14 turns from 7 photos):**

| Source | Photos | Q&A turns | Labels |
|---|---|---|---|
| Project archive (Ann Arbor sky photographs) | 7 | 14 | Cloud type (clear / partly_cloudy / cloudy / overcast) + cloud_pct (0–100) — manually confirmed |

Numeric production claims in image-grounded answers trace to the cloud-cover
label via the same temperature-derated GHI formula used in text rows.

### 6.3 Sufficiency Criteria

We do NOT aim for equal counts per category. Instead, volume scales with
learning difficulty:

- **High-volume (100+):** Skills requiring data-grounded numeric reasoning
  (production estimation, grid economics, battery strategy). The model must
  learn to produce specific numbers that vary with conditions — this requires
  more examples to generalize the pattern.
- **Medium-volume (48–64):** Skills where the base model already has
  foundational knowledge and we are adding domain-specific framing (panel
  health, consumption optimization, seasonal forecasts, emergency response).
  50 well-crafted examples are sufficient to steer an instruction-tuned model.
- **Tool-calling (183 total):** Gemma 4 has native function-calling
  capability. We are reinforcing — not teaching from scratch — the
  `call:fn_name{args}` format, routing logic (real-time → tool, general →
  knowledge), and multi-step synthesis. The When2Call-style mix of
  *should-call* (106), *should-not-call* (53), *unable-to-answer* (10),
  *follow-up clarification* (6), and failure-recovery (8) examples covers
  the four distinct decision boundaries the model must internalize.

### 6.4 Diversity Over Volume

When a category has sufficient volume, we prioritize diversity over additional
count:

- **Question diversity:** the 1,122 generated examples are deduplicated
  before assembly (5 dupes removed → 1,117 unique algorithmic Q&A). When
  the same template runs against different `(location, hourly timestamp)`
  draws, the resulting answers carry different data-grounded numbers,
  teaching the model to vary responses based on context.
- **Multi-source authoring:** Categories covered by both finetune static
  (hand-written, detailed) and datagen (template-driven, API-grounded) benefit
  from stylistic diversity. The model sees the same concept expressed two ways.
- **Geographic variation:** Examples span MISO/Midwest and CAISO/Pacific
  climate zones. The model learns region-specific patterns (snow → MISO,
  UV degradation → CAISO) rather than memorizing one location.
- **Temporal variation:** Datagen samples across ~52,560 hourly data points
  per location (Open-Meteo Archive endpoint, ~3-year window per location at
  hourly resolution). Training answers cover dawn, noon, dusk, night; summer
  and winter; clear and cloudy conditions.

### 6.5 When to Add More Data

More examples are warranted only when post-training benchmarks reveal a
specific gap — not speculatively:

1. Run fine-tune with current dataset (1,727 examples).
2. Evaluate with held-out benchmark questions (Cell 6/6b in finetune notebook).
3. If a category scores below baseline on >30% of its benchmark questions,
   add 20–30 targeted examples for that category only.
4. Re-train and re-evaluate. One surgical iteration, not broad expansion.

This avoids the common pitfall of over-generating data that shifts category
balance and introduces regressions in categories that were already performing
well.

## 7. Fine-Tuning Results

Dual fine-tune on Google Colab G4 VM — NVIDIA RTX PRO 6000 Blackwell
Server Edition (96 GB GDDR7, ~95 GB usable), Unsloth 2026.4.4,
transformers 5.5.0. Both models are LoRA-fine-tuned on the canonical
1,727-row corpus for 3 epochs (BF16).

**Fine-tuning is text-only on the multimodal-capable corpus.** Image
rows in the dataset are skipped at the data-prep layer (per the Apr 30
Option C revert in `solarhive_finetune.py`). VQA at inference time uses
the base Gemma 4 model's pretrained vision encoder — ~150M parameters
for E4B and ~550M for 26B A4B per the
[official model card](https://ai.google.dev/gemma/docs/core/model_card_4).
Our LoRA targets only the language-model linear layers
(`target=all-linear`); the vision tower is not modified. This
mirrors the Vertex AI Gemma 4 SFT recipe documented in the
[Hugging Face blog](https://huggingface.co/blog/gemma4), which
explicitly freezes both vision and audio towers during text-focused
fine-tuning.

### 7.1 Shared Hyperparameters

Both models use identical hyperparameters — only batch size differs (VRAM-auto-tuned):

| Hyperparameter | Value |
|---------------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target modules | all-linear |
| Learning rate | 2e-4 |
| Optimizer | adamw_8bit |
| Epochs | 3 |
| Warmup steps | 5 |
| Max sequence length | 2048 |
| Precision | BF16 |
| Seed | 3407 |

### 7.2 VRAM-Auto-Tuned Batch Size

Batch size is auto-tuned at runtime based on free GPU VRAM after model loading.
Gradient accumulation compensates for smaller per-device batches, preserving
learning signal quality (e.g., batch=1 × accum=8 ≈ batch=8 × accum=1, just slower).

| Free VRAM | E4B (per_device × accum = effective) | 26B A4B (per_device × accum = effective) |
|-----------|--------------------------------------|------------------------------------------|
| ≥ 40 GB | 4 × 4 = **16** | 2 × 8 = **16** |
| ≥ 20 GB | 2 × 8 = **16** | 1 × 8 = **8** |
| < 20 GB | 1 × 8 = **8** | 1 × 4 = **4** |

**Result:** ~33 GB free after loading each model → E4B: 4×4=16, 26B A4B: 1×8=8.
The smaller effective batch for 26B A4B means more gradient updates per epoch (387 vs 195 steps),
giving it finer-grained learning — likely contributing to its lower converged loss.

### 7.3 Training Metrics

| Model | Converged Loss (last 20 avg) | Final Step | Min | HF Avg (all steps) | Steps | Batch | Time | Trainable |
|-------|-------------------------------|------------|------|-------------------|-------|-------|------|-----------|
| E4B (8B) | **1.059** | 1.182 | 0.469 | 1.933 | 195 | 4x4=16 | 257s | 41.2M / 8.0B (0.51%) |
| 26B A4B (MoE) | **0.742** | 0.776 | 0.364 | 1.053 | 387 | 1x8=8 | 4266s | 29.6M / 25.8B (0.11%) |

### 7.4 Benchmark Results

**Q&A (5 held-out domain questions — expect direct answers, no tool calls):**

| Question | E4B | 26B A4B |
|----------|-----|---------|
| Humidity >80% impact on production | Direct answer | Direct answer |
| Battery SOC export threshold | Direct answer | Direct answer |
| Home underperforming 22% — diagnostic checklist | Direct answer | Direct answer |
| Winter snow on panels — prioritize actions | Direct answer | Direct answer |
| Grid frequency 59.8 Hz — microgrid impact | Direct answer | Direct answer |
| **Score** | **5/5** | **5/5** |

**Tool-calling (3 held-out questions — expect correct routing):**

| Question | Expected | E4B | 26B A4B |
|----------|----------|-----|---------|
| Current battery state? | `call:get_battery_state` | Correct | Correct |
| Solar production in Seattle? | `call:get_weather` | Correct | Correct |
| General maintenance tips? | No tool call | Correct | Correct |
| **Score** | — | **3/3** | **3/3** |

### 7.5 Key Findings

1. **System prompt unification was critical.** Earlier runs with separate
   prompts for Q&A (generic) vs tool-calling (rich) taught the model to use
   the prompt itself as a routing signal — resulting in 0/5 Q&A benchmark
   (all attempted tool calls). Unifying to one prompt fixed this completely.
2. **26B A4B (MoE) absorbs domain knowledge more effectively** than E4B,
   producing richer answers with specific numbers and structured reasoning
   at lower loss (1.04 vs 1.94).
3. **The Q&A : tool-calling ratio teaches correct routing.** ~89% direct
   answers (1,530 unique Q&A) + ~11% tool examples (183) — with 53 explicit
   *should-not-call* + 10 *unable-to-answer* + 6 *follow-up clarification*
   cases — is sufficient for the model to learn the decision boundary.
4. **GGUF export for E4B requires manual conversion.** Unsloth's llama.cpp
   build fails on the Gemma 4 vision projector (mmproj). Fallback: merged
   16-bit safetensors model, importable directly by Ollama.

### 7.6 Training Architecture Observations

**Attention mask warning (Cell 6 — harmless).** Gemma 4 uses the same token
for padding and end-of-sequence (EOS). When generating a single sequence,
transformers warns that it cannot distinguish padding from EOS without an
explicit attention mask. This is irrelevant in practice: padding exists only
when batching multiple variable-length inputs together. Both fine-tuning
(SFTTrainer handles masking internally) and inference (single user query
at a time) process sequences without ambiguous padding. The warning does
not affect output quality.

**MoE expert LoRA targeting (Cell 9 — Unsloth-specific).** The 26B A4B
uses Mixture of Experts (128 experts, 8 active per token). The standard
LoRA library (PEFT) fails to locate MoE expert layers through its regex-based
parameter matching — reporting "no matching parameters." Unsloth bypasses
PEFT's targeting and directly attaches LoRA adapters to `mlp.experts.gate_up_proj`
(expert routing) and `mlp.experts.down_proj` (expert output). This is confirmed
by 29.6M trainable parameters (0.11%) and monotonically decreasing training
loss. Without this Unsloth-specific MoE handling, LoRA fine-tuning of the
26B A4B would not be possible through standard PEFT alone.
