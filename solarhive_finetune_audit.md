# SolarHive Finetune Notebook — Audit & Changes Record

**Date:** April 25, 2026
**Notebook:** `solarhive_finetune.py` / `.ipynb` (E4B + 26B A4B dual fine-tune)
**Sibling audit:** `litert_e2b_audit.md` (E2B LiteRT pre-run audit, also Apr 25)
**Status:** Forward-only changes applied. Existing v1 weights on Hugging Face unchanged. Notebook is now doc-aligned and prepped for the planned Option A multimodal re-run when Colab Pro compute units refresh.

---

## Context

The `solarhive_finetune.py` notebook produced the published v1 artifacts:
- `Truthseeker87/solarhive-26b-a4b-lora` (Run 2 LoRA, Apr 15; cited in 8/8 Run 6 benchmark)
- `Truthseeker87/solarhive-26b-a4b-nf4` (pre-quantized cloud variant)
- `Truthseeker87/solarhive-e4b-ollama` (E4B safetensors, source for GGUF)
- `Truthseeker87/solarhive-e4b-gguf` (Q4_K_M variants + mmproj, 10/10 Sol B benchmark)

The user audited the notebook against Unsloth's Gemma 4 documentation. The audit identified four classes of finding (Row 1–4). Compute units are exhausted, so retraining is not feasible immediately. The decision tree below filters which findings can be applied as forward-only edits (safe, no impact on v1 weights) versus which need to wait for the next training cycle (Option A multimodal).

---

## Sources

All findings cite Unsloth's Gemma 4 documentation as fetched April 25, 2026:

- https://unsloth.ai/docs/models/gemma-4 — variant overview, hardware, GGUF
- https://unsloth.ai/docs/models/gemma-4/train — fine-tuning code reference
- https://unsloth.ai/docs/models/gemma-4/train#bug-fixes--tips — gotchas, model-specific quirks

Numbered tips below correspond to entries in the *Bug Fixes & Tips* section.

---

## Findings — applied vs deferred

| # | Finding | Doc citation | Applied to E4B? | Applied to A4B? | Why |
|---|---|---|---|---|---|
| 1 | `chat_template="gemma-4-thinking"` is for 26B/31B; smaller models use `gemma-4` | Tip #1 | **Yes — switched to `gemma-4`** | **No — `gemma-4-thinking` is doc-correct for 26B** | E4B is the smaller model; `gemma-4-thinking` was off-doc. 26B A4B is reasoning-class and benefits from the thinking slot. |
| 2 | `finetune_vision_layers=True` on text-only data wastes trainable params | Tip #5 | **Deferred** | **Deferred** | The next planned run is Option A (combined text + sky-image data) — `True` becomes correct once images are in the dataset. Flipping to `False` now and back to `True` later is churn. |
| 3 | Loader call should pass `dtype`, `max_seq_length`, `full_finetuning` explicitly | https://unsloth.ai/docs/models/gemma-4/train#quickstart | **Yes** | **Yes** | Forward-only plumbing — equivalent to defaults that were already applied; no effect on v1 weights. |
| 4 | SFTConfig should pass `weight_decay`, `lr_scheduler_type`, `max_grad_norm` explicitly | Same | **Yes (text-mode values)** | **Yes (text-mode values)** | Forward-only plumbing — `lr_scheduler_type="linear"` and `max_grad_norm=1.0` were already the implicit defaults; only `weight_decay` is a behavioral change for the next run. Multimodal-mode values (`cosine`, `0.3`) come in a separate edit pass when Option A's vision data collator is wired. |

---

## Code changes applied

### Cell 2 — E4B loader + chat template

`solarhive_finetune.py:230` and `solarhive_finetune.py:239`.

**Before:**
```python
model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    load_in_4bit=_use_4bit,
    use_gradient_checkpointing="unsloth",
)
processor = get_chat_template(processor, chat_template="gemma-4-thinking")
```

**After:**
```python
model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    load_in_4bit=_use_4bit,
    use_gradient_checkpointing="unsloth",
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,                            # auto-detect (BF16 on supporting GPUs)
    full_finetuning=False,                 # explicit LoRA mode
)
# Per Unsloth Tip #1: use "gemma-4-thinking" for 26B/31B reasoning-class
# variants and "gemma-4" for E2B/E4B. Simpler template is more robust across
# downstream Ollama/llama.cpp runtimes that don't expose enable_thinking=False.
processor = get_chat_template(processor, chat_template="gemma-4")
```

### Cell 5 — E4B SFTConfig

`solarhive_finetune.py:2731` (post-edit line numbers shift).

**Added:**
```python
weight_decay=0.001,                # Unsloth Gemma 4 train docs default
lr_scheduler_type="linear",        # text-mode default; switch to "cosine" for Option A multimodal
max_grad_norm=1.0,                 # text-mode default; switch to 0.3 for Option A multimodal
```

Other SFTConfig args unchanged (per_device_train_batch_size, gradient_accumulation_steps, warmup_steps, num_train_epochs, learning_rate, fp16/bf16, logging_steps, output_dir, optim, seed, report_to, dataset_text_field, max_seq_length).

### Cell 8 — A4B loader + chat template

`solarhive_finetune.py:3038` and `solarhive_finetune.py:3046`.

Loader gets the same `max_seq_length / dtype / full_finetuning` additions as E4B. Chat template stays at `gemma-4-thinking` per Tip #1 (reasoning-class variant).

```python
a4b_model, a4b_processor = FastVisionModel.from_pretrained(
    model_name=A4B_MODEL_PATH,
    load_in_4bit=_a4b_use_4bit,
    use_gradient_checkpointing="unsloth",
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    full_finetuning=False,
)
# 26B A4B is a reasoning-class variant — Unsloth Tip #1 recommends
# "gemma-4-thinking" for 26B/31B. The thinking template gives the model
# an explicit reasoning slot. Inference passes enable_thinking=False to
# suppress the slot at generation time when not needed.
a4b_processor = get_chat_template(a4b_processor, chat_template="gemma-4-thinking")
```

### Cell 10 — A4B SFTConfig

Same three args added as the E4B Cell 5 (weight_decay=0.001, lr_scheduler_type="linear", max_grad_norm=1.0).

---

## Repo versioning split — v1 / v2

Decided alongside the audit, captured in `project_hf_repo_versioning.md` (auto-memory).

**v1 — text-only, frozen, do NOT overwrite:**
- `Truthseeker87/solarhive-26b-a4b-lora` (Run 2 LoRA + Run 6 8/8 benchmark provenance)
- `Truthseeker87/solarhive-26b-a4b-nf4`
- `Truthseeker87/solarhive-e4b-ollama`
- `Truthseeker87/solarhive-e4b-gguf`
- `Truthseeker87/solarhive-community-solar-1k`

**v2 — Option A multimodal (notebook now points here):**
- `Truthseeker87/solarhive-26b-a4b-multimodal-lora` ← `HF_REPO_A4B`
- `Truthseeker87/solarhive-e4b-multimodal-ollama` ← `HF_REPO`
- `Truthseeker87/solarhive-e4b-multimodal-gguf` (future quantize-notebook target)
- `Truthseeker87/solarhive-community-solar-multimodal` (future datagen target)

**Why new repos rather than overwriting:**
- HF `upload_folder` overwrites same-named files at HEAD (creates new commit). Page view shows whatever's at HEAD.
- v1 model cards cite specific benchmarks (8/8 Run 6 cloud, 10/10 Sol B Ollama) tied to specific weights at v1 URLs. Pushing v2 weights to v1 URLs would invalidate those narratives without rewriting the cards.
- Pushing v2 to new repos: both versions visible side-by-side, judges can verify v1 reproducibility independently, v2 cards cite NEW (text+image) benchmarks without rewriting v1 history.
- Cross-link via Companion Repositories tables in each model card — judges browsing either side discover the other.

**Stale-variable resolution:** the prior `HF_REPO_A4B = "Truthseeker87/solarhive-26b-a4b"` (no `-lora` suffix) didn't match the canonical v1 `solarhive-26b-a4b-lora` either — likely a manual rename post-publish. The variable now points unambiguously at the v2 multimodal target, ending that ambiguity.

---

## Why apply now even though we can't retrain

| Change | Affects v1 weights? | Affects v1 model cards? | Safe to apply now? |
|---|---|---|---|
| Row 1 (E4B chat template) | No — only affects future training runs | No — v1 weights produced under old template stay accurate to old card descriptions | Yes |
| Row 3 (loader plumbing) | No — equivalent to defaults that already applied | No | Yes |
| Row 4 (SFTConfig) | No — only `weight_decay` shifts; other two were defaults already | No | Yes |
| v2 repo rename | No — pushes go to new repos that don't exist yet | No — v1 cards still describe v1 weights at v1 URLs | Yes |

All four changes are forward-only — they affect the *next* training run, not the existing weights. The `solarhive_finetune.py` notebook now describes a future Option A run, not the runs that produced v1. v1 model cards continue to accurately describe the v1 weights.

---

## What still needs to happen for Option A

The current notebook state is: **doc-aligned text-only**. To complete Option A (combined text + sky-image fine-tune), additional changes are needed once the multimodal dataset exists:

1. **`solarhive_datagen.py`** — image acquisition + image-grounded Q&A generator + multimodal dataset push to `Truthseeker87/solarhive-community-solar-multimodal`. Estimated 13–17 hours of GPU-free work. Not yet started.

2. **`solarhive_finetune.py` Cell 5 + Cell 10** — switch `SFTConfig` to multimodal-mode values: `lr_scheduler_type="cosine"`, `max_grad_norm=0.3`, `dataset_text_field=""`, `dataset_kwargs={"skip_prepare_dataset": True}`, `remove_unused_columns=False`, `max_length=2048` (instead of `max_seq_length=2048`). Add `data_collator=UnslothVisionDataCollator(model, processor)` to the SFTTrainer call.

3. **`solarhive_finetune.py` Cell 3 + Cell 9** — `finetune_vision_layers=True` becomes correct once images are in the dataset (currently True; matches Option A target without change, but the *current* dataset doesn't justify it).

4. **Compute units refresh** — kick off Option A on Colab Pro. Estimated runtime parallel to v1 runs (E4B ~280s + A4B ~4400s on RTX PRO 6000).

---

## Verification

```
$ python -c "import ast; ast.parse(open('solarhive_finetune.py').read())"
OK: parses cleanly

$ grep -c 'chat_template="gemma-4"' solarhive_finetune.py
1   # E4B (was: gemma-4-thinking)
$ grep -c 'chat_template="gemma-4-thinking"' solarhive_finetune.py
1   # A4B (kept — doc-correct for 26B)

$ grep -c 'weight_decay=0.001' solarhive_finetune.py
2   # E4B Cell 5 + A4B Cell 10
$ grep -c 'full_finetuning=False' solarhive_finetune.py
2   # E4B Cell 2 + A4B Cell 8
$ grep -c 'max_grad_norm=1.0' solarhive_finetune.py
2

$ grep -c 'solarhive-e4b-multimodal-ollama' solarhive_finetune.py
2   # HF_REPO assignment + comment header
$ grep -c 'solarhive-26b-a4b-multimodal-lora' solarhive_finetune.py
2   # HF_REPO_A4B assignment + comment header

$ python scripts/py2ipynb.py solarhive_finetune.py
✅ solarhive_finetune.py → solarhive_finetune.ipynb (20 code + 22 markdown cells)
```

---

## Cross-references

| File | Relation |
|---|---|
| `solarhive_finetune.py` / `.ipynb` | Subject of this audit |
| `litert_e2b_audit.md` | Sibling Apr 25 audit on `solarhive_e2b_liteRT_finetune.py`. Same Unsloth doc references; different verdict on Row 1 (E2B used `gemma-4` from the start because the audit was applied pre-run) and Row 2 (E2B applied `False` because LiteRT track is text-only with vision routed to cloud) |
| `litert_plan.md` | Master LiteRT plan; Phase 1 Day 4 references the E2B audit, not this one |
| `MEMORY.md` (auto-memory) | Pinned: `feedback_unsloth_gemma4_chat_template.md`, `project_hf_repo_versioning.md` |
| `hf_model_card_26b_a4b.md`, `hf_model_card_e4b.md` | v1 model cards — describe v1 weights, unchanged content. "Future Versions" subsection added to flag v2 trajectory. |
| `the-gemma4-good-hackathon-solarhive/README.md` | Public landing page — Roadmap subsection added to flag v2 multimodal trajectory. |
