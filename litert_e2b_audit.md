# LiteRT E2B Fine-Tune — Audit & Changes Record

**Date:** April 25, 2026
**Notebook:** `solarhive_e2b_liteRT_finetune.py` / `.ipynb`
**Phase:** Pre-Phase 1 Day 4 hardening (before fine-tune run)
**Outcome:** Two passes — Unsloth doc compliance + LiteRT compatibility — applied as code edits before any Colab run.
**Sibling audit:** `solarhive_finetune_audit.md` covers the parallel E4B + 26B A4B notebook. Same Unsloth doc references; different per-row verdicts because E4B/A4B v1 weights already shipped and compute units are exhausted (forward-only edits + v2 repo split, no retrain).

---

## Context

The E2B fine-tune notebook produces `Truthseeker87/solarhive-e2b-merged`, which is the input to `litert-torch`'s `export_hf` conversion (`solarhive_litert_e2b_phase0.py` Cell 7) for the `.litertlm` LiteRT bundle. A bug introduced at fine-tune time costs both a training cycle (~30–60 min on Colab Pro) and a conversion cycle (~30 min) before it's caught — so the notebook was audited against the upstream docs before its first run.

Two audit passes were run:

1. **Unsloth Gemma 4 documentation** — does the LoRA fine-tune config match what Unsloth recommends?
2. **LiteRT-LM / litert-torch contract** — does the merged-safetensors output meet the requirements of `export_hf --use_jinja_template --bundle_litert_lm`?

---

## Sources

### Unsloth Gemma 4
- https://unsloth.ai/docs/models/gemma-4 — overview, supported variants
- https://unsloth.ai/docs/models/gemma-4/train — fine-tuning code patterns
- https://unsloth.ai/docs/models/gemma-4/train#bug-fixes--tips — gotchas

### LiteRT
- https://github.com/google-ai-edge/LiteRT-LM — runtime + CLI
- https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm — reference bundle (2.58 GB: 0.79 GB text decoder + 1.12 GB embedder + 2 GB browser `.task`)
- https://developers.googleblog.com/google-ai-edge-small-language-models-multimodality-rag-function-calling/ — stack overview, function calling on Android, RAG library

---

## Part 1 — Unsloth Doc Audit

### Findings (5 issues)

| # | Severity | Location | Issue | Doc cite |
|---|---|---|---|---|
| 1 | **High** | Cell 3 — `get_chat_template` | `chat_template="gemma-4-thinking"` used on E2B | Unsloth Tip #1: *"Use `gemma-4-thinking` for 26B/31B; `gemma-4` for smaller models."* Tip #11 also warns mismatched template at inference degrades runtime quality. |
| 2 | **High** | Cell 4 — `get_peft_model` | `finetune_vision_layers=True` on a text-only dataset | Unsloth Tip #5: *"Start with `finetune_vision_layers = False`, fine-tune language/attention/MLP only. Enable vision or audio layers later if your task needs it."* Vision LoRA receives zero training signal without images. |
| 3 | Medium | Cell 3 — `from_pretrained` | Missing `dtype`, `max_seq_length`, `full_finetuning` args | Unsloth documented call: `FastModel.from_pretrained(model_name=..., dtype=None, max_seq_length=8192, load_in_4bit=True, full_finetuning=False)`. |
| 4 | Medium | Cell 5 — `SFTConfig` | Missing `weight_decay`, `lr_scheduler_type`, `max_grad_norm` | Unsloth documented text SFTConfig: `weight_decay=0.001, lr_scheduler_type="linear", max_grad_norm` (text) / `0.3` (vision). |
| 5 | Low | Cell 3 — loader class | `FastVisionModel` used despite text-only dataset and no `UnslothVisionDataCollator` | Unsloth text path uses `FastModel.from_pretrained`. Kept `FastVisionModel` for cross-fine-tune training-provenance parity with E4B/A4B (same dataset, same loader); decision documented here. |

### Pre-fix code (Cell 3, 4, 5)

```python
# Cell 3 — pre-audit
e2b_model, e2b_processor = FastVisionModel.from_pretrained(
    model_name=E2B_MODEL_PATH,
    load_in_4bit=_e2b_use_4bit,
    use_gradient_checkpointing="unsloth",
)
e2b_processor = get_chat_template(e2b_processor, chat_template="gemma-4-thinking")

# Cell 4 — pre-audit
e2b_model = FastVisionModel.get_peft_model(
    e2b_model,
    finetune_vision_layers=True,       # VLM — train vision encoder alongside language
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    random_state=3407, use_rslora=False, loftq_config=None,
    target_modules="all-linear",
)

# Cell 5 — pre-audit (excerpt)
args=SFTConfig(
    per_device_train_batch_size=_e2b_batch,
    gradient_accumulation_steps=_e2b_accum,
    warmup_steps=5,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    output_dir="solarhive_e2b_out",
    optim="adamw_8bit",
    seed=3407,
    report_to="none",
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
)
```

### Post-fix code

```python
# Cell 3 — post-audit
e2b_model, e2b_processor = FastVisionModel.from_pretrained(
    model_name=E2B_MODEL_PATH,
    load_in_4bit=_e2b_use_4bit,
    use_gradient_checkpointing="unsloth",
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    full_finetuning=False,
)
# Per Unsloth Gemma 4 train docs: use "gemma-4" for E2B/E4B, "gemma-4-thinking"
# only for the larger 26B/31B variants. The thinking template emits
# <|channel>thought blocks that conflict with the LiteRT emoji format probe.
e2b_processor = get_chat_template(e2b_processor, chat_template="gemma-4")

# Cell 4 — post-audit
# Per Unsloth Gemma 4 train tip: start with finetune_vision_layers=False and
# only enable vision/audio later if the dataset has those modalities. This
# notebook trains on text-only Q&A — turning vision layers on would consume
# trainable params on encoders that get no signal during training.
e2b_model = FastVisionModel.get_peft_model(
    e2b_model,
    finetune_vision_layers=False,      # was: True
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    random_state=3407, use_rslora=False, loftq_config=None,
    target_modules="all-linear",
)

# Cell 5 — post-audit (additions)
args=SFTConfig(
    ...,
    optim="adamw_8bit",
    weight_decay=0.001,                # NEW
    lr_scheduler_type="linear",        # NEW
    max_grad_norm=1.0,                 # NEW
    seed=3407,
    ...,
)
```

### Validated against docs (no change needed)

- `r=16, lora_alpha=16, lora_dropout=0, bias="none"` — between Unsloth's text default (8/8) and multimodal (32/32); defensible
- `optim="adamw_8bit"`, `seed=3407`, `warmup_steps=5`, `learning_rate=2e-4`, `logging_steps=1`, `report_to="none"` — match docs exactly
- `dataset_text_field="text"` — correct for the pre-rendered chat-template strings in `Truthseeker87/solarhive-community-solar-1k`
- `bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported()` — correctly avoids fp16. Bug Fix #5 in Unsloth docs warns about Gemma4Audio fp16 overflow on T4; bf16 is safer regardless of modality.
- `use_gradient_checkpointing="unsloth"` — Tip #8: *"Keep this on (it's designed to reduce VRAM use and extend context length)."*
- Inference sampling `temperature=1.0, top_p=0.95, top_k=64` — matches Google's recommended Gemma 4 defaults
- `FastVisionModel.for_inference(e2b_model)` before benchmark generation — correct
- Pinned `unsloth==2026.4.5` — assumed post Bug Fix #2 (gradient accumulation) and Bug Fix #4 (`use_cache=False` E2B/E4B garbage)
- `save_pretrained_merged(..., save_method="merged_16bit")` — correct for LiteRT input (LiteRT-Torch wants merged BF16 safetensors, not GGUF)

### Loss expectation note

Plan + notebook expect *"converged loss ~1.0–1.2"* for E2B fine-tune. Unsloth Bug Fix #1 says: *"Gemma-4 E2B and E4B having a loss of 13-15 is perfectly normal — characteristic of multimodal architectures."* This sounds contradictory but isn't — the 13–15 baseline only applies when image/audio tokens contribute to the loss. Pure text training on E2B (which is what this notebook does) should still land in the 1–3 range. Run 2 confirmed E4B converged at 0.952 on the same dataset; same expectation holds for E2B. The expected-loss comment is correct only because Issue #2 above is fixed (vision layers off → no multimodal loss surface).

---

## Part 2 — LiteRT Compatibility Audit

### Cell 8 contract

Cell 8 produces the merged-safetensors artifact that `litert-torch.generative.export_hf` consumes in Phase 0 Cell 7. The Phase 0 CLI is:

```bash
python -m litert_torch.generative.export_hf \
    /path/to/solarhive-e2b-merged /out/solarhive-e2b \
    --prefill_lengths=[256] \
    --cache_length=4096 \
    --externalize_embedder \
    --use_jinja_template \
    --bundle_litert_lm
```

The merged dir must contain at minimum:
- `config.json` (model architecture metadata)
- `tokenizer_config.json` (tokenizer + embedded chat template)
- `chat_template.jinja` (required when `--use_jinja_template` is passed)
- `*.safetensors` shards (BF16 weights)
- `preprocessor_config.json` (image/audio preprocessor — only matters if vision/audio bundling is later requested)

### Findings (4 items)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | **High** | `chat_template.jinja` may not be written as a separate file; transformers serializes the template into `tokenizer_config.json` reliably, but the standalone `.jinja` write varies by version. `--use_jinja_template` requires the file. | Defensive write: extract `chat_template` from `tokenizer_config.json` and write `chat_template.jinja` if absent. |
| 2 | Medium | Variable name `_e2b_merge_tokenizer` was misleading — `FastVisionModel.from_pretrained` returns the full multimodal processor (tokenizer + image_processor + audio_feature_extractor), not a bare tokenizer. Made it harder to reason about whether `preprocessor_config.json` was being saved. | Renamed to `_e2b_merge_processor`. No behavior change — `save_pretrained_merged` was already receiving the full processor. |
| 3 | Medium | No verification of merged dir contents before pushing 5 GB to HF. Silent merge failure (e.g., disk full mid-save) would only surface at Phase 0 conversion. | Added file listing + total size print. |
| 4 | Medium | No preflight check for the four required files before HF upload. | Added explicit check for `config.json`, `tokenizer_config.json`, `chat_template.jinja`, and at least one `*.safetensors` shard. |

### Post-fix Cell 8 addition

```python
# === LiteRT compatibility verification =======================================
# litert-torch's `export_hf --use_jinja_template` (run in Phase 0/1 Day 6)
# expects `chat_template.jinja` as a separate file. Transformers serializes
# the chat template into tokenizer_config.json; whether it ALSO writes the
# standalone .jinja varies by version. Write it explicitly so the conversion
# step never fails on a missing file. Also list the merged dir so any
# missing artifact is caught locally before the 5 GB HF upload.
_jinja_path = os.path.join(_e2b_merged, "chat_template.jinja")
_tok_cfg_path = os.path.join(_e2b_merged, "tokenizer_config.json")
if not os.path.exists(_jinja_path) and os.path.exists(_tok_cfg_path):
    with open(_tok_cfg_path, "r", encoding="utf-8") as _f:
        _tok_cfg = json.load(_f)
    _tmpl = _tok_cfg.get("chat_template")
    if isinstance(_tmpl, str) and _tmpl.strip():
        with open(_jinja_path, "w", encoding="utf-8") as _f:
            _f.write(_tmpl)

# Required files for litert-torch generative.export_hf
_required = ["config.json", "tokenizer_config.json", "chat_template.jinja"]
_missing = [f for f in _required if not os.path.exists(os.path.join(_e2b_merged, f))]
_has_safetensors = any(
    f.endswith(".safetensors") for f in os.listdir(_e2b_merged)
    if os.path.isfile(os.path.join(_e2b_merged, f))
)
```

(Full block including file listing in the notebook itself.)

### Deliberately NOT changed in Cell 8

- **Merge dtype stays at `merged_16bit` (BF16)** — correct input format for `litert-torch export_hf`. The reference `litert-community/gemma-4-E2B-it-litert-lm` bundle is itself produced from BF16 inputs.
- **No vision/audio bundling additions** — text-only fine-tune feeds the text decoder path. Per the three-tier plan, vision routes via 🔬 to cloud 26B A4B, not to the LiteRT bundle. Adding vision/audio bundling would multiply Phase 0 risk for no demo benefit.
- **No function-calling artifacts** — Google's "AI Edge Function Calling" is Android-only per the dev blog. The LiteRT-LM repo positions tool use as a runtime feature, not a conversion artifact. Per `litert_plan.md`, on-device tool calling is explicitly skipped (📡 emoji escalates to the microgrid hub).
- **`.task` browser format not produced here** — Phase 2 Day 8 has an open audit item: whether `export_hf` has a `.task`-emitting flag, or whether MediaPipe Tasks Web has its own converter. That's separate scope; Cell 8 should not attempt blindly.

---

## Part 3 — Effect on Emoji UX (Suburban + Rural)

The fix that matters for the LiteRT browser deployment is **Issue 1 in Part 1 (chat template)**. Pre-fix, the thinking template wraps every E2B response:

```
<|channel>thought
[internal reasoning about clouds, GHI, battery...]
<channel|>
☀️🟢 Run now, peak solar plus off-peak grid.
```

The Cell 7 emoji-wellformed heuristic checks the **first whitespace-delimited token** for a non-ASCII codepoint. With the thinking wrapper, the first token is `<|channel>thought`, not an emoji — every probe fails. In the browser, the user would see `<channel>` tags rendered as literal text.

Post-fix (`gemma-4`, non-thinking template), responses are direct:

```
☀️🟢 Run now, peak solar plus off-peak grid.
```

This is exactly what the UI emoji parser expects, and it matches the few-shot examples in `SYS_SUBURBAN_BASE`/`SYS_RURAL_BASE`. The Cell 7 ≥7/10 gate is much more likely to pass.

### Mode-specific implications

Both modes share the `[1-2 emojis] [imperative <15w]` contract — neither needed thinking blocks. The rural mode benefits *more* because its escalation emojis (🛰️ cloud, 📡 hub, ⛈️📥 stow-now) are time-sensitive — a 200-token reasoning preamble before "stow panels in next 20 min" would defeat the action-prompt UX entirely.

### Distribution alignment note

Training data was rendered with `enable_thinking=False` (project pattern). Pre-fix, the runtime template (`gemma-4-thinking`) was off-distribution at inference vs. training. Post-fix, runtime template (`gemma-4`) agrees with the training template — another reason the Cell 7 probe should improve.

### Other audit changes — no emoji impact

- `finetune_vision_layers=False` — text-only data has no images, vision encoder got zero signal regardless. Same emoji output, fewer wasted LoRA params. The 🔬 escalation pattern still works because base Gemma 4 vision weights pass through unchanged in `save_pretrained_merged` (LoRA didn't touch them).
- Loader args (`max_seq_length`, `dtype`, `full_finetuning`) — plumbing alignment with documented loader, no effect on output.
- SFTConfig additions (`weight_decay=0.001`, `lr_scheduler_type="linear"`, `max_grad_norm=1.0`) — small training stability improvements; `linear` is already the default and `max_grad_norm=1.0` matches transformers' default. Marginal upside, no qualitative change.
- Cell 8 LiteRT preflight — deployment-time plumbing for the Phase 0 handoff, no runtime behavior change.

---

## Verification

```
$ python -c "import ast; ast.parse(open('solarhive_e2b_liteRT_finetune.py').read())"
OK

$ grep -c "finetune_vision_layers=True" solarhive_e2b_liteRT_finetune.py
0
$ grep -c "finetune_vision_layers=False" solarhive_e2b_liteRT_finetune.py
2  # Cell 4 actual + Cell 4 docstring "vision_layers=False" comment
$ grep -c "_e2b_merge_tokenizer" solarhive_e2b_liteRT_finetune.py
0  # fully renamed
$ grep -c "chat_template.jinja" solarhive_e2b_liteRT_finetune.py
5  # write + verification + comment references

$ python scripts/py2ipynb.py solarhive_e2b_liteRT_finetune.py
✅ solarhive_e2b_liteRT_finetune.py → solarhive_e2b_liteRT_finetune.ipynb (9 code + 10 markdown cells)

$ bash scripts/validate.sh
=== ✅ All checks passed (8 passed, 0 failed) ===
```

---

## Cross-references

| File | Relation |
|---|---|
| `litert_plan.md` | Master plan — Phase 1 Day 4–5 spec for the fine-tune notebook |
| `solarhive_e2b_liteRT_finetune.py` / `.ipynb` | Subject of this audit; both files now carry all four Unsloth fixes + Cell 8 LiteRT preflight |
| `solarhive_litert_e2b_phase0.py` | Downstream consumer — uses `litert-torch export_hf --use_jinja_template` against the merged repo |
| `hf_model_card_e2b_litert.md` | Public model card — already drafted; describes the deterministic-workflow + E2B-reasoning deployment pattern |
| `the-gemma4-good-hackathon-solarhive/README.md` | Has uncommitted "On-device reasoning pattern" section pending repo publish |
| `MEMORY.md` (auto-memory) | Pinned the Unsloth `chat_template="gemma-4"` gotcha for future sessions |

---

## Open questions / next steps

1. **Run Phase 0 on Colab Pro High-RAM** — clear the `solarhive_litert_e2b_phase0.py` abort gate (3 days overdue per the original Apr 22–24 schedule).
2. **Verify Kaggle E2B slug** — `kagglehub.model_download("google/gemma-4/transformers/gemma-4-e2b-it")`. Inferred from E4B/A4B pattern; not yet confirmed.
3. **Resolve `.task` export path for Phase 2** — does `export_hf` have a `.task`-emitting sibling flag, or does MediaPipe Tasks Web ship its own converter? Open audit item.
4. **Run the fine-tune notebook end-to-end** — once Phase 0 passes, this notebook executes Phase 1 Day 4–5 in ~30–60 min on Colab Pro; pushes `Truthseeker87/solarhive-e2b-merged` to HF.
5. **Confirm Cell 7 emoji probe ≥7/10** — gate for proceeding to LiteRT browser demo. With the chat template fix, expectation is now ≥8/10 baseline.
