# SolarHive — LiteRT Browser Demo

On-device Gemma 4 E4B inference in the browser. Vanilla HTML/JS/CSS, no build
step. Runs the upstream pre-converted `litert-community/gemma-4-E4B-it-litert-lm`
`.task` bundle via MediaPipe LLM Inference Web on WebGPU. **E4B chosen for
variant consistency** — Cactus mobile, Ollama, llama.cpp all deploy the
fine-tuned E4B; the LiteRT browser tier uses upstream-base E4B (no fine-tune
because the public LiteRT conversion path doesn't yet ship a `gemma4` example
module — see `litert_plan.md`) plus the SolarHive UX layer + on-device
agentic loop on top.

## Files

| File | Purpose |
|---|---|
| `index.html` | Two-screen shell — loading, chat |
| `styles.css` | Mobile-first dark theme (deep navy + sun yellow accents) |
| `app.js` | MediaPipe loader · emoji parser · on-device agentic loop |

## How it works

1. **Model download** — MediaPipe Tasks GenAI fetches the ~2.96 GB `.task`
   bundle from the Hugging Face mirror. Browser caches it; subsequent loads
   are instant.
2. **Inference + on-device agentic loop** — every user query enters
   `runAgenticLoop()` (up to 3 rounds). Base Gemma 4 E4B has [native function
   calling](https://ai.google.dev/gemma/docs/core/model_card_4) (pre-trained,
   not fine-tune-dependent). Two on-device tools execute via the loop —
   `get_solar_production` (Open-Meteo realtime GHI, keyless, CORS-friendly)
   and `get_battery_state` (session simulator). Numerically aligned with
   `solarhive_inference.py` Cell 4 (SYSTEM_EFF=0.85, Fahrenheit-based
   temperature derating, identical return shapes including `temp_derate_pct`,
   `kwh_stored`, `charging`).
3. **Emoji parsing** — `parseEmojiResponse()` extracts the leading 1–2 emojis
   and renders them at 96px. The remaining imperative sentence sits underneath.
4. **Single unified interface for all user groups** — the browser companion
   presents the same chat UI regardless of the user's solar deployment context
   (rooftop-fixed, deployable, off-grid). Mode-driven system-prompt branching
   is a future-roadmap item, retained for post-submission iteration.

## System prompt

A single SolarHive system prompt and the emoji vocabulary are defined in
`litert_plan.md` and inlined in `app.js` (see `systemPrompt()` and
`EMOJI_SET`). Identical between this LiteRT browser app and the Cactus
Flutter app — single source of UX truth across both edge runtimes.

## Browser support

| Browser | Status |
|---|---|
| Chrome 121+ desktop | ✅ WebGPU enabled by default |
| Edge 121+ desktop | ✅ WebGPU enabled by default |
| Chrome Android (recent) | ✅ WebGPU on supported GPUs |
| Safari iOS | ⚠️ WebGPU behind a flag as of the demo date |
| Firefox | ⚠️ WebGPU behind `dom.webgpu.enabled` |

If WebGPU is missing the loading screen halts with a "WebGPU not available"
message. No fallback — the demo is explicitly an edge-AI showcase.

## Local testing

```bash
# any static server works; example with Python:
cd web-litert
python -m http.server 8000
# open http://localhost:8000 in Chrome
```

The first load downloads ~2.6 GB; have a fast connection or use Chrome's
network tab to throttle and verify the progress UI.

## Deployment

Push to a Hugging Face Space with the **Static** SDK. The `.task` bundle
streams from `litert-community/gemma-4-E2B-it-litert-lm` directly — we do
not host model weights in the Space, only the static assets.

```yaml
# README frontmatter for the deployed Space
---
title: SolarHive — LiteRT Browser Demo
emoji: ☀️
sdk: static
pinned: false
license: apache-2.0
---
```

## Known scope

- ✅ scaffold, parser, single unified chat shell
- ⬜ parser hardening, HF Space deployment, demo GIFs

The submission ships a single unified chat interface for all user groups.
Mode-specific dashboards (maintenance-alert layout for rooftop-fixed,
action-countdown layout for deployable / off-grid) are a future-roadmap
item retained for post-submission iteration.

## Citations

This is the LiteRT track entry for the Gemma 4 Good Hackathon
(Google DeepMind × Kaggle). The on-device inference architecture follows
the LiteRT-LM browser pattern documented at
[litert-community/Gemma4](https://huggingface.co/spaces/litert-community/Gemma4).

The tool-routing decision boundary the model learned during fine-tuning
follows [Ross et al. (2025), *When2Call: When (not) to Call Tools*,
arXiv:2504.18851](https://arxiv.org/abs/2504.18851) — taught via the
`SYS_TOOLS` context examples in `solarhive_datagen.py` Cell 7a.

## License

MIT — same as the parent repository.
