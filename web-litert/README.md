# SolarHive — LiteRT Browser Demo

On-device Gemma 4 E2B inference in the browser. Vanilla HTML/JS/CSS, no build
step. Runs the upstream pre-converted `litert-community/gemma-4-E2B-it-litert-lm`
`.task` bundle via MediaPipe LLM Inference Web on WebGPU.

## Files

| File | Purpose |
|---|---|
| `index.html` | Three-screen shell — onboarding, loading, chat |
| `styles.css` | Mobile-first dark theme (deep navy + sun yellow accents) |
| `app.js` | MediaPipe loader · emoji parser · mode toggle · routing escalation |

## How it works

1. **First load** — user picks a mode (🏘️ Suburban for fixed roof panels,
   🌾 Rural / off-grid for deployable panels). Choice persists to `localStorage`.
2. **Model download** — MediaPipe Tasks GenAI fetches the ~2.6 GB `.task`
   bundle from the Hugging Face mirror. Browser caches it; subsequent loads
   are instant.
3. **Inference** — every user query is wrapped in the mode-specific system
   prompt and sent through `LlmInference.generateResponse()` on WebGPU.
4. **Emoji parsing** — `parseEmojiResponse()` extracts the leading 1–2 emojis
   and renders them at 96px. The remaining imperative sentence sits underneath.
5. **Routing escalation** — if the model emits 🛰️, 📡, or 🔬, the UI surfaces
   a one-tap button to escalate to the cloud (26B A4B) or microgrid hub
   (Ollama on E4B). Three-tier task routing made explicit to the user.

## System prompts

The two mode-specific prompts and the emoji vocabulary are defined in
`litert_plan.md` and inlined in `app.js` (see `systemPrompt()` and
`EMOJI_SET`). Identical between this LiteRT browser app and the upcoming
Cactus Flutter app — single source of UX truth across both edge runtimes.

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

## Known scope (Apr 28)

- ✅ Day 8–9: scaffold, onboarding, parser, chat shell
- ⬜ Day 10–11: suburban dashboard with maintenance alerts row
- ⬜ Day 12–13: rural action-card layout with countdown timers
- ⬜ Day 14: parser hardening, HF Space deployment, demo GIFs

This is the Day 8–9 scaffold. The shared chat shell already renders the
emoji card the same way both modes will — Days 10–13 layer mode-specific
secondary panels (maintenance row, action countdown) on top.

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
