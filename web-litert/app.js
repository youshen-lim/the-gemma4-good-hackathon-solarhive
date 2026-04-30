// SolarHive LiteRT — vanilla JS, no build step.
// Runs Gemma 4 E2B in the browser via MediaPipe LLM Inference Web on WebGPU.
// Bundle: upstream litert-community/gemma-4-E2B-it-litert-lm `.task` file.

import { FilesetResolver, LlmInference } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@0.10.27/genai_bundle.mjs";

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

const MODEL_BUNDLE_URL =
  "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task";

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@0.10.27/wasm";

const STORAGE_KEY = "solarhive.mode";  // SUBURBAN | RURAL_OFFGRID

// Emoji vocabulary (must mirror litert_plan.md §"Emoji vocabulary").
// Used both for the system prompt and for the parser fallback.
const EMOJI_SET = [
  // sky / weather
  "☀️", "🌤️", "⛅", "☁️", "🌧️", "💨", "⛈️", "🌅", "🌫️", "❄️", "🌪️",
  // battery
  "🔋", "🪫",
  // grid pricing
  "⚡", "🟢",
  // action
  "📤", "📥", "🧹", "🪛", "💧",
  // maintenance
  "🔧", "⚠️", "📉", "🍂",
  // routing escalation (UI affordances)
  "🔬", "🛰️", "📡",
  // time + load
  "⏰",
];

const ROUTING_EMOJIS = {
  "🔬": "vqa",     // panel-photo VQA → 26B A4B with image
  "🛰️": "cloud",   // cloud reasoning → HF Space (26B A4B)
  "📡": "hub",     // microgrid hub  → Ollama endpoint
};

// Per-mode example chips
const CHIPS = {
  SUBURBAN: [
    "How is solar today?",
    "Run dishwasher now?",
    "Battery status?",
    "Anything wrong with my panels?",
  ],
  RURAL_OFFGRID: [
    "Should I deploy now?",
    "Storm coming?",
    "Adjust angle for low sun?",
    "Quick clean today?",
  ],
};

// System prompts per mode (from litert_plan.md §"System prompt for both modes")
function systemPrompt(mode) {
  return `You are SolarHive, an AI energy advisor for community solar households in ${mode} mode. Reply in this exact format:

[1-2 emojis] [one short imperative sentence under 15 words]

Use these emoji codes:
- Sky: ☀️ 🌤️ ⛅ ☁️ 🌧️ 💨 ⛈️
- Battery: 🔋 🪫
- Grid: ⚡ 🟢
- Action — deploy: 📤   stow: 📥   clean: 🧹   adjust: 🪛
- Maintenance: 🔧 ⚠️ 📉
- Escalate: 🔬 (panel photo analysis) 🛰️ (cloud reasoning) 📡 (microgrid hub)

Examples:
"Deploy panels now?" → "🌅📤 Deploy now, peak GHI window opens in 30 min."
"Should I run dishwasher?" → "☀️🟢 Run now, free solar plus off-peak grid."
"Storm warning incoming?" → "⛈️📥 Stow panels in next 20 min, gusts above 35 mph."
"Why is output low?" → "🍂🧹 Leaves accumulated this week, clean before Saturday."`;
}

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------

let mode = null;        // SUBURBAN | RURAL_OFFGRID
let llm = null;         // MediaPipe LlmInference instance
let busy = false;

// ----------------------------------------------------------------------------
// DOM helpers
// ----------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

function showScreen(id) {
  for (const el of document.querySelectorAll(".screen")) el.hidden = true;
  $(id).hidden = false;
}

function setProgress(pct, status) {
  $("progress-bar").style.width = `${Math.round(pct * 100)}%`;
  if (status) $("loading-status").textContent = status;
}

// ----------------------------------------------------------------------------
// Emoji parser
// ----------------------------------------------------------------------------

// Build a single regex matching any emoji in our vocabulary, longest-first
// (so the variation-selector forms like "🌤️" match before bare "🌤").
const EMOJI_REGEX = (() => {
  const escaped = [...EMOJI_SET]
    .sort((a, b) => b.length - a.length)
    .map((e) => e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  return new RegExp(`^\\s*((?:${escaped})+)\\s*(.*)$`, "us");
})();

/**
 * Parse a model response into { emojis, text, routing }.
 * - emojis: string of 1-2 leading emojis (or "❓" if none recognized)
 * - text: the imperative tail with the leading emojis stripped
 * - routing: "vqa" | "cloud" | "hub" | null — set if any routing emoji present
 */
function parseEmojiResponse(raw) {
  if (!raw) return { emojis: "❓", text: "", routing: null };
  const trimmed = raw.trim();
  const m = trimmed.match(EMOJI_REGEX);

  let emojis, text;
  if (m) {
    emojis = m[1];
    text = m[2].trim();
  } else {
    // Fallback: scan first 24 chars for any vocabulary emoji
    const found = [];
    for (const e of EMOJI_SET) {
      if (trimmed.slice(0, 24).includes(e)) found.push(e);
      if (found.length >= 2) break;
    }
    emojis = found.length > 0 ? found.join("") : "❓";
    text = trimmed;
  }

  // Detect routing escalation
  let routing = null;
  for (const [emoji, kind] of Object.entries(ROUTING_EMOJIS)) {
    if (emojis.includes(emoji)) { routing = kind; break; }
  }

  return { emojis, text, routing };
}

// Expose parser for testing in DevTools
window.__solarhive_parse = parseEmojiResponse;

// ----------------------------------------------------------------------------
// Model loading
// ----------------------------------------------------------------------------

async function loadModel() {
  showScreen("loading");
  setProgress(0.05, "Initializing WebGPU runtime…");

  if (!("gpu" in navigator)) {
    setProgress(1, "WebGPU not available. Use Chrome/Edge desktop or Chrome Android.");
    return false;
  }

  setProgress(0.15, "Resolving MediaPipe assets…");
  const fileset = await FilesetResolver.forGenAiTasks(WASM_BASE);

  setProgress(0.25, "Downloading Gemma 4 E2B bundle (~2.6 GB on first load)…");
  llm = await LlmInference.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_BUNDLE_URL },
    maxTokens: 512,
    topK: 64,
    topP: 0.95,
    temperature: 1.0,
    randomSeed: 3407,
  });

  setProgress(1, "Ready.");
  return true;
}

// ----------------------------------------------------------------------------
// Inference
// ----------------------------------------------------------------------------

async function ask(userQuery) {
  if (busy || !llm) return;
  busy = true;
  $("query-submit").disabled = true;
  $("emoji-text").textContent = "Thinking…";

  const prompt = `<start_of_turn>user
${systemPrompt(mode)}

${userQuery}<end_of_turn>
<start_of_turn>model
`;

  try {
    const raw = await llm.generateResponse(prompt);
    const { emojis, text, routing } = parseEmojiResponse(raw);
    $("emoji-large").textContent = emojis;
    $("emoji-text").textContent = text || "(no text)";
    showRouting(routing);
  } catch (err) {
    console.error(err);
    $("emoji-large").textContent = "⚠️";
    $("emoji-text").textContent = "Inference failed. Check console.";
    showRouting(null);
  } finally {
    busy = false;
    $("query-submit").disabled = false;
  }
}

function showRouting(kind) {
  $("routing-actions").hidden = kind === null;
  $("route-vqa").hidden = kind !== "vqa";
  $("route-cloud").hidden = kind !== "cloud";
  $("route-hub").hidden = kind !== "hub";
}

// ----------------------------------------------------------------------------
// Mode toggle + chips
// ----------------------------------------------------------------------------

function applyMode(nextMode) {
  mode = nextMode;
  localStorage.setItem(STORAGE_KEY, nextMode);
  const pill = nextMode === "SUBURBAN" ? "🏘️ Suburban" : "🌾 Rural / off-grid";
  $("mode-pill").textContent = pill;
  renderChips();
}

function renderChips() {
  const row = $("chip-row");
  row.innerHTML = "";
  for (const chip of CHIPS[mode] || []) {
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.type = "button";
    btn.textContent = chip;
    btn.addEventListener("click", () => {
      $("query-input").value = chip;
      $("query-form").requestSubmit();
    });
    row.appendChild(btn);
  }
}

// ----------------------------------------------------------------------------
// Wire-up
// ----------------------------------------------------------------------------

function bind() {
  // Onboarding
  for (const btn of document.querySelectorAll(".mode-btn")) {
    btn.addEventListener("click", async () => {
      applyMode(btn.dataset.mode);
      const ok = await loadModel();
      if (ok) showScreen("chat");
    });
  }

  // Switch mode (back to onboarding without re-downloading model)
  $("reset-mode").addEventListener("click", () => {
    showScreen("onboarding");
  });

  // Query submit
  $("query-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("query-input").value.trim();
    if (!q) return;
    $("query-input").value = "";
    ask(q);
  });

  // Routing escalation handlers (placeholders — wire to real endpoints later)
  $("route-cloud").addEventListener("click", () => {
    window.open("https://huggingface.co/spaces/Truthseeker87/solarhive-26b-a4b", "_blank");
  });
  $("route-hub").addEventListener("click", () => {
    alert("Microgrid hub sync not yet available in browser demo. Use Ollama endpoint locally.");
  });
  $("route-vqa").addEventListener("click", () => {
    alert("Panel-photo VQA routes to the cloud 26B A4B model — feature in progress.");
  });
}

function boot() {
  bind();
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === "SUBURBAN" || saved === "RURAL_OFFGRID") {
    applyMode(saved);
    loadModel().then((ok) => { if (ok) showScreen("chat"); });
  } else {
    showScreen("onboarding");
  }
}

boot();
