// SolarHive LiteRT — vanilla JS, no build step.
// Runs Gemma 4 E4B in the browser via MediaPipe LLM Inference Web on WebGPU.
// Bundle: upstream litert-community/gemma-4-E4B-it-litert-lm `.task` file.
// E4B chosen for variant consistency with every other SolarHive tier
// (Cactus mobile, Ollama, llama.cpp all use fine-tuned E4B; cloud uses 26B A4B).

import { FilesetResolver, LlmInference } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@0.10.27/genai_bundle.mjs";

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

const MODEL_BUNDLE_URL =
  "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-web.task";

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

// System prompts per mode (from litert_plan.md §"System prompt for both modes").
// Hybrid tool-calling pattern: keyless tools (Open-Meteo, battery simulator) execute
// on-device via the agentic loop; keyed tools (OWM/EIA/NREL) route to the microgrid
// hub via 📡 because API keys cannot live in browser source.
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

Two on-device tools are available for grounded answers (call by emitting "call:NAME{}"):
- get_solar_production() — production_kw, capacity_kw 72, efficiency_pct, ghi_wm2 (Open-Meteo realtime irradiance), temp_derate_pct, source. Keyless API, no arguments.
- get_battery_state() — soc_pct, kwh_stored, capacity_kwh 100, charging (true if soc < 50). Session simulator, no arguments.

Call these tools when the user asks about CURRENT solar production, output efficiency, weather-impact on production, or battery state. For keyed-API queries (specific weather forecast, grid pricing, NREL PVWatts annual baseline) emit 📡 to route to the microgrid hub. For cloud reasoning emit 🛰️. For panel-photo VQA emit 🔬.

Examples:
"Deploy panels now?" → "🌅📤 Deploy now, peak GHI window opens in 30 min."
"Should I run dishwasher?" → "☀️🟢 Run now, free solar plus off-peak grid."
"How is solar today?" → call:get_solar_production{}, then "☀️ 820 W/m² GHI, 51 kW out of 72 kW."
"Battery status?" → call:get_battery_state{}, then "🔋 72% SoC, 72 kWh stored, discharging."
"What's the grid rate?" → "📡 Sync with microgrid hub for live grid pricing."
"Storm warning incoming?" → "⛈️📥 Stow panels in next 20 min, gusts above 35 mph."
"Why is output low?" → "🍂🧹 Leaves accumulated this week, clean before Saturday."`;
}

// ----------------------------------------------------------------------------
// On-device tools (keyless only — keyed APIs route via 📡)
// ----------------------------------------------------------------------------

// Battery simulator — mirrors `_BatterySimulator` in solarhive_inference.py:
//   self.soc = round(random.uniform(55, 85), 1)  # randomize once at session start
//   def get_state(self):
//       kwh = round(self.soc / 100 * self.capacity)
//       return {"soc_pct": self.soc, "kwh_stored": kwh,
//               "capacity_kwh": self.capacity, "charging": self.soc < 50}
// Static within session — no time-of-day drift (Python doesn't drift either).
const BATTERY = {
  soc_pct: Math.round((55 + Math.random() * 30) * 10) / 10,  // uniform(55, 85), 1dp
  capacity_kwh: 100,
};

async function tool_get_solar_production() {
  // Mirrors solarhive_inference.py get_solar_production():
  //   SYSTEM_EFF = 0.85  # inverter 97% × wiring 98% × soiling 97% × mismatch 98%
  //   temp_derate = max(0.75, 1.0 - 0.004 * max(0, temp_f - 77))
  //   production = max(0, COMMUNITY_CAPACITY_KW * (ghi / 1000) * SYSTEM_EFF * temp_derate)
  const COMMUNITY_CAPACITY_KW = 72;
  const SYSTEM_EFF = 0.85;
  const url = "https://api.open-meteo.com/v1/forecast?latitude=42.2808&longitude=-83.7430&current=shortwave_radiation,temperature_2m,cloud_cover&timezone=America/Detroit";
  try {
    const resp = await fetch(url);
    if (!resp.ok) return { error: `Open-Meteo HTTP ${resp.status}` };
    const data = await resp.json();
    const ghi = Number(data?.current?.shortwave_radiation ?? 0);
    const temp_c = Number(data?.current?.temperature_2m ?? 20);
    const temp_f = (temp_c * 9 / 5) + 32;
    const temp_derate = Math.max(0.75, 1.0 - 0.004 * Math.max(0, temp_f - 77));
    const production_kw = Math.max(
      0,
      Math.round(COMMUNITY_CAPACITY_KW * (ghi / 1000) * SYSTEM_EFF * temp_derate * 10) / 10,
    );
    return {
      production_kw,
      capacity_kw: COMMUNITY_CAPACITY_KW,
      efficiency_pct: Math.round((production_kw / COMMUNITY_CAPACITY_KW) * 1000) / 10,
      ghi_wm2: Math.round(ghi * 10) / 10,
      temp_derate_pct: Math.round(temp_derate * 1000) / 10,
      source: "open-meteo",
    };
  } catch (e) {
    return { error: `fetch failed: ${String(e)}` };
  }
}

function tool_get_battery_state() {
  const kwh_stored = Math.round((BATTERY.soc_pct / 100) * BATTERY.capacity_kwh);
  return {
    soc_pct: BATTERY.soc_pct,
    kwh_stored,
    capacity_kwh: BATTERY.capacity_kwh,
    charging: BATTERY.soc_pct < 50,
  };
}

const TOOL_DISPATCH = {
  get_solar_production: tool_get_solar_production,
  get_battery_state: tool_get_battery_state,
};

// ----------------------------------------------------------------------------
// Tool-call parser (JS port of `_extract_tool_calls` + `_parse_tool_args`
// from solarhive_inference.py). Two-pattern fallback: wrapped Google form
// `<|tool_call>call:fn{args}<tool_call|>` first, then bare `call:fn{args}`
// for thinking-mode-stripped outputs.
// ----------------------------------------------------------------------------

function parseToolArgs(raw) {
  const args = {};
  if (!raw || !raw.trim()) return args;
  // Split on commas at the top level (no nesting expected for our scoped tools)
  const parts = raw.split(/,(?=\s*\w+\s*:)/);
  for (const part of parts) {
    const idx = part.indexOf(":");
    if (idx < 0) continue;
    const key = part.slice(0, idx).trim();
    let val = part.slice(idx + 1).trim();
    // Strip <|"|> wrappers + plain quotes
    val = val.replace(/^<\|"\|>/, "").replace(/<\|"\|>$/, "");
    val = val.replace(/^"/, "").replace(/"$/, "");
    if (val === "true") args[key] = true;
    else if (val === "false") args[key] = false;
    else if (val === "null" || val === "None") args[key] = null;
    else if (/^-?\d+$/.test(val)) args[key] = parseInt(val, 10);
    else if (/^-?\d*\.\d+$/.test(val)) args[key] = parseFloat(val);
    else args[key] = val;
  }
  return args;
}

function extractToolCalls(raw) {
  // Mirror solarhive_inference.py exactly:
  //   _TOOL_CALL_WRAPPED_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL)
  //   _TOOL_CALL_BARE_RE    = re.compile(r"call:(\w+)\{([^}]*)\}")
  // Wrapped uses lazy `.*?` with dotall (s flag) — required so string args
  // delimited by `<|"|>...<|"|>` are not truncated at the first `<` char.
  // Bare has NO leading anchor — matches mid-sentence emissions too.
  const wrapped = /<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>/gs;
  const bare = /call:(\w+)\{([^}]*)\}/g;
  const calls = [];
  let m;
  while ((m = wrapped.exec(raw)) !== null) {
    calls.push({ name: m[1], args: parseToolArgs(m[2]) });
  }
  if (calls.length === 0) {
    while ((m = bare.exec(raw)) !== null) {
      calls.push({ name: m[1], args: parseToolArgs(m[2]) });
    }
  }
  return calls;
}

// Expose for DevTools testing — these implement the same semantics as
// _extract_tool_calls + _parse_tool_args in solarhive_inference.py Cell 3.
// Sample probe: __solarhive_extract_tool_calls('<|tool_call>call:get_weather{location:<|"|>Ann Arbor, MI<|"|>}<tool_call|>')
window.__solarhive_extract_tool_calls = extractToolCalls;
window.__solarhive_parse_tool_args = parseToolArgs;

// ----------------------------------------------------------------------------
// Agentic loop — up to 3 rounds of tool calls, mirrors solarhive_inference.py
// Cell 4 generate_with_tools(). Uses 2-message tool_calls + tool_response
// sequence to match the format the cloud + hub variants are trained on.
// ----------------------------------------------------------------------------

async function runAgenticLoop(userQuery, maxRounds = 3) {
  let history = `<start_of_turn>user
${systemPrompt(mode)}

${userQuery}<end_of_turn>
<start_of_turn>model
`;
  for (let round = 0; round < maxRounds; round++) {
    const raw = await llm.generateResponse(history);
    const calls = extractToolCalls(raw);
    if (calls.length === 0) {
      return raw; // Final answer — no more tool calls
    }
    let toolBlock = "";
    for (const call of calls) {
      const fn = TOOL_DISPATCH[call.name];
      let result;
      if (!fn) {
        result = {
          error: `Unknown on-device tool '${call.name}'. Available: get_solar_production, get_battery_state. Keyed tools (weather/grid/PVWatts) route to the microgrid hub via 📡.`,
        };
      } else {
        try { result = await fn(call.args); }
        catch (e) { result = { error: String(e) }; }
      }
      toolBlock += `<start_of_turn>tool
${call.name}: ${JSON.stringify(result)}<end_of_turn>
`;
    }
    history += `${raw}<end_of_turn>
${toolBlock}<start_of_turn>model
`;
  }
  // Max rounds hit — final completion
  return await llm.generateResponse(history);
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

  setProgress(0.25, "Downloading Gemma 4 E4B bundle (~3 GB on first load)…");
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

  try {
    // Agentic loop: base Gemma 4 supports native function calling per the model card.
    // Up to 3 rounds of on-device tool calls (get_solar_production, get_battery_state)
    // before the final emoji-format response. Keyed-API queries return a 📡 routing
    // emoji instead of a tool call (no API keys live in browser source).
    const raw = await runAgenticLoop(userQuery, 3);
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
