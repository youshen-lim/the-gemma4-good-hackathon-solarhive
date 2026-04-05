# SolarHive
## AI-Powered Community Energy Intelligence

> **Gemma 4 Good Hackathon** — Google DeepMind × Kaggle
> **Track:** Global Resilience · Climate & Green Energy
> **Special Technology Tracks:** Ollama · Unsloth

[![Kaggle](https://img.shields.io/badge/Kaggle-Gemma%204%20Good%20Hackathon-20BEFF?logo=kaggle)](https://kaggle.com/competitions/gemma-4-good-hackathon)
[![Model](https://img.shields.io/badge/Gemma%204-26B%20MoE-4285F4?logo=google)](https://kaggle.com/models/google/gemma-4)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-black?logo=ollama)](https://ollama.com/library/gemma4)
[![Unsloth](https://img.shields.io/badge/Unsloth-Fine--Tuned-FF6B35)](https://unsloth.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Dream

> *Each household in any given community is a potential energy producer
> and consumer — making it a potential clean energy island.*
>
> *Community-based storage such as fuel cells (e.g., solid oxide fuel
> cells or salt-based fuel cells) will help capture excess energy in
> each community of households to compensate for low production
> efficiency of solar.*
>
> *Energy can be distributed across communities based on energy needs
> in a decentralized live or hybrid grid — as opposed to a dead grid.*

**SolarHive is the AI intelligence layer that makes this dream actionable.**

Powered by Gemma 4, it transforms fragmented household-level solar data
into a unified, conversational, visual community energy picture —
helping suburban neighborhoods collectively optimize their distributed
solar generation and shared battery storage.

---

## The Problem: Why Solar Production Efficiency Is Low

Solar energy and photovoltaic cells have low production efficiency due
to a cascade of factors that compound at every stage from sunlight to
usable electricity.

### Environmental & Locational Factors

**Location, climate, and seasonality:** Communities and households in
different locations experience varying amounts of sunlight. Spring,
summer, fall, and winter each produce dramatically different solar
irradiance profiles. Michigan receives roughly 1,400 kWh/m²/year
versus 2,000+ kWh/m²/year in Colorado — a 30% gap from geography alone.

**Changes in cloud cover:** Atmospheric conditions including clouds,
aerosols, and pollutants can reduce electricity output by up to 60%.
Cloud intermittency causes sudden production drops that are particularly
wasteful without local storage.

**Dust, dirt, and debris:** Soiling from dust, pollen, and bird
droppings reduces light absorption and can cut panel efficiency by
5–30%. In extreme cases, debris creates hot spots that permanently
damage cells.

### Physics & Material Losses

**Electron losses across silicon layers:** As electrons travel across
the anodes, cathodes, and silicon layers of photovoltaic cells, energy
is lost to resistive and recombination effects. These charge carrier
collection losses and conduction losses are inherent to the
semiconductor physics of solar cells.

**Silicon type matters:** The use of amorphous or polycrystalline
silicon (as opposed to monocrystalline) significantly reduces
efficiency. Amorphous silicon panels achieve only 6–10% efficiency
versus 20–25% for monocrystalline. Polycrystalline falls in between
at 15–17%. The crystal structure directly affects how freely electrons
can move.

**Temperature and thermal losses:** Panel efficiency drops 0.4–0.5%
per degree Celsius above 25°C. A hot summer day at 40°C can reduce
output by 7–8% from thermal effects alone.

**Optical reflection losses:** Light bouncing off the panel surface
rather than being absorbed. Anti-reflective coatings help but cannot
eliminate this entirely.

**Spectral mismatch:** Solar cells can only convert certain wavelengths
of sunlight. Energy outside the cell's absorption band is wasted as
heat, bounded by the Shockley–Queisser theoretical limit of ~33% for
single-junction cells.

### System-Level Losses

**Inverter conversion losses:** Converting DC to AC power costs 3–5%
efficiency through the inverter.

**Panel degradation over time:** Monocrystalline panels lose 0.3–0.5%
efficiency per year, accumulating ~8% loss over 20 years.

**Shading and partial obstruction:** Trees, buildings, or even one
panel's shadow on another can disproportionately reduce output for an
entire string of panels.

**Wiring and connection losses:** DC cable losses occur whenever
current flows. Bypass diode losses, mismatch between panels, and
junction box resistance all compound.

### The Compounding Problem

These losses compound. A community in Ann Arbor with polycrystalline
panels on a partly cloudy 85°F day with moderate dust might see:

```
70% location factor
× 60% cloud factor
× 95% thermal efficiency
× 95% soiling factor
× 95% inverter efficiency
─────────────────────────────
≈ 34% of rated capacity reaching the household as usable electricity
```

This is why community-level intelligence and shared storage are not
luxuries — they are necessities for making solar viable.

---

## The Solution

SolarHive addresses the compounding efficiency problem at the community
level through three mechanisms:

**Collective intelligence:** Each household is both a producer and
consumer. SolarHive's AI layer sees the whole community's energy state
— who's overproducing, who's underproducing, where storage capacity
exists — and optimizes the collective rather than each home in
isolation.

**Community-based storage:** Shared battery storage (and eventually
fuel cell technologies like solid oxide or salt-based fuel cells)
captures surplus energy that would otherwise be exported at low grid
rates. This compensates for low production efficiency by ensuring no
generated watt is wasted.

**Decentralized live grid:** Energy flows within the community based on
real-time need — a live, breathing grid rather than the dead,
one-directional grid of traditional utility infrastructure. SolarHive
is the brain that orchestrates these flows.

> *SolarHive doesn't change the physics of solar panels.
> It changes how communities USE what those panels produce —
> and that changes everything.*

---

## How Gemma 4 Powers SolarHive

SolarHive leverages three core Gemma 4 capabilities, directly aligned
with what the hackathon demands: **multimodal power** and **native
function calling**.

---

### Feature 1 — Multimodal VQA (Three Modes)

#### VQA Mode 1 — Sky Condition Analysis *(Primary Demo Moment)*

A community member photographs the sky. The user asks:
*"How will this affect our solar production?"*

Gemma 4 analyzes cloud formations, estimates coverage percentage,
combines this with live weather API data via function calling, checks
community battery state, and responds with a grounded, actionable
forecast — multimodal vision and native tool calling working together
in a single agentic turn.

> *"I see scattered cumulus clouds with roughly 40% sky coverage moving
> east. Expect production to dip 30% over the next two hours. Storage
> is healthy at 72%, so no action needed."*

#### VQA Mode 2 — Panel Health Inspection

A homeowner photographs their rooftop panels. Gemma 4 visually
identifies dirt and debris buildup, shading from trees, physical
damage, or suboptimal tilt. It cross-references visual findings with
that home's production data to quantify the efficiency impact.

> *"I can see significant dust on your panels. Your production has been
> 15% below the neighborhood average, which is consistent with what
> I'm seeing."*

#### VQA Mode 3 — Neighborhood Aerial Assessment

Upload a satellite or aerial image of the neighborhood. Gemma 4
identifies which roofs have panels, estimates orientation and potential
capacity, and spots shading issues. Useful for community planning.

> *"Three homes on Oak Street have south-facing roofs with no panels —
> they'd add an estimated 12kW to the community grid."*

---

### Feature 2 — Native Function Calling (Agentic)

SolarHive uses Gemma 4's native tool-use protocol — **not
prompt-engineered function calling.** This is a critical distinction.

Tools are passed via `apply_chat_template(tools=[...])` and the model
autonomously decides which tools to invoke using Gemma 4's dedicated
control tokens (`<|tool_call>` and `<|tool_response>`).

**Four tools Gemma 4 can invoke autonomously:**

| Tool | API | Returns |
|------|-----|---------|
| `get_weather(location)` | OpenWeatherMap | Cloud cover %, temperature, wind, humidity |
| `get_solar_production(clouds_pct)` | NREL PVWatts model | Community kW production estimate |
| `get_battery_state()` | Community BMS | State of charge, capacity, charging status |
| `get_grid_status()` | EIA Open Data | Pricing period (peak/mid/off-peak), rate/kWh |

**The four-stage agentic loop:**

```
Stage 1 — Tool Definition
  Four tools passed to Gemma 4 via apply_chat_template(tools=[...])
  with typed Python signatures and Google-style docstrings.
  Gemma 4's chat template automatically generates the tool schema.
          ↓
Stage 2 — Model Decides
  Gemma 4 analyzes the question, reasons about which data it needs,
  and emits structured <|tool_call> tokens requesting specific tools.
          ↓
Stage 3 — Developer Executes
  Code intercepts each tool_call, executes the real API request
  (OpenWeatherMap, NREL, EIA), feeds results back via tool_responses.
          ↓
Stage 4 — Model Responds
  Gemma 4 reads the tool results and synthesizes a final,
  grounded recommendation based on actual live data.
```

**Why this matters — selective tool reasoning:**

```
"What time does peak pricing start?"
→ Model calls: get_grid_status() only

"Should I run my pool heater now?"
→ Model calls: get_weather() + get_solar_production()
               + get_battery_state() + get_grid_status()
```

The model decides which tools are relevant based on the question —
not blindly fetching everything. This demonstrates genuine agentic
reasoning.

**Example grounded response:**

> *"Battery is at 72%, partly cloudy with 55% production. You have
> headroom — run it now before peak pricing starts at 4pm."*

**VQA + Function Calling in one agentic turn:**

When the user uploads a sky photo alongside their question, Gemma 4
processes the image via its vision encoder, reasons about both the
visual input and the question, then decides which tools to call. This
is multimodal and native function calling working together in a single
agentic turn — the exact pattern the hackathon emphasizes.

```python
# One call: image analysis + live API tool calling + grounded response
result = solarhive_agent(
    question="How will this sky affect our production? Should I charge my EV now?",
    image=sky_photo   # Gemma 4 processes image AND calls tools in one turn
)
# → tool_calls: [get_weather, get_solar_production, get_battery_state, get_grid_status]
# → response: grounded answer citing both visual observation and live API data
```

---

### Feature 3 — Fine-Tuned Domain Expert (Unsloth + Ollama)

**Model:** Gemma 4 E4B fine-tuned on solar energy community Q&A using
Unsloth QLoRA, exported to GGUF, and served locally via Ollama.

> **Note:** Gemma 4 E4B is the text-optimized edge model (4B params,
> fits on T4 16GB in 4-bit). It is purpose-built for local deployment
> on resource-constrained hardware. Multimodal VQA (sky photos, panel
> inspection) is handled by the Gemma 4 26B MoE model in the inference
> notebook.

**Fine-tuning pipeline:**
```
Gemma 4 E4B (base)
    → Unsloth QLoRA (r=16, energy domain Q&A data)
    → SFT on community solar scenarios
    → GGUF export (q4_k_m quantization)
    → Ollama local deployment
```

**Local-first, privacy-first:** Running Gemma 4 via Ollama means
community energy data never leaves the neighborhood. No cloud
dependency, no latency penalty, no data privacy concerns — the AI
runs where the community lives.

---

## Architecture

```
User (dashboard / mobile)
        ↓
React Frontend  ──────────────────────────────────────────
(Vercel, free tier)                                       |
        ↓  REST API                                       |
FastAPI Backend                                           |
        ├── Gemma 4 Agent (Ollama, local-first)           |
        │     ├── VQA Mode 1: Sky analysis                |
        │     ├── VQA Mode 2: Panel inspection            |
        │     ├── VQA Mode 3: Aerial assessment           |
        │     └── Native function calling:                |
        │           ├── get_weather()   → OpenWeatherMap  |
        │           ├── get_solar()     → NREL PVWatts    |
        │           ├── get_battery()   → Community BMS   |
        │           └── get_grid()      → EIA API         |
        └── Simulation Engine                             |
              └── 12-home neighborhood model              |
                  (NREL NSRDB + Pecan Street baselines) ──┘
```

---

## Repository Structure

```
the-gemma4-good-hackathon-solarhive/
├── solarhive_inference.py    # Gemma 4 26B inference: VQA (3 modes) +
│                             # native function calling + agentic loop
├── solarhive_finetune.py     # Unsloth QLoRA fine-tuning: E4B →
│                             # GGUF → Ollama deployment pipeline
├── simulation.py             # 12-home neighborhood simulation engine
│                             # (NREL NSRDB + Pecan Street) [Phase 4]
├── backend.py                # FastAPI REST API backend [Phase 4]
├── frontend/                 # React dashboard [Phase 5]
│   ├── src/
│   │   ├── Dashboard.jsx     # Energy flow visualization
│   │   ├── ChatPanel.jsx     # Gemma 4 chat + image upload
│   │   └── HomeCard.jsx      # Per-home production/consumption card
└── README.md
```

---

## Setup

### Kaggle Notebook (recommended for inference)

1. Open a Kaggle Notebook
2. Settings → Accelerator → **GPU T4 x2**
3. Add-ons → Secrets: add `OWM_API_KEY`, `NREL_API_KEY`, `EIA_API_KEY`
4. Upload `solarhive_inference.py` and run cells sequentially

### API Keys Required (all free tier)

| Service | Portal | Free Tier |
|---------|--------|-----------|
| OpenWeatherMap | [openweathermap.org/api](https://openweathermap.org/api) | 1,000 calls/day |
| NREL PVWatts | [developer.nrel.gov](https://developer.nrel.gov) | Unlimited |
| EIA Open Data | [api.eia.gov](https://www.eia.gov/opendata) | Unlimited |

### Local Deployment via Ollama

```bash
# After running solarhive_finetune.py to generate the GGUF file:

# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./solarhive-gemma4-e4b-q4_k_m.gguf
SYSTEM "You are SolarHive, an AI energy advisor for a community solar microgrid."
EOF

# Create and run the model
ollama create solarhive -f Modelfile
ollama run solarhive "What is our community battery status?"
```

---

## Data Sources

| Source | What It Provides | Cost |
|--------|-----------------|------|
| NREL PVWatts v8 API | Solar production estimates by location, roof specs, weather | Free |
| NREL NSRDB | 30 years hourly solar irradiance data for Ann Arbor, MI | Free |
| NOAA CDO | Historical weather, cloud cover | Free |
| OpenWeatherMap API | Real-time weather, cloud cover, forecasts | Free (1K/day) |
| EIA Open Data API | Grid demand, electricity pricing, peak/off-peak periods | Free |
| Pecan Street (Kaggle) | 10-home real household solar + consumption at 1-min intervals | Free |
| EPA AirNow | Air quality data (affects panel efficiency) | Free |

**Simulation approach:** NREL NSRDB historical data for Ann Arbor, MI
generates realistic production baselines for a simulated 12-home
neighborhood. Real-time OpenWeatherMap data is overlaid during demos.
The community battery is modeled with simple charge/discharge physics.
Pecan Street sample data provides realistic household consumption curves.

In production, synthetic simulation feeds would be replaced by actual
smart meter and IoT data.

---

## Real-World Impact

### The Waste Crisis Is Urgent

Global renewable energy curtailment exceeded 50 TWh in 2024 —
equivalent to the annual electricity consumption of Norway. This
represents approximately 15–20 million tons of CO₂ emissions that
could have been avoided but weren't. Clean energy was generated and
then thrown away because the grid couldn't absorb it.

In California alone, 11.5% of potential solar generation was curtailed
in early 2025. Germany saw solar curtailment surge 97% year-over-year.
The problem is accelerating: by 2030, variable renewables will generate
almost 30% of global electricity — double today's level — and
curtailment will grow with it unless community-level intelligence
intervenes.

At the household level, residential solar systems without storage
typically achieve only 25–40% self-consumption — meaning 60–75% of
what they generate either gets exported at a fraction of the retail
rate or is wasted entirely. Battery storage can increase
self-consumption to 60–90%, and intelligent load shifting (the core
of what SolarHive does) can increase self-consumption by 15–40%
with **zero additional hardware investment.**

### Quantifiable CO₂ Reduction

A 12-home community with 72kW capacity in Ann Arbor generates roughly
90,000 kWh per year. If SolarHive improves self-consumption from 35%
to 60% through intelligent load-shifting and community-level
coordination — conservative, given that storage alone can reach 80%
— that's approximately **22,500 additional kWh** consumed locally
instead of wasted or exported at low value.

Using Michigan's grid emissions factor, that displaces roughly
**16 tons of CO₂ per year — from software alone, no new hardware
required.**

| Scale | CO₂ Displaced Annually |
|-------|------------------------|
| 1 neighborhood (12 homes) | 16 tons |
| 1,000 neighborhoods | 16,000 tons |
| 100,000 neighborhoods | **1.6 million tons** |

The marginal cost of each additional deployment is near zero because
SolarHive is software.

### Business Opportunities

**Homeowners** save money directly: self-consumption is 3–6x more
valuable than grid export in 2025, with feed-in tariffs at 3–8
cents/kWh while retail rates sit at 25–45 cents/kWh. Every kWh
SolarHive redirects from export to self-consumption saves the
homeowner **$0.20–0.35.**

**Community solar developers and HOAs** gain a management platform
for coordinating across households — the operating system for
community solar programs.

**Utilities** benefit from reduced grid strain and behind-the-meter
visibility. Currently, rooftop solar generation is usually unmeasured,
complicating grid reliability and safety. SolarHive provides the data
utilities desperately need.

**Battery storage companies** (Tesla Powerwall, Enphase, etc.) gain
an AI optimization layer that makes their hardware more effective
through smarter charge/discharge cycles.

**Insurance and real estate:** Climate-resilient neighborhoods with
demonstrable energy independence command premium valuations. SolarHive
provides verifiable energy resilience data.

---

## What SolarHive Does NOT Solve

SolarHive is an intelligence and optimization layer. It is important
to be clear about what it does not do:

- **Does not change the fundamental physics of solar cell efficiency.**
  The Shockley–Queisser theoretical limit of ~33% for single-junction
  cells remains. SolarHive cannot make a 20%-efficient panel produce
  at 40%.

- **Does not manufacture batteries or fuel cells.** Community-based
  storage hardware must be sourced, installed, and maintained
  separately.

- **Does not build physical grid infrastructure.** Wiring,
  transformers, and interconnection agreements between households
  require separate engineering and permitting.

- **Does not replace utility-scale grid management.** SolarHive
  operates at the community level (12–200 homes). Regional grid
  balancing remains the responsibility of utilities and grid operators.

What SolarHive does do is make existing infrastructure work
significantly better through information, coordination, and
community-level optimization.

> *It doesn't change the physics of solar panels.
> It changes how communities use what those panels produce —
> and that changes everything.*

---

## Compute Strategy

This project does **not** require expensive GPU cloud services.

| Task | Platform | Cost |
|------|----------|------|
| Prototyping & experimentation | Kaggle Notebooks (T4, 30+ hrs/week free) | $0 |
| Fine-tuning via Unsloth QLoRA | Kaggle Notebooks or Colab Pro | $0–$10/mo |
| Live demo inference | Ollama on local machine (quantized) | $0 |
| Backup / deadline crunch | Google Colab Pro A100 | $10/mo |

**Total estimated compute budget: $0–$30**

---

## Hackathon Submission

| Item | Detail |
|------|--------|
| Competition | [The Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon) |
| Organizers | Google DeepMind × Kaggle |
| Track | Global Resilience — Climate & Green Energy |
| Special Tech | Ollama + Unsloth |
| Deadline | May 18, 2026 at 11:59 PM UTC |
| Kaggle profile | [melricko](https://kaggle.com/melricko) |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built with Gemma 4 · Ann Arbor, Michigan · April–May 2026*
