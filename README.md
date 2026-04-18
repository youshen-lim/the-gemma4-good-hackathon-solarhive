![SolarHive — AI-Powered Community Solar Energy Intelligence](SolarHive_HeaderImage_1280x640_GitHub%20Repo%20Header.png)

# SolarHive
## AI-Powered Community Solar Energy Intelligence

> **The Gemma 4 Good Hackathon** — Google DeepMind x Kaggle
> **Track:** Global Resilience
> **Special Technology Tracks:** Ollama, Unsloth

[![Kaggle](https://img.shields.io/badge/Kaggle-Gemma%204%20Good%20Hackathon-20BEFF?logo=kaggle)](https://kaggle.com/competitions/gemma-4-good-hackathon)
[![Model](https://img.shields.io/badge/Gemma%204-26B%20A4B-4285F4?logo=google)](https://kaggle.com/models/google/gemma-4)
[![Demo](https://img.shields.io/badge/HF%20Space-Live%20Demo-FFD21E?logo=huggingface)](https://huggingface.co/spaces/Truthseeker87/solarhive)
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

## Why This Matters to Me

I chose Ann Arbor, Michigan as SolarHive's prototype location because I live here — after years in New York and Boston. Ann Arbor's low-lying suburban landscape eliminates the shading complications of high-rise cities, making it an ideal testbed for community solar.

My motivation runs deeper than a hackathon:

**The dream of household energy production.** Every home in a community is a potential energy producer and consumer — a clean energy island. Community-based storage using batteries and fuel cells (solid oxide or salt-based) can capture surplus energy to compensate for solar's low production efficiency. This dream was sparked during my undergraduate years at the National University of Singapore, where electives in green energy, atmospheric chemistry, and evolutionary biology taught me the deep connection between photosynthesis, carbon sequestration, and Earth's climate systems.

**Live grids for critical infrastructure.** Decentralized energy generation can supplement existing grids for burstable demands — hospitals, AI-optimized data centers, telecommunications networks, traffic systems, and solar-powered seawater desalination plants like the one announced in Yeosu, South Korea.

**The unsolved side of the energy equation.** In 2024, I visited Schneider Electric's headquarters in Paris to learn from chief executives including former Group CFO Hilary Maxson. While strong private-sector players like Schneider have solved many problems in energy *distribution*, the other side of the equation — energy *production* and source diversity — remains a global challenge. That visit refocused my research on production.

**Singapore's example.** My home country continues to invest actively in energy production diversification, providing real-world case studies for household energy generation and community storage.

**A personal stake in the future.** My wife and I are expecting our firstborn — a girl we are naming Gemma. The coincidence with Google's model name is not lost on me. I need to play my part, however small, to help pave the way for future generations rather than leave our problems to them.

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
output by 6–7.5% from thermal effects alone.

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
with what the hackathon demands: **multimodal power**, **native
function calling**, and **domain-adapted fine-tuning**.

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

> *"The image shows heavy, uniform overcast conditions with thick gray
> cloud cover obscuring the sun completely. This type of overcast sky
> typically reduces solar production to 10-25% of clear-sky capacity.
> At this moment, expect roughly 10-25 kW community output from your
> 72 kW array. This is a good time to conserve battery for evening
> peak hours and defer non-essential loads."*

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
| `get_weather(location)` | OpenWeatherMap | Temperature, clouds %, wind, humidity, sunrise/sunset |
| `get_solar_production(clouds_pct, temp_f)` | Open-Meteo GHI | Production kW, efficiency %, GHI W/m², temp derating |
| `get_battery_state()` | Community BMS | State of charge, capacity, charging status |
| `get_grid_status()` | EIA Open Data | Pricing period, rate/kWh, renewable %, CO2 intensity |

**The four-stage agentic loop:**

```
Stage 1 — Tool Definition
  Four tools passed to Gemma 4 via apply_chat_template(tools=[...])
  with typed Python signatures and Google-style docstrings.
  Gemma 4's chat template automatically generates the tool schema.
          ↓
Stage 2 — Model Decides
  Gemma 4 analyzes the question, reasons about which data it needs,
  and emits structured tool_call tokens requesting specific tools.
          ↓
Stage 3 — Developer Executes
  Code intercepts each tool_call via regex parsing (call:fn_name{args}),
  executes the real API request (OpenWeatherMap, Open-Meteo, EIA),
  feeds results back as {"role": "tool", "name": ..., "content": ...}
  messages, and loops up to 3 rounds.
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

### Feature 3 — Dual Fine-Tuned Domain Expert (Unsloth + Ollama)

Two models fine-tuned on 1,029 community solar energy examples using
Unsloth LoRA, targeting both cloud and edge deployment:

| Model | Role | Training | Export |
|-------|------|----------|--------|
| Gemma 4 26B A4B (MoE) | Cloud inference + VQA demo | LoRA r=16, BF16, 4,393s | LoRA adapters |
| Gemma 4 E4B (8B) | Edge deployment via Ollama | LoRA r=16, BF16, 282s | LoRA → Ollama |

**Training data:** 929 Q&A across 9 energy domains + 100 tool-calling
examples (50 with tool invocations, 50 without — teaching the model
when NOT to call tools). Two complementary sources:

- **413 hand-crafted static examples** spanning 15+ US cities, covering
  sky conditions, battery management, panel health, consumption
  optimization, community/grid strategy, emergency resilience, seasonal
  planning, multi-step reasoning, and alternative storage
- **516 API-grounded examples** generated from live Open-Meteo, PVWatts,
  OpenWeatherMap, and EIA data for Ann Arbor, MI and San Mateo, CA —
  including hourly production scenarios, PVWatts cross-validation,
  grid mix analysis, weather impact, and battery/grid strategy

**Shared hyperparameters:** LoRA rank=16, alpha=16, dropout=0,
target=all-linear, lr=2e-4, optimizer=adamw_8bit, warmup=5 steps,
max_seq_len=2048, precision=BF16, seed=3407. Batch size auto-tuned
by free GPU VRAM at runtime.

**Training results (Google Colab G4 VM, RTX PRO 6000, BF16):**

| Model | Converged Loss | Trainable Params | Isolated Benchmark | Production Benchmark (agentic) | Time |
|-------|---------------|-----------------|-------------------|-------------------------------|------|
| Gemma 4 26B A4B | **0.675** | 505.4M / 26.3B (1.92%) | 6/8 (5/5 Q&A, 1/3 tool) | **8/8** (5/5 Q&A, 3/3 tool) | 4,393s |
| Gemma 4 E4B | **0.952** | 41.2M / 8.0B (0.51%) | 7/8 (5/5 Q&A, 2/3 tool) | — | 282s |

*Isolated benchmarks run without tool schemas. Production benchmarks run in the full agentic loop with tool definitions passed via `apply_chat_template(tools=[...])`. The 26B A4B reliably calls tools when given the function signatures it was trained on.*

**Local-first, privacy-first:** Running Gemma 4 via Ollama means
community energy data never leaves the neighborhood. No cloud
dependency, no latency penalty, no data privacy concerns — the AI
runs where the community lives.

---

## Architecture

```
Three Colab Notebooks — the complete pipeline:

solarhive_datagen.py     Data generation: 4 live APIs → 551 training examples
        ↓                + 12 diagnostic charts
solarhive_finetune.py    Dual LoRA fine-tuning via Unsloth
        ↓                E4B (282s) + 26B A4B (4,393s)
solarhive_inference.py   Live demo: fine-tuned 26B A4B + 4 API tools
                         + 3 VQA modes + agentic loop + benchmarks

Hardware: Google Colab G4 VM — NVIDIA RTX PRO 6000 Blackwell
          Server Edition (96 GB GDDR7), BF16 precision
Platform: Google Colab Pro
```

**Agentic loop (Cell 4):**
1. System prompt repeated twice — improves instruction following in causal
   LLMs ([Leviathan et al., 2024, Google Research](https://arxiv.org/abs/2512.14982))
2. Tools defined as Python functions with typed signatures → auto-schema
3. Two-step tokenization (critical for transformers 5.5.x compatibility)
4. Model emits `call:fn_name{args}` → parsed via regex
5. Real APIs execute, results fed back as tool messages
6. Loop up to 3 rounds → final grounded response

**LoRA loading (Cell 2b):**
Fine-tuned adapters loaded via Unsloth's `FastVisionModel` — standard
PEFT cannot handle Gemma 4's `Gemma4ClippableLinear` and MoE expert
layers.

---

## Live Demo Output (Inference Run 5, April 15 2026)

Five interactive queries against the fine-tuned Gemma 4 26B A4B with LoRA
adapters loaded via Unsloth. Real sky photos from Ann Arbor uploaded during
the session. All tool calls hit live APIs (OpenWeatherMap, Open-Meteo, EIA).

**Query 1 (VQA Mode 1: sky photo uploaded)**
> **Q:** "How is current cloud cover affecting potential household solar production?"
>
> **Image:** `annarbor_sky_5.jpeg` (heavy overcast)
>
> **Tools called:** none (direct visual answer)
>
> **A:** "The image shows heavy, uniform overcast conditions with thick gray
> cloud cover obscuring the sun completely. This type of overcast sky
> typically reduces solar production to 10-25% of clear-sky capacity.
> At this moment, expect roughly 10-25 kW community output from your
> 60 kW array.* This is a good time to conserve battery for evening
> peak hours and defer non-essential loads."

*\*Known hallucination: the model says "60 kW" instead of the correct
72 kW community capacity. This is a pre-fine-tune base model tendency
that persists in early LoRA runs. The model correctly uses 72 kW in
other responses (see Query 2). Addressing this through additional
fine-tuning examples is planned for the next training run.*

**Query 2 (tool calling: solar production)**
> **Q:** "What was estimated solar production at 9.00am Eastern time?"
>
> **Tools called:** `get_solar_production`
>
> **A:** "At 9:00am in Ann Arbor, MI, the community produced 40.4 kW out
> of 72 kW capacity (56.1% efficiency). GHI was 660.0 W/m² with 10%
> cloud cover and no temperature derating at 65°F. This is a solid
> early morning output, production will ramp up toward the noon peak."

**Query 3 (VQA + energy storage planning: sky photo uploaded)**
> **Q:** "Referring to the cloud cover in attached image, how should I plan
> my household solar production and energy storage (e.g., battery or
> fuel cell) for the next two hours?"
>
> **Image:** `annarbor_sky_03.jpeg` (broken cumulus)
>
> **Tools called:** none (direct visual answer)
>
> **A:** "The image shows heavy, broken cumulus and stratocumulus clouds
> covering approximately 80-90% of the sky. While there are some brighter
> patches suggesting intermittent direct sunlight, the overall irradiance
> is significantly reduced. Production will be highly variable, cycling
> between 15-30% of capacity during cloud gaps and dropping to 5-10%
> under the thicker gray patches. For the next two hours: expect
> inconsistent production with no sustained peak output. If your battery
> is below 50%, prioritize essential loads only and avoid running heavy
> appliances during the darker cloud passages. If you have a fuel cell,
> it may be worth running at a low steady rate to supplement the
> intermittent solar."

**Query 4 (tool calling: grid status)**
> **Q:** "What is current grid capacity now? And how has it changed over the
> past two hours?"
>
> **Tools called:** `get_grid_status`
>
> **A:** "Peak pricing at $0.28/kWh with 30.3% renewable grid mix. CO2
> intensity is 381.5 kg/MWh. Peak rates typically run 2-7pm, prioritize
> battery discharge and minimize grid imports."

**Query 5 (tool calling: weather)**
> **Q:** "What is the weather forecast going to be like this week in Ann
> Arbor and how should we plan for it?"
>
> **Tools called:** `get_weather`
>
> **A:** "Current conditions in Ann Arbor: 67°F with 100% cloud cover and
> mist. Production is likely minimal at this moment. With 100% cloud
> cover, expect significantly reduced solar output, run only essential
> loads and preserve battery for evening peak. If this overcast pattern
> persists through the week, consider pre-charging the battery from the
> grid during off-peak hours to ensure coverage."

---

## Community Model

**Location:** Ann Arbor, Michigan (42.2808°N, 83.7430°W)
**Community:** 12 homes, 72 kW total panel capacity, 100 kWh shared battery
**Grid region:** MISO (Midcontinent Independent System Operator)

---

## Repository Structure

```
the-gemma4-good-hackathon-solarhive/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── data_principles.md           # Data methodology and training data documentation
├── datagen_charts/              # 12 diagnostic charts from data generation pipeline
│   ├── chart_01.png … chart_12.png
├── solarhive_inference.py       # Gemma 4 26B A4B inference: VQA (3 modes) +
│                                # native function calling + agentic loop + benchmarks
├── solarhive_inference.ipynb    # Jupyter notebook version
├── solarhive_finetune.py        # Unsloth LoRA dual fine-tuning:
│                                # E4B + 26B A4B → LoRA adapters
├── solarhive_finetune.ipynb     # Jupyter notebook version
├── solarhive_datagen.py         # Data generation: 4 live APIs → training examples
│                                # + 12 diagnostic charts
└── solarhive_datagen.ipynb      # Jupyter notebook version
```

---

## Data Sources

| Source | What It Provides | Access | Cost |
|--------|-----------------|--------|------|
| Open-Meteo | GHI solar irradiance (W/m²), historical hourly | api.open-meteo.com — no API key | Free |
| OpenWeatherMap | Real-time temperature, wind, humidity, cloud cover, sunrise/sunset | openweathermap.org/api — free API key | Free |
| EIA Open Data v2 | Grid fuel mix, renewable %, CO2 intensity | api.eia.gov — free API key | Free |
| NREL PVWatts v8 | Solar production estimates for cross-validation | developer.nrel.gov — free API key | Free |

**Data principles:** All numeric claims trace back to real API responses
— no synthetic or hallucinated numbers. Full methodology documented in
[`data_principles.md`](data_principles.md).

---

## Data Pipeline Diagnostics (12 Charts from `solarhive_datagen.py`)

All charts generated automatically from live API data during training
data generation. These visualizations validate data quality, reveal
geographic patterns, and cross-validate between independent sources.

### Solar Irradiance & Production

| | |
|:---:|:---:|
| ![GHI Distribution](datagen_charts/chart_01.png) | ![Hourly Production](datagen_charts/chart_02.png) |
| **Chart 1:** GHI distribution for Ann Arbor (median 265 W/m²) vs San Mateo (median 364 W/m²) — Michigan receives ~27% less solar irradiance | **Chart 2:** Average hourly production curve (mean ± 1 std). Peak at 1-2pm, Ann Arbor peaks higher but with wider variance |
| ![Production Heatmap](datagen_charts/chart_03.png) | ![Temperature Derating](datagen_charts/chart_04.png) |
| **Chart 3:** Month × hour production heatmaps. Ann Arbor peaks June-July at 45+ kW midday. San Mateo has a broader, flatter production season | **Chart 4:** Temperature derating factor — flat at 1.0 below 77°F, then linear decline (0.4%/°F). Validates the derating formula in `get_solar_production()` |

### Environmental Correlations

| | |
|:---:|:---:|
| ![Correlation Matrix](datagen_charts/chart_05.png) | ![Cloud Cover by Season](datagen_charts/chart_06.png) |
| **Chart 5:** Feature correlation matrix. GHI→production r=0.97 (near-perfect). Humidity→GHI r=−0.57 (clouds trap moisture). Cloud cover weakly anti-correlated with GHI (r=−0.22) because GHI captures direct + diffuse radiation | **Chart 6:** Cloud cover distribution by season and location. Ann Arbor consistently cloudier than San Mateo across all seasons |
| ![Seasonal Production](datagen_charts/chart_07.png) | ![GHI vs Production](datagen_charts/chart_08.png) |
| **Chart 7:** Daytime (7am-6pm) production by season. Summer median ~33 kW (Ann Arbor) vs ~26 kW (San Mateo). Winter drops to ~12 kW for both | **Chart 8:** GHI vs production scatter, colored by cloud cover. Two distinct bands: clear-sky (red, tight linear) and cloudy (blue, scattered). Demonstrates the physics of diffuse vs direct radiation |

### Cross-Validation & Grid Analysis

| | |
|:---:|:---:|
| ![PVWatts Cross-Validation](datagen_charts/chart_09.png) | ![OWM Conditions](datagen_charts/chart_10.png) |
| **Chart 9:** Monthly production — Open-Meteo vs NREL PVWatts. Strong seasonal agreement validates our GHI-based formula against NREL's industry-standard model | **Chart 10:** OWM current conditions snapshot at data generation time — temperature, clouds, wind, humidity for both locations |
| ![Grid Fuel Mix](datagen_charts/chart_11.png) | ![Renewable & CO2](datagen_charts/chart_12.png) |
| **Chart 11:** Average fuel mix — MISO (33.5% natural gas, 23.4% wind, 18.8% coal) vs CAISO (35.8% solar, 20.6% wind). CAISO's grid is dramatically cleaner | **Chart 12:** Renewable % and CO2 intensity over one week. CISO hits 100% renewable during midday solar peaks; MISO ranges 20-50%. CO2 intensity inversely tracks renewable share |

---

## Setup

### Google Colab (recommended)

1. Open `solarhive_inference.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (A100 or RTX PRO 6000 recommended)
3. Secrets: add `OWM_API_KEY`, `EIA_API_KEY`
4. Mount Google Drive (for LoRA adapter cache)
5. Run cells sequentially

> **Note:** Gemma 4 26B A4B requires ~48 GB VRAM in BF16 or ~16 GB in
> 4-bit NF4. T4 x2 (32 GB) cannot run this model — BitsAndBytes NF4
> is incompatible with CPU offloading.

### API Keys Required (all free tier)

| Service | Portal | Free Tier |
|---------|--------|-----------|
| OpenWeatherMap | [openweathermap.org/api](https://openweathermap.org/api) | 1,000 calls/day |
| EIA Open Data | [api.eia.gov](https://www.eia.gov/opendata) | Unlimited |
| NREL PVWatts | [developer.nrel.gov](https://developer.nrel.gov) | Unlimited |

### Local Deployment via Ollama

```bash
# After fine-tuning E4B via solarhive_finetune.py:
# Option 1: Import safetensors directly (GGUF export has known mmproj issue)
ollama create solarhive -f Modelfile

# Option 2: Pull from HuggingFace (after upload)
# See HuggingFace model card for instructions
```

**Local-first, privacy-first:** Running via Ollama means community
energy data never leaves the neighborhood — no cloud dependency, no
latency penalty, no data privacy concerns.

---

## Real-World Impact

### The Waste Crisis Is Urgent

Global renewable energy curtailment exceeded 50 TWh in 2024 —
equivalent to the annual electricity consumption of Norway. This
represents approximately 15–20 million tons of CO2 emissions that
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

### Quantifiable CO2 Reduction

A 12-home community with 72kW capacity in Ann Arbor generates roughly
90,000 kWh per year. If SolarHive improves self-consumption from 35%
to 60% through intelligent load-shifting and community-level
coordination — conservative, given that storage alone can reach 80%
— that's approximately **22,500 additional kWh** consumed locally
instead of wasted or exported at low value.

Using Michigan's grid emissions factor, that displaces roughly
**16 tons of CO2 per year — from software alone, no new hardware
required.**

| Scale | CO2 Displaced Annually |
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

| Task | Platform | GPU |
|------|----------|-----|
| Data generation | Google Colab Pro | RTX PRO 6000 (96 GB) |
| Fine-tuning (dual LoRA) | Google Colab Pro | RTX PRO 6000 (96 GB), BF16 |
| Inference demo | Google Colab Pro | RTX PRO 6000 (96 GB), BF16 |
| Edge deployment | Ollama on laptop | CPU (E2B, 5.1B params) |

---

## Models & Resources

| Resource | Link | Purpose |
|----------|------|---------|
| **Live Demo** | [HF Space](https://huggingface.co/spaces/Truthseeker87/solarhive) | Interactive Gradio demo (ZeroGPU) |
| **26B A4B Merged** | [solarhive-26b-a4b-merged](https://huggingface.co/Truthseeker87/solarhive-26b-a4b-merged) | Production inference — no Unsloth needed |
| **26B A4B LoRA** | [solarhive-26b-a4b-lora](https://huggingface.co/Truthseeker87/solarhive-26b-a4b-lora) | LoRA adapters for further fine-tuning |
| **E4B Ollama** | [solarhive-e4b-ollama](https://huggingface.co/Truthseeker87/solarhive-e4b-ollama) | Edge deployment via Ollama |
| **Dataset** | [solarhive-community-solar-1k](https://huggingface.co/datasets/Truthseeker87/solarhive-community-solar-1k) | 1,029 training examples |

---

## Hackathon Submission

| Item | Detail |
|------|--------|
| Competition | [The Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon) |
| Organizers | Google DeepMind x Kaggle |
| Track | Global Resilience |
| Special Tech | Ollama + Unsloth |
| Deadline | May 18, 2026 at 11:59 PM UTC |
| Kaggle profile | [melricko](https://kaggle.com/melricko) |
| GitHub | [youshen-lim/the-gemma4-good-hackathon-solarhive](https://github.com/youshen-lim/the-gemma4-good-hackathon-solarhive) |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built with Gemma 4 in Ann Arbor, Michigan — April 2026*

*Gemma is a trademark of Google LLC.*
