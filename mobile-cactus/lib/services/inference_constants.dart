// Shared inference constants for the SolarHive Cactus on-device LLM.
//
// Sampling parameters mirror the Kaggle-recommended Gemma 4 defaults used
// in `solarhive_inference.py` (the cloud / Colab inference pipeline). Both
// the cloud agentic loop and the on-device single-prompt path are driving
// the same fine-tuned model family, so the sampling regime stays aligned
// across runtimes. Drift here would make the on-device behaviour silently
// diverge from the cloud benchmarks the project publishes.
//
// The system prompt is intentionally a NARROWER variant of the cloud
// `SYSTEM_PROMPT` in `solarhive_inference.py` (line ~772). The cloud
// prompt instructs the model to call tools for real-time data; the
// on-device tier has no tools wired (by design — the routing strategy
// escalates real-time queries to the cloud HF Space via the 🛰️ emoji).
// The on-device prompt therefore drops the "call available tools" sentence
// and reframes data references from "actual data" to "reasonable
// assumptions". Identity / community facts / response-length guidance
// stay aligned with the cloud prompt.
//
// PROMPT REPETITION:
// Following solarhive_inference.py line 772, the body is concatenated to
// itself with a blank-line separator and sent as a SINGLE system message.
// Doubling lets every token in the prompt attend to every other prompt
// token, improving instruction-following without a latency hit. Reference:
// Leviathan, Kalman & Matias (2024), "Repeat to Improve Non-Reasoning
// LLMs", Google Research. https://arxiv.org/abs/2512.14982
//
// LATEX HANDLING:
// One short directive ("answer in plain prose; no LaTeX/markdown") is a
// defensive layer. The actual rendering safety net lives in
// `cactus_engine.dart`'s `_stripLatex` post-processor, which mechanically
// strips `\text{...}`, `$...$`, `$$...$$`, `\(...\)`, `\[...\]` from the
// model's output before it hits the chat UI.
//
// Drift detectors live in `test/services/inference_constants_test.dart`.

// =======================================================================
// Sampling parameters -- aligned with Kaggle-recommended Gemma 4 defaults
// (see `solarhive_inference.py` lines 288, 815, 1112, 1146, 1282)
// =======================================================================

/// Sampling temperature. Kaggle Gemma 4 default = 1.0.
const double kKaggleTemperature = 1.0;

/// Top-p (nucleus) sampling threshold. Kaggle Gemma 4 default = 0.95.
const double kKaggleTopP = 0.95;

/// Top-k sampling cutoff. Kaggle Gemma 4 default = 64.
const int kKaggleTopK = 64;

/// Max generation tokens for an on-device smoke-test prompt. The cloud
/// agentic loop uses 1024; on-device defaults are tighter to keep first-
/// token-to-completion latency under a phone-friendly budget. The chat
/// screen passes a 256-token cap by default, sized to fit comfortably
/// within the measured ~3.6 tok/s decode throughput on Snapdragon 865
/// at contextSize=512.
const int kDefaultMaxNewTokens = 512;

// =======================================================================
// System prompt body -- doubled at the bottom per "Repeat to Improve"
// =======================================================================

/// Single-occurrence body. Kept terse — the verbose unit-formatting
/// example block was dropped in favour of a one-line directive backed by
/// `_stripLatex` in cactus_engine.dart, since the post-processor handles
/// LaTeX-flavoured output more reliably than a long prompt directive.
/// Token count of the body alone: ~135 tokens; the final
/// `kSolarHiveSystemPrompt` doubles to ~275 tokens, well under any
/// reasonable contextSize cap.
const String _kUnifiedSystemBody =
    'You are SolarHive, an AI energy advisor for a community of 12 homes '
    'with rooftop solar and shared battery storage in Ann Arbor, Michigan. '
    'Provide specific, data-grounded advice on solar production, energy '
    'storage, grid coordination, and panel maintenance. For general '
    'guidance, scenario planning, or domain knowledge, answer directly. '
    'Be specific, reference reasonable assumptions, and keep responses '
    'concise (3-5 sentences). Answer in plain prose; do not use LaTeX or '
    'markdown.';

/// SolarHive system prompt for the on-device tier. Doubled body matches
/// `solarhive_inference.py`'s `SYSTEM_PROMPT = body + "\n\n" + body`
/// pattern verbatim, so on-device behaviour stays comparable to the
/// cloud agentic-loop benchmarks the project publishes.
const String kSolarHiveSystemPrompt =
    '$_kUnifiedSystemBody\n\n$_kUnifiedSystemBody';
