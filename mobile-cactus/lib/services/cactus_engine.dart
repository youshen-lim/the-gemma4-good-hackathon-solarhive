// Cactus engine wrapper for the SolarHive on-device LLM.
//
// Loads the SolarHive INT4 multimodal artifact via the docs-canonical FFI
// surface from `lib/cactus.dart` — a verbatim copy of `flutter/cactus.dart`
// at cactus-compute/cactus commit d917981f. The matching `libcactus.so`
// binary (built from the same commit) is vendored at
// `android/app/src/main/jniLibs/arm64-v8a/libcactus.so`. Pinning the
// converter, the binary, and the Dart bindings to a single commit avoids
// the C ABI / file-header drift that occurs when the converter and the
// runtime binary disagree on the artifact's on-disk layout (the pub.dev
// `cactus 1.3.0` package is several months older than the artifact's
// converter and exhibits this drift).
//
// Diagnostic infrastructure (see `diagnostics.dart`):
//   - installCactusLogSink hooks cactus_log_set_callback so every C engine
//     log line is captured into ${appDocs}/solarhive_diag.log + an in-memory
//     ring buffer + a broadcast Stream.
//   - auditArtifact runs before cactus_init to enumerate per-file sizes,
//     flag zero-byte / suspect-truncated / missing-metadata files. Issues
//     are written to the same diagnostic log so post-mortem analysis can
//     correlate the audit with the engine's own log output.
//
// Cloud handoff is implemented independently via direct POST to the
// SolarHive HF Space (see `chat_screen.dart`'s 🛰️ button wiring). The
// roll-own POST is the only cloud routing path; hybrid completion in the
// pub.dev OO API falls back to OpenRouter, which does not host the
// SolarHive 26B A4B fine-tune.

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import '../cactus.dart' as cactus;
import '../models/chat_message.dart';

import 'diagnostics.dart';
import 'inference_constants.dart';
import 'llm_engine.dart';

/// Default context size for `cactus_init`. Set to 1024 because on-device
/// throughput is highly sensitive to contextSize on this engine:
/// contextSize=2048 caused a ~6× throughput regression vs 512
/// (cactus@main pre-allocates KV cache + iterates per-step attention
/// over the full contextSize regardless of how much is actually in use).
/// Typical active usage is ~540 tokens (doubled-system 270 + user 20 +
/// generated 256), so 1024 leaves a comfortable 2× headroom for
/// chat-length prompts while halving per-step attention work vs 2048.
/// Trade-off: long multi-turn conversations exceeding 1024 tokens of
/// rolling history would truncate; mitigated by the on-device tier
/// being narrow (single-prompt UX), with longer-context queries
/// escalated via the 🛰️ cloud handoff.
const int kDefaultContextSize = 1024;

/// Parsed token-count + extracted response text from a single
/// `cactus_complete` call. The Cactus C engine wraps generation output in
/// a llama.cpp-style JSON envelope; we surface the fields the cycle-time
/// instrumentation needs so chat_screen can compute `respTok / generate_s`
/// without re-parsing the envelope itself.
class GenerationStats {
  /// Output tokens emitted by the model. Sourced from `decode_tokens` in
  /// the Cactus envelope (the canonical Run 13 schema discovered via
  /// `[Gen] envelopeKeys=[…]` forensic dump on the validation Android
  /// device). Falls
  /// back to llama.cpp-canonical `tokens_predicted` / `predicted_n` for
  /// future-proofing if Cactus realigns its envelope upstream. Null
  /// when the envelope was non-JSON or all candidate fields absent.
  final int? responseTokens;

  /// Prompt tokens consumed. Sourced from `prefill_tokens` in the Cactus
  /// envelope (Run 13 schema). Falls back to `tokens_evaluated` /
  /// `prompt_n` for cross-runtime tolerance. Null when absent.
  final int? promptTokens;

  /// Engine-reported decode throughput in tokens/sec, taken verbatim
  /// from the Cactus envelope's `decode_tps` field. Preferred over a
  /// Dart-side wall-clock derivation because it excludes prefill time
  /// from the denominator (cleaner number for steady-state decode
  /// throughput). Null when absent.
  final double? engineDecodeTps;

  /// Engine-reported prefill throughput in tokens/sec from the
  /// envelope's `prefill_tps` field. Useful for diagnosing
  /// prefill-vs-decode performance separately. Null when absent.
  final double? enginePrefillTps;

  /// Time-to-first-token in milliseconds from the envelope's
  /// `time_to_first_token_ms` field. Separates prompt-prefill latency
  /// from the per-token decode loop. Null when absent.
  final int? timeToFirstTokenMs;

  /// Engine's internal total wall-clock measurement in milliseconds
  /// from `total_time_ms`. Comparing against the Dart-side
  /// engineEntry→engineExit delta reveals FFI marshalling + JSON
  /// encode/decode overhead. Null when absent.
  final int? engineTotalMs;

  const GenerationStats({
    this.responseTokens,
    this.promptTokens,
    this.engineDecodeTps,
    this.enginePrefillTps,
    this.timeToFirstTokenMs,
    this.engineTotalMs,
  });

  /// True when both token counts were parsed — the minimum needed for
  /// chat UI / log readers to render "decode tok/s" annotations.
  bool get isComplete => responseTokens != null && promptTokens != null;
}

/// Wrapper around the cactus@main FFI bindings, scoped to the SolarHive
/// INT4 artifact directory.
class CactusEngine implements LlmEngine {
  final Directory artifactDir;
  cactus.CactusModelT? _handle;
  Object? _loadError;
  ArtifactAuditReport? _lastAudit;
  GenerationStats? _lastGenerationStats;

  CactusEngine({required this.artifactDir});

  @override
  Object? get loadError => _loadError;

  /// The most recent artifact audit, or null if `ensureLoaded()` hasn't
  /// run yet. Surfaced to the chat UI so the failure card can show
  /// `report.totalGiBSummary` and the issue list inline.
  ArtifactAuditReport? get lastAudit => _lastAudit;

  /// Token counts parsed from the most recent `cactus_complete` call.
  /// Null until at least one `generate()` succeeds. The cycle-time
  /// instrumentation reads this in `chat_screen._send()` to enrich the
  /// `[CYCLE]` summary line with prompt/response token counts and a
  /// computed decode tok/s.
  GenerationStats? get lastGenerationStats => _lastGenerationStats;

  /// Idempotent. The first call:
  ///   1. Installs the C engine log sink (captures cactus internal logs
  ///      to ${appDocs}/solarhive_diag.log).
  ///   2. Audits the artifact directory (per-file sizes + anomaly flags).
  ///   3. Calls `cactus.cactusInit(path, null, false)` per cactus@main's
  ///      docs-canonical signature. If the C engine returns null, the
  ///      wrapper surfaces the engine's `cactus_get_last_error` plus the
  ///      audit summary so the chat UI can render both inline.
  @override
  Future<void> ensureLoaded() async {
    if (_handle != null) return;

    // Stage 1: install log sink BEFORE the C engine starts emitting,
    // otherwise we miss the early init lines.
    await installCactusLogSink();
    await writeDartDiagnostic('ensureLoaded() invoked for ${artifactDir.path}');

    // Stage 2: audit the artifact.
    final audit = await auditArtifact(artifactDir);
    _lastAudit = audit;
    await writeDartDiagnostic('Artifact audit: ${audit.totalGiBSummary}');
    for (final issue in audit.issues) {
      await writeDartDiagnostic('Audit issue: $issue');
    }

    // Stage 3: hand off to the C engine.
    try {
      // cactus@main signature: cactusInit(modelPath, corpusDir, cacheIndex)
      // - modelPath: absolute path to the artifact directory
      // - corpusDir: optional RAG corpus path; null (no RAG)
      // - cacheIndex: whether to cache the index for RAG; false (no RAG)
      _handle = cactus.cactusInit(artifactDir.path, null, false);
      _loadError = null;
      await writeDartDiagnostic('cactus_init succeeded; model loaded');
    } catch (e) {
      // cactus.dart's cactusInit throws with the engine's last_error
      // string in the message. Surface verbatim plus the audit summary so
      // the user sees both the engine reason and the cache state.
      final issuesSummary = audit.issues.isEmpty
          ? 'audit clean'
          : '${audit.issues.length} audit issue(s): ${audit.issues.take(3).join("; ")}'
              '${audit.issues.length > 3 ? " (+${audit.issues.length - 3} more)" : ""}';
      final wrapped = Exception(
        'cactus_init failed for path "${artifactDir.path}" '
        '(contextSize=$kDefaultContextSize). $e\n'
        'Cache audit: ${audit.totalGiBSummary} | $issuesSummary',
      );
      _loadError = wrapped;
      _handle = null;
      await writeDartDiagnostic('cactus_init FAILED: $e');
      throw wrapped;
    }
  }

  /// Multi-turn generation via `cactus.cactusComplete`. The caller passes
  /// the conversation history (alternating user / assistant turns ending
  /// on a fresh user turn); this method prepends the on-device SolarHive
  /// `kSolarHiveSystemPrompt` so the system anchor stays in place
  /// regardless of UI state. Sampling options are pinned to the
  /// Kaggle-recommended Gemma 4 defaults from `inference_constants.dart`.
  /// Tools and PCM audio inputs stay null — the on-device tier is
  /// intentionally narrow.
  @override
  Future<String> generate({
    required List<ChatMessage> messages,
    int maxNewTokens = kDefaultMaxNewTokens,
  }) async {
    await ensureLoaded();
    final handle = _handle;
    if (handle == null) {
      throw StateError('CactusEngine: model not loaded');
    }
    final messagesJson = jsonEncode([
      {'role': 'system', 'content': kSolarHiveSystemPrompt},
      ...messages.map((m) => m.toJsonShape()),
    ]);
    final optionsJson = jsonEncode({
      'max_tokens': maxNewTokens,
      'temperature': kKaggleTemperature,
      'top_p': kKaggleTopP,
      'top_k': kKaggleTopK,
    });

    // Capture a pre-generation memory baseline + start periodic sampling.
    // The sampler writes RSS/HWM/sysFree to the diagnostic log every
    // second so a SIGKILL by lmkd or a SIGABRT mid-generation leaves a
    // trajectory we can pull post-crash.
    final preSample = await readMemorySample();
    if (preSample != null) {
      await writeDartDiagnostic('Memory pre-generate: ${preSample.formatted}');
    }
    startMemorySampler();

    final totalContentChars = messages.fold<int>(0, (a, m) => a + m.content.length);
    await writeDartDiagnostic(
        'cactus_complete: turns=${messages.length}, '
        'totalContentChars=$totalContentChars, maxNewTokens=$maxNewTokens');

    try {
      final resultJson = cactus.cactusComplete(
        handle,
        messagesJson,
        optionsJson,
        null, // toolsJson
        null, // onToken streaming callback
      );
      stopMemorySampler();
      final postSample = await readMemorySample();
      if (postSample != null) {
        await writeDartDiagnostic(
            'Memory post-generate: ${postSample.formatted}');
      }
      final parsed = _parseEnvelope(resultJson);
      _lastGenerationStats = parsed.stats;
      final s = parsed.stats;
      await writeDartDiagnostic(
          '[Gen] promptTok=${s.promptTokens ?? "?"} '
          'respTok=${s.responseTokens ?? "?"} '
          'engineDecodeTps=${s.engineDecodeTps?.toStringAsFixed(2) ?? "?"} '
          'enginePrefillTps=${s.enginePrefillTps?.toStringAsFixed(2) ?? "?"} '
          'ttftMs=${s.timeToFirstTokenMs ?? "?"} '
          'engineTotalMs=${s.engineTotalMs ?? "?"} '
          'envelopeJsonBytes=${resultJson.length} '
          'envelopeKeys=${parsed.envelopeKeysSummary}');
      return parsed.text;
    } finally {
      // If cactus_complete throws (Dart-side) the finally still cancels
      // the sampler; if the C engine SIGABRTs the timer is moot since
      // the process is gone, but the file-based log already captured
      // the trajectory.
      stopMemorySampler();
    }
  }

  /// Cleanup. Call from `dispose()` of the parent widget. Releases the FFI
  /// handle via `cactus.cactusDestroy`.
  @override
  Future<void> unload() async {
    final handle = _handle;
    if (handle == null) return;
    cactus.cactusDestroy(handle);
    _handle = null;
  }
}

/// Parsed result of `cactus_complete`'s JSON envelope: the cleaned
/// assistant text plus the token-count stats used by the cycle-time
/// instrumentation. Kept private — chat_screen reads the stats from the
/// engine via `lastGenerationStats`.
class _ParsedEnvelope {
  final String text;
  final GenerationStats stats;

  /// Forensic-grade summary of the envelope's top-level structure: either
  /// `<non-json>` when the response wasn't parseable, or a sorted list of
  /// the top-level keys we saw. Written into the `[Gen]` diag line so the
  /// log captures the actual schema cactus@<pinned-commit> emits — once a
  /// real on-device run lands a missing token-count parse, the next run's
  /// log row shows the exact key names so we can extend the parser
  /// without needing to ship a debug-only envelope dump separately.
  final String envelopeKeysSummary;

  const _ParsedEnvelope({
    required this.text,
    required this.stats,
    required this.envelopeKeysSummary,
  });
}

/// Reads an integer field from a JSON map under any of the supplied key
/// names, returning the first hit (in order). Tolerates ints encoded as
/// either Dart `int` or `num` (some JSON serializers emit `42.0` instead
/// of `42`); doubles round-trip to int via `.toInt()` so we don't reject
/// equivalent representations.
int? _readIntField(Map<String, dynamic> m, List<String> names) {
  for (final n in names) {
    final v = m[n];
    if (v is int) return v;
    if (v is num) return v.toInt();
  }
  return null;
}

/// Reads a double field from a JSON map under any of the supplied key
/// names. Tolerates either `int` or `num` JSON encoding — `42` and `42.0`
/// both round-trip to the same Dart `double` so cross-version Cactus
/// envelopes that flip an integer-valued `decode_tps` between `int` and
/// `double` representations don't silently fall through.
double? _readDoubleField(Map<String, dynamic> m, List<String> names) {
  for (final n in names) {
    final v = m[n];
    if (v is num) return v.toDouble();
  }
  return null;
}

/// Parses the Cactus C engine envelope and extracts:
///   - `response`           -> assistant text (run through `_stripLatex`).
///   - `decode_tokens`      -> output tokens (the cycle-line's `respTok`).
///   - `prefill_tokens`     -> prompt tokens (the cycle-line's `promptTok`).
///   - `decode_tps`         -> engine-reported decode throughput.
///   - `prefill_tps`        -> engine-reported prefill throughput.
///   - `time_to_first_token_ms` -> prefill latency in ms.
///   - `total_time_ms`      -> engine-internal total wall clock in ms.
///
/// The schema was discovered empirically via the Run 13 `[Gen]
/// envelopeKeys=[…]` forensic dump on the validation Android device: the
/// on-device cactus@d917981f envelope contains 15 keys, of which six
/// are timing / counter fields (`decode_tokens`, `prefill_tokens`,
/// `decode_tps`,
/// `prefill_tps`, `time_to_first_token_ms`, `total_time_ms`, plus
/// `total_tokens` which is just the sum). The full key set is:
/// `[cloud_handoff, confidence, decode_tokens, decode_tps, error,
///  function_calls, prefill_tokens, prefill_tps, ram_usage_mb, response,
///  segments, success, time_to_first_token_ms, total_time_ms, total_tokens]`.
///
/// Field discovery is multi-schema-tolerant for cross-runtime
/// robustness. Lookup order, per field:
///
///   responseTokens: `decode_tokens` (Cactus canonical) → `tokens_predicted`
///     / `predicted_n` (llama.cpp canonical / short) → nested
///     `timings.predicted_n` / `timings.tokens_predicted` (llama.cpp modern).
///   promptTokens: `prefill_tokens` (Cactus) → `tokens_evaluated` / `prompt_n`
///     → nested `timings.prompt_n` / `timings.tokens_evaluated`.
///   engineDecodeTps: `decode_tps`. enginePrefillTps: `prefill_tps`.
///   timeToFirstTokenMs: `time_to_first_token_ms`. engineTotalMs: `total_time_ms`.
///
/// If no token-count field matches, the parser still records the
/// top-level key list in `envelopeKeysSummary` so the next `[Gen]` diag
/// line surfaces any future schema drift for follow-up.
///
/// Tested against the Cactus envelope, llama.cpp envelopes, and partial
/// / malformed shapes in `test/services/cactus_engine_test.dart`.
_ParsedEnvelope _parseEnvelope(String resultJson) {
  if (resultJson.isEmpty) {
    return const _ParsedEnvelope(
      text: '',
      stats: GenerationStats(),
      envelopeKeysSummary: '<empty>',
    );
  }
  String raw = resultJson;
  int? promptTokens;
  int? responseTokens;
  double? engineDecodeTps;
  double? enginePrefillTps;
  int? timeToFirstTokenMs;
  int? engineTotalMs;
  String envelopeKeysSummary = '<non-json>';
  try {
    final decoded = jsonDecode(resultJson);
    if (decoded is Map<String, dynamic>) {
      final keys = decoded.keys.toList()..sort();
      envelopeKeysSummary = '[${keys.join(",")}]';

      final response = decoded['response'];
      if (response is String) raw = response;

      responseTokens = _readIntField(decoded, const [
        'decode_tokens',     // Cactus canonical (Run 13 schema)
        'tokens_predicted',  // llama.cpp canonical
        'predicted_n',       // llama.cpp short form
        'tokensPredicted',   // hypothetical camelCase
      ]);
      promptTokens = _readIntField(decoded, const [
        'prefill_tokens',    // Cactus canonical
        'tokens_evaluated',  // llama.cpp canonical
        'prompt_n',          // llama.cpp short
        'tokensEvaluated',
      ]);
      engineDecodeTps = _readDoubleField(decoded, const ['decode_tps']);
      enginePrefillTps = _readDoubleField(decoded, const ['prefill_tps']);
      timeToFirstTokenMs =
          _readIntField(decoded, const ['time_to_first_token_ms']);
      engineTotalMs = _readIntField(decoded, const ['total_time_ms']);

      // Cross-runtime fallback: modern llama.cpp nests timing counters
      // inside a `timings` object. Probe that if we didn't find them at
      // the top level (Cactus puts them top-level, but a future
      // realignment shouldn't silently break the parser).
      final timings = decoded['timings'];
      if (timings is Map<String, dynamic>) {
        responseTokens ??= _readIntField(timings, const [
          'predicted_n',
          'tokens_predicted',
        ]);
        promptTokens ??= _readIntField(timings, const [
          'prompt_n',
          'tokens_evaluated',
        ]);
      }
    } else {
      envelopeKeysSummary = '<not-a-map>';
    }
  } on FormatException {
    // Non-JSON envelope (buffer overflow path); fall through with the raw
    // string and empty stats — the cycle line will show promptTok=? respTok=?
    // which the log reader can recognize as "envelope was unparseable".
  }
  return _ParsedEnvelope(
    text: _stripLatex(raw),
    stats: GenerationStats(
      promptTokens: promptTokens,
      responseTokens: responseTokens,
      engineDecodeTps: engineDecodeTps,
      enginePrefillTps: enginePrefillTps,
      timeToFirstTokenMs: timeToFirstTokenMs,
      engineTotalMs: engineTotalMs,
    ),
    envelopeKeysSummary: envelopeKeysSummary,
  );
}

/// Generalized LaTeX/math-markup stripper. Belt-and-suspenders companion
/// to the system prompt's "no LaTeX" directive — even if the model
/// regresses and emits LaTeX, the chat UI shows clean prose.
///
/// Two passes, both general:
///
/// 1. Any LaTeX-style command of the form `\<word>{<content>}` reduces
///    to its content. Catches `\text{...}`, `\mathrm{...}`,
///    `\boldsymbol{...}`, `\emph{...}`, `\texttt{...}`, etc. without
///    enumerating each. The loop iterates so nested commands like
///    `\text{\mathrm{abc}}` collapse fully.
///
/// 2. Any standard math-delimiter wrapper reduces to its content:
///    `$$...$$`, `$...$`, `\(...\)`, `\[...\]`. Conservative on the
///    `$...$` form: requires non-empty content with no inner `$` or
///    newline so currency strings like "$5 per kWh" are left intact.
String _stripLatex(String input) {
  var s = input;

  final commandRe = RegExp(r'\\[a-zA-Z]+\{([^{}]*)\}');
  String prev;
  do {
    prev = s;
    s = s.replaceAllMapped(commandRe, (m) => m.group(1) ?? '');
  } while (s != prev);

  s = s.replaceAllMapped(
    RegExp(
        r'\$\$([^$]*?)\$\$|\$([^$\n]+?)\$|\\\(([^)]*?)\\\)|\\\[([^\]]*?)\\\]'),
    (m) =>
        m.group(1) ?? m.group(2) ?? m.group(3) ?? m.group(4) ?? '',
  );

  return s;
}
