// Cactus engine diagnostics — pre-init artifact audit + C engine log capture.
//
// Two diagnostic surfaces:
//
// 1. PRE-INIT AUDIT (auditArtifact)
//    Walks the artifact directory, records per-file sizes, flags anomalies
//    (zero-byte files, files much smaller than peers of the same kind,
//    missing required metadata files like config.txt / chat_template.jinja2).
//    Runs before cactus_init so we can distinguish "the cache is bad" from
//    "the binary can't read this format" before the C engine throws.
//
// 2. C ENGINE LOG CAPTURE (installCactusLogSink)
//    Hooks the cactus_log_set_callback FFI surface (defined in
//    `lib/cactus.dart`) and pipes every line emitted by the C engine into
//    a ring buffer plus an on-device file at
//    ${appDocs}/solarhive_diag.log. The C engine logs which weight file
//    it's loading when init fails, so the next failure shows the EXACT
//    file path that triggered "File corrupted: data extends beyond file
//    size" instead of the generic message.
//
// The on-device log file is pullable via:
//   adb shell "run-as com.solarhive.mobile_cactus cat \
//     /data/data/com.solarhive.mobile_cactus/app_flutter/solarhive_diag.log"

import 'dart:async';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../cactus.dart' as cactus;

/// One file's worth of audit data.
class FileAudit {
  final String relativePath;
  final int sizeBytes;

  FileAudit({required this.relativePath, required this.sizeBytes});

  bool get isZeroByte => sizeBytes == 0;
}

/// Anomaly classes the artifact audit can flag.
enum ArtifactIssueKind {
  /// File length is 0 (e.g., download truncated before any bytes written).
  zeroByteFile,

  /// A required metadata file the C engine reads at init time is missing.
  /// Per cactus@main convert output: chat_template.jinja2, config.txt,
  /// merges.txt, special_tokens.json, tokenizer.json, tokenizer_config.txt,
  /// vocab.txt.
  missingMetadataFile,

  /// File is suspiciously small relative to the median of files with the
  /// same suffix. May indicate a truncated download that the
  /// `if (file.exists() && file.length() > 0)` resumable check accepted.
  suspectTruncated,
}

class ArtifactIssue {
  final ArtifactIssueKind kind;
  final String relativePath;
  final String detail;

  ArtifactIssue(
      {required this.kind, required this.relativePath, required this.detail});

  @override
  String toString() => '[${kind.name}] $relativePath: $detail';
}

class ArtifactAuditReport {
  final int totalFiles;
  final int totalBytes;
  final List<FileAudit> files;
  final List<ArtifactIssue> issues;

  ArtifactAuditReport({
    required this.totalFiles,
    required this.totalBytes,
    required this.files,
    required this.issues,
  });

  String get totalGiBSummary =>
      '${(totalBytes / (1024 * 1024 * 1024)).toStringAsFixed(2)} GiB '
      'across $totalFiles files';
}

/// Required runtime metadata files. The C engine reads these at
/// cactus_init and rejects the artifact if any is missing.
const Set<String> kRequiredMetadataFiles = {
  'chat_template.jinja2',
  'config.txt',
  'merges.txt',
  'special_tokens.json',
  'tokenizer.json',
  'tokenizer_config.txt',
  'vocab.txt',
};

/// Walks the artifact directory and audits per-file state.
Future<ArtifactAuditReport> auditArtifact(Directory artifactDir) async {
  final files = <FileAudit>[];
  final issues = <ArtifactIssue>[];

  if (!await artifactDir.exists()) {
    return ArtifactAuditReport(
      totalFiles: 0,
      totalBytes: 0,
      files: const [],
      issues: [
        ArtifactIssue(
          kind: ArtifactIssueKind.missingMetadataFile,
          relativePath: '<artifact dir>',
          detail: 'directory does not exist at ${artifactDir.path}',
        ),
      ],
    );
  }

  await for (final entity in artifactDir.list(recursive: true, followLinks: false)) {
    if (entity is! File) continue;
    final size = await entity.length();
    final rel = entity.path.replaceFirst(artifactDir.path, '').replaceAll('\\', '/');
    final relTrim = rel.startsWith('/') ? rel.substring(1) : rel;
    files.add(FileAudit(relativePath: relTrim, sizeBytes: size));
    if (size == 0) {
      issues.add(ArtifactIssue(
        kind: ArtifactIssueKind.zeroByteFile,
        relativePath: relTrim,
        detail: 'file is 0 bytes',
      ));
    }
  }

  // Required-metadata-file check (top-level only).
  final namesAtRoot = files
      .where((f) => !f.relativePath.contains('/'))
      .map((f) => f.relativePath)
      .toSet();
  for (final required in kRequiredMetadataFiles) {
    if (!namesAtRoot.contains(required)) {
      issues.add(ArtifactIssue(
        kind: ArtifactIssueKind.missingMetadataFile,
        relativePath: required,
        detail: 'required metadata file not found at the artifact root',
      ));
    }
  }

  // NOTE: previous versions ran a cohort-median heuristic to flag
  // suspiciously-small files. Empirical validation on an Android device
  // run showed the heuristic produced ~1000 false positives because the
  // Cactus weight-file size distribution is bimodal (large transformer
  // tensor weights at MB scale vs. tiny per-tensor INT4 quantization
  // scale/zero-point parameters at 98–224 B). The 98-byte files are
  // CORRECT — `cactus_init` reads them successfully. The heuristic has
  // been removed.
  //
  // Partial-write truncation is now caught at download time by
  // `artifact_downloader.dart`'s post-download size check (HF
  // Content-Length verification), so the audit doesn't need a runtime
  // heuristic for it either.

  final totalBytes = files.fold<int>(0, (a, f) => a + f.sizeBytes);
  return ArtifactAuditReport(
    totalFiles: files.length,
    totalBytes: totalBytes,
    files: files,
    issues: issues,
  );
}

// ============================================================================
// C engine log capture
// ============================================================================

/// Ring-buffer capacity for the in-memory C engine log tail. Large enough
/// to capture a few thousand lines around an init failure so the chat UI
/// can show the most recent context inline.
const int _kLogBufferCapacity = 2000;

final List<String> _logBuffer = [];
File? _logFile;
StreamController<String>? _logController;
bool _sinkInstalled = false;

/// Stream of log lines emitted by the cactus C engine. The chat UI can
/// listen to surface live tail lines under the response area.
Stream<String> get cactusLogStream {
  _logController ??= StreamController<String>.broadcast();
  return _logController!.stream;
}

/// Most recent N log lines from the in-memory ring buffer. Useful for
/// rendering the failure context in the chat UI without round-tripping
/// the on-device log file.
List<String> recentCactusLogs({int max = 50}) {
  if (_logBuffer.length <= max) return List.of(_logBuffer);
  return _logBuffer.sublist(_logBuffer.length - max);
}

/// Opens the on-device diagnostic log file at `${appDocs}/solarhive_diag.log`
/// for Dart-side append-writes. Idempotent.
///
/// IMPORTANT: as of the on-device SIGABRT investigation, we no
/// longer hook `cactus_log_set_callback`. Reason:
/// `NativeCallable.isolateLocal` (used by cactus.dart's Dart-side wrapper
/// to bridge the C-engine callback to a Dart closure) MUST be invoked
/// from the isolate that registered it. The Cactus inference path runs
/// on a worker thread and emits log lines from there; calling our
/// closure from the worker thread crashes the libc++ `std::function`
/// dispatch (resolved via `llvm-addr2line` to
/// `cactus::Logger::log + 0x6c`, inside the std::function operator()).
///
/// The audit + Dart-side `writeDartDiagnostic` calls below run on the
/// main isolate so they remain safe. We give up the live C-engine log
/// stream in exchange; if needed for future debugging, switch to
/// `NativeCallable.listener` + a `ReceivePort` so cross-thread
/// invocation is queue-mediated rather than direct.
Future<void> installCactusLogSink({bool verbose = false}) async {
  if (_sinkInstalled) return;
  _sinkInstalled = true;

  final docsDir = await getApplicationDocumentsDirectory();
  _logFile = File('${docsDir.path}/solarhive_diag.log');
  // Truncate on first install so each app run starts with a clean log.
  await _logFile!.writeAsString(
      '=== solarhive_diag.log opened at ${DateTime.now().toIso8601String()} '
      '(C-engine log callback disabled — see installCactusLogSink note) ===\n');

  // Set engine log level so it would still write to native log surfaces
  // (logcat) at the requested verbosity, but DO NOT register a Dart
  // callback. cactus.cactusLogSetCallback(...) is intentionally absent.
  cactus.cactusLogSetLevel(verbose ? 0 : 1);
}

// ============================================================================
// Memory pressure / OOM sampler (Snapdragon-friendly)
// ============================================================================

/// One sample of process + system memory state, parsed from /proc.
class MemorySample {
  /// Process resident set size in kB (from /proc/self/status VmRSS).
  /// This is what Android's lmkd watches when scoring the app for kill.
  final int vmRssKb;

  /// High-water-mark RSS in kB (from /proc/self/status VmHWM). Useful
  /// post-crash diagnostic — tells us how much we used at peak.
  final int vmHwmKb;

  /// Total virtual memory mapping size (VmSize). Includes mmaped weights
  /// (which don't all stay resident); useful for sanity-check vs RSS.
  final int vmSizeKb;

  /// System-wide MemAvailable from /proc/meminfo. When this approaches
  /// zero across the device, lmkd starts killing low-priority apps.
  final int sysMemAvailableKb;

  /// System-wide total RAM (MemTotal in /proc/meminfo). For ratios.
  final int sysMemTotalKb;

  final DateTime timestamp;

  MemorySample({
    required this.vmRssKb,
    required this.vmHwmKb,
    required this.vmSizeKb,
    required this.sysMemAvailableKb,
    required this.sysMemTotalKb,
    required this.timestamp,
  });

  String get formatted {
    final rssGib = (vmRssKb / (1024 * 1024)).toStringAsFixed(2);
    final hwmGib = (vmHwmKb / (1024 * 1024)).toStringAsFixed(2);
    final sizeGib = (vmSizeKb / (1024 * 1024)).toStringAsFixed(2);
    final sysAvailGib = (sysMemAvailableKb / (1024 * 1024)).toStringAsFixed(2);
    final sysTotalGib = (sysMemTotalKb / (1024 * 1024)).toStringAsFixed(2);
    final pct = sysMemTotalKb > 0
        ? (100 * sysMemAvailableKb / sysMemTotalKb).toStringAsFixed(1)
        : '?';
    return 'RSS=${rssGib}GiB HWM=${hwmGib}GiB VmSize=${sizeGib}GiB '
        '| sysFree=${sysAvailGib}/${sysTotalGib}GiB ($pct% free)';
  }
}

/// One-shot read of process + system memory state. Returns null on
/// platforms where /proc isn't available (iOS, macOS) — Android is the
/// only target where this resolves.
Future<MemorySample?> readMemorySample() async {
  if (!Platform.isAndroid && !Platform.isLinux) return null;
  try {
    final status = await File('/proc/self/status').readAsString();
    final meminfo = await File('/proc/meminfo').readAsString();
    int parseKb(String haystack, String key) {
      final pattern = RegExp('^$key:\\s*(\\d+)\\s*kB', multiLine: true);
      final match = pattern.firstMatch(haystack);
      if (match == null) return 0;
      return int.tryParse(match.group(1)!) ?? 0;
    }

    return MemorySample(
      vmRssKb: parseKb(status, 'VmRSS'),
      vmHwmKb: parseKb(status, 'VmHWM'),
      vmSizeKb: parseKb(status, 'VmSize'),
      sysMemAvailableKb: parseKb(meminfo, 'MemAvailable'),
      sysMemTotalKb: parseKb(meminfo, 'MemTotal'),
      timestamp: DateTime.now(),
    );
  } catch (e) {
    return null;
  }
}

Timer? _memorySamplerTimer;
int _samplerSequence = 0;

/// Starts a periodic memory sampler. Each tick:
///   1. Reads /proc/self/status + /proc/meminfo.
///   2. Writes a one-line summary to the diagnostic log file.
///   3. Optionally invokes [onSample] (e.g., to render in the chat UI).
///
/// Idempotent — calling while a sampler is already running is a no-op.
/// Use [stopMemorySampler] to cancel.
///
/// Sample on every tick whether memory is under pressure or not so the
/// post-mortem log shows the full RSS trajectory leading up to a crash.
/// Default interval 1 second strikes a balance between resolution and
/// log file growth.
void startMemorySampler({
  Duration interval = const Duration(seconds: 1),
  void Function(MemorySample sample)? onSample,
}) {
  if (_memorySamplerTimer != null) return;
  _samplerSequence = 0;
  _memorySamplerTimer = Timer.periodic(interval, (_) async {
    final sample = await readMemorySample();
    if (sample == null) return;
    _samplerSequence += 1;
    final line = '[Mem][#$_samplerSequence] ${sample.formatted}';
    _logBuffer.add(line);
    if (_logBuffer.length > _kLogBufferCapacity) {
      _logBuffer.removeRange(0, _logBuffer.length - _kLogBufferCapacity);
    }
    _logController?.add(line);
    await _logFile?.writeAsString('$line\n', mode: FileMode.append, flush: false);
    onSample?.call(sample);
  });
}

/// Stops the periodic memory sampler. Idempotent.
void stopMemorySampler() {
  _memorySamplerTimer?.cancel();
  _memorySamplerTimer = null;
}

// ============================================================================
// CycleTimer — tap-to-render latency instrumentation
// ============================================================================

/// Lightweight wall-clock timer for a single chat-send cycle.
///
/// The chat screen creates one of these at the moment the send button is
/// tapped (T0), then calls [mark] at known phase boundaries:
///
///   - `engineEntry`     — right before `await engine.generate(...)`.
///   - `engineExit`      — right after the await returns (success or throw).
///   - `responseRendered` — inside `addPostFrameCallback` after the setState
///     that appends the assistant bubble; this is as close to "pixels
///     scheduled for the GPU" as Flutter exposes without platform plumbing.
///
/// [writeSummary] emits a single structured `[CYCLE]` line into the on-device
/// diagnostic log so `adb pull solarhive_diag.log` + `grep '\[CYCLE\]'`
/// yields one row per chat turn with the full phase breakdown + decode
/// throughput when token counts are supplied.
class CycleTimer {
  /// Short label that appears in the summary line. Defaults to `send`;
  /// future surfaces (e.g., a future "regenerate" button) can pass their
  /// own label so the same diag file disambiguates flows.
  final String label;

  /// All recorded marks, including the implicit `T0_sendTap` set in the
  /// constructor. Insertion order is preserved.
  final Map<String, DateTime> marks = {};

  CycleTimer([this.label = 'send']) {
    marks['T0_sendTap'] = DateTime.now();
  }

  /// First-recorded mark (T0). Convenience for callers computing
  /// total-cycle wall clock against an external timestamp.
  DateTime get t0 => marks['T0_sendTap']!;

  /// Stamps the current wall clock against [name]. Overwrites if [name]
  /// was already used (last-write-wins is fine for these phases).
  void mark(String name) {
    marks[name] = DateTime.now();
  }

  /// Elapsed milliseconds between two named marks, or null if either is
  /// missing. Negative values are possible if marks were recorded out of
  /// order; callers should treat them as a logic bug rather than swallow.
  int? deltaMs(String from, String to) {
    final a = marks[from];
    final b = marks[to];
    if (a == null || b == null) return null;
    return b.difference(a).inMilliseconds;
  }

  /// Builds the summary line without touching the shared log sink. Pure
  /// function over the recorded marks + the supplied token counts so unit
  /// tests can assert the wire format without `path_provider` plumbing.
  ///
  /// Schema (one line, machine-greppable):
  ///
  ///   [CYCLE] <label> tap->render=<ms>ms (prep=<ms>, generate=<ms>,
  ///     render=<ms>) promptTok=<n|?> respTok=<n|?>
  ///     decode=<f>tok/s|? engineDecode=<f>tok/s|? ttft=<n>ms|?
  ///
  /// Phase semantics:
  ///   - prep      = T0_sendTap     -> engineEntry
  ///   - generate  = engineEntry    -> engineExit
  ///   - render    = engineExit     -> responseRendered
  ///   - tap->render = T0_sendTap   -> responseRendered
  ///
  /// `decode` is the Dart-side wall-clock derived value
  /// (`respTok / generate_seconds`); it folds prefill into the
  /// denominator, so it under-reports steady-state decode rate.
  /// `engineDecode` is Cactus's own internal `decode_tps` — same
  /// numerator, denominator excludes prefill, so a cleaner number.
  /// `ttft` is the engine-reported time-to-first-token. All three
  /// degrade to `?` when their inputs are absent so log parsers can
  /// branch without panicking on `NaN`.
  String formatSummary({
    int? promptTokens,
    int? responseTokens,
    double? engineDecodeTps,
    int? timeToFirstTokenMs,
  }) {
    final prepMs = deltaMs('T0_sendTap', 'engineEntry');
    final genMs = deltaMs('engineEntry', 'engineExit');
    final renderMs = deltaMs('engineExit', 'responseRendered');
    final totalMs = deltaMs('T0_sendTap', 'responseRendered');

    String fmtMs(int? v) => v == null ? '?' : '${v}ms';
    String fmtTok(int? v) => v == null ? '?' : '$v';

    String decode;
    if (responseTokens != null && genMs != null && genMs > 0) {
      final tps = responseTokens / (genMs / 1000.0);
      decode = '${tps.toStringAsFixed(2)}tok/s';
    } else {
      decode = '?';
    }

    final engineDecode = engineDecodeTps != null
        ? '${engineDecodeTps.toStringAsFixed(2)}tok/s'
        : '?';
    final ttft = timeToFirstTokenMs != null ? '${timeToFirstTokenMs}ms' : '?';

    return '[CYCLE] $label tap->render=${fmtMs(totalMs)} '
        '(prep=${fmtMs(prepMs)}, generate=${fmtMs(genMs)}, '
        'render=${fmtMs(renderMs)}) '
        'promptTok=${fmtTok(promptTokens)} respTok=${fmtTok(responseTokens)} '
        'decode=$decode engineDecode=$engineDecode ttft=$ttft';
  }

  /// Writes the structured summary line to the diagnostic log + the
  /// in-memory ring buffer. Returns the formatted line.
  Future<String> writeSummary({
    int? promptTokens,
    int? responseTokens,
    double? engineDecodeTps,
    int? timeToFirstTokenMs,
  }) async {
    final line = formatSummary(
      promptTokens: promptTokens,
      responseTokens: responseTokens,
      engineDecodeTps: engineDecodeTps,
      timeToFirstTokenMs: timeToFirstTokenMs,
    );
    await writeDartDiagnostic(line);
    return line;
  }
}

/// Writes a Dart-side diagnostic line into the same on-device log file as
/// the C engine logs, so the audit + the engine output interleave in the
/// pull. Includes an ISO timestamp.
Future<void> writeDartDiagnostic(String line) async {
  if (_logFile == null) {
    final docsDir = await getApplicationDocumentsDirectory();
    _logFile = File('${docsDir.path}/solarhive_diag.log');
  }
  final stamped = '[Dart][${DateTime.now().toIso8601String()}] $line';
  _logBuffer.add(stamped);
  if (_logBuffer.length > _kLogBufferCapacity) {
    _logBuffer.removeRange(0, _logBuffer.length - _kLogBufferCapacity);
  }
  _logController?.add(stamped);
  await _logFile!.writeAsString('$stamped\n', mode: FileMode.append, flush: false);
}
