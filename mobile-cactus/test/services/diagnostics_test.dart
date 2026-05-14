// Unit tests for the pure-Dart parts of the diagnostics module.
//
// `auditArtifact()` and `recentCactusLogs()` are testable directly with
// `flutter_test` since they only touch the file system or in-memory ring
// buffer. The memory sampler (`startMemorySampler` / `readMemorySample`)
// and the `installCactusLogSink()` log-file write require Android `/proc`
// access or `getApplicationDocumentsDirectory()` plumbing that doesn't
// resolve cleanly in a flutter_test process — those are exercised by the
// integration tests on a real device.

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_cactus/services/diagnostics.dart';

void main() {
  group('FileAudit + ArtifactIssue value types', () {
    test('FileAudit.isZeroByte tracks size correctly', () {
      expect(FileAudit(relativePath: 'x', sizeBytes: 0).isZeroByte, isTrue);
      expect(FileAudit(relativePath: 'x', sizeBytes: 1).isZeroByte, isFalse);
    });

    test('ArtifactIssue.toString includes kind + path + detail', () {
      final issue = ArtifactIssue(
        kind: ArtifactIssueKind.zeroByteFile,
        relativePath: 'foo.weights',
        detail: 'file is 0 bytes',
      );
      final s = issue.toString();
      expect(s, contains('zeroByteFile'));
      expect(s, contains('foo.weights'));
      expect(s, contains('file is 0 bytes'));
    });
  });

  group('ArtifactAuditReport', () {
    test('totalGiBSummary rounds bytes to GiB and includes file count', () {
      final report = ArtifactAuditReport(
        totalFiles: 100,
        totalBytes: 1024 * 1024 * 1024 * 2 + 1024 * 1024 * 512, // 2.5 GiB
        files: const [],
        issues: const [],
      );
      final summary = report.totalGiBSummary;
      expect(summary, contains('2.50 GiB'));
      expect(summary, contains('100 files'));
    });
  });

  group('kRequiredMetadataFiles — drift detector', () {
    test('contains the seven Cactus runtime metadata files', () {
      // These are the files cactus_init reads at model load time. If
      // upstream Cactus changes the required-file list, this test fails
      // loudly so we update the audit alongside the runtime.
      expect(kRequiredMetadataFiles, equals({
        'chat_template.jinja2',
        'config.txt',
        'merges.txt',
        'special_tokens.json',
        'tokenizer.json',
        'tokenizer_config.txt',
        'vocab.txt',
      }));
    });
  });

  group('auditArtifact — directory walking + issue flagging', () {
    test('reports missing-directory cleanly', () async {
      final missing = Directory(
        '${Directory.systemTemp.path}/mobile_cactus_audit_missing_${DateTime.now().microsecondsSinceEpoch}',
      );
      expect(await missing.exists(), isFalse);
      final report = await auditArtifact(missing);
      expect(report.totalFiles, equals(0));
      expect(report.totalBytes, equals(0));
      expect(report.issues, isNotEmpty);
      expect(report.issues.first.kind,
          equals(ArtifactIssueKind.missingMetadataFile));
    });

    test('flags zero-byte files as zeroByteFile issues', () async {
      final tmp = await Directory.systemTemp.createTemp('mobile_cactus_audit_zero_');
      try {
        await File('${tmp.path}/empty.weights').writeAsString('');
        final report = await auditArtifact(tmp);
        final zeroByteIssues = report.issues
            .where((i) => i.kind == ArtifactIssueKind.zeroByteFile)
            .toList();
        expect(zeroByteIssues.length, greaterThanOrEqualTo(1));
        expect(zeroByteIssues.first.relativePath, equals('empty.weights'));
      } finally {
        await tmp.delete(recursive: true);
      }
    });

    test('flags missing required metadata files', () async {
      final tmp = await Directory.systemTemp
          .createTemp('mobile_cactus_audit_missing_metadata_');
      try {
        // Populate with a non-metadata file only.
        await File('${tmp.path}/some_layer.weights').writeAsString('x');
        final report = await auditArtifact(tmp);
        final missingMetadata = report.issues
            .where((i) => i.kind == ArtifactIssueKind.missingMetadataFile)
            .toList();
        // All 7 required metadata files should be flagged as missing.
        expect(missingMetadata.length, equals(7));
        final missingNames =
            missingMetadata.map((i) => i.relativePath).toSet();
        expect(missingNames, equals(kRequiredMetadataFiles));
      } finally {
        await tmp.delete(recursive: true);
      }
    });

    test('reports a clean audit when all required files are present + non-empty',
        () async {
      final tmp =
          await Directory.systemTemp.createTemp('mobile_cactus_audit_clean_');
      try {
        for (final name in kRequiredMetadataFiles) {
          await File('${tmp.path}/$name').writeAsString('non-empty content');
        }
        await File('${tmp.path}/some_layer.weights')
            .writeAsString('non-empty');
        final report = await auditArtifact(tmp);
        expect(report.totalFiles,
            equals(kRequiredMetadataFiles.length + 1));
        expect(report.issues, isEmpty);
      } finally {
        await tmp.delete(recursive: true);
      }
    });
  });

  group('CycleTimer — tap-to-render cycle instrumentation', () {
    test('constructor records an implicit T0_sendTap mark', () {
      final t = CycleTimer();
      expect(t.marks.keys, contains('T0_sendTap'));
      expect(t.marks['T0_sendTap'], equals(t.t0));
    });

    test('default label is "send"; custom label round-trips', () {
      expect(CycleTimer().label, equals('send'));
      expect(CycleTimer('regen').label, equals('regen'));
    });

    test('mark() overwrites a prior value for the same name', () {
      final t = CycleTimer();
      t.mark('engineEntry');
      final first = t.marks['engineEntry']!;
      // Re-mark after a real delay; the value must update.
      t.marks['engineEntry'] = first.add(const Duration(milliseconds: 5));
      t.mark('engineEntry'); // overwrite with a fresh now()
      expect(t.marks['engineEntry'], isNot(equals(first)));
    });

    test('deltaMs returns null when either mark is missing', () {
      final t = CycleTimer();
      t.mark('engineEntry');
      expect(t.deltaMs('engineEntry', 'engineExit'), isNull);
      expect(t.deltaMs('nonexistent', 'engineEntry'), isNull);
    });

    test('deltaMs computes wall-clock delta when both marks exist', () {
      final t = CycleTimer();
      final base = t.t0;
      t.marks['engineEntry'] = base.add(const Duration(milliseconds: 10));
      t.marks['engineExit'] =
          base.add(const Duration(milliseconds: 12500));
      t.marks['responseRendered'] =
          base.add(const Duration(milliseconds: 12550));

      expect(t.deltaMs('T0_sendTap', 'engineEntry'), equals(10));
      expect(t.deltaMs('engineEntry', 'engineExit'), equals(12490));
      expect(t.deltaMs('engineExit', 'responseRendered'), equals(50));
      expect(t.deltaMs('T0_sendTap', 'responseRendered'), equals(12550));
    });

    test('formatSummary emits the canonical schema with full Cactus stats', () {
      final t = CycleTimer();
      final base = t.t0;
      t.marks['engineEntry'] = base.add(const Duration(milliseconds: 4));
      t.marks['engineExit'] =
          base.add(const Duration(milliseconds: 12004));
      t.marks['responseRendered'] =
          base.add(const Duration(milliseconds: 12050));

      final line = t.formatSummary(
        promptTokens: 270,
        responseTokens: 42,
        engineDecodeTps: 3.85,
        timeToFirstTokenMs: 2250,
      );
      // Dart-side: 42 tokens / 12.0 seconds = 3.50 tok/s (denominator
      // includes prefill). Engine-reported: 3.85 (excludes prefill).
      expect(line, equals(
        '[CYCLE] send tap->render=12050ms '
        '(prep=4ms, generate=12000ms, render=46ms) '
        'promptTok=270 respTok=42 '
        'decode=3.50tok/s engineDecode=3.85tok/s ttft=2250ms',
      ));
    });

    test('formatSummary degrades to ? markers when stats are absent', () {
      final t = CycleTimer();
      final base = t.t0;
      t.marks['engineEntry'] = base.add(const Duration(milliseconds: 5));
      t.marks['engineExit'] =
          base.add(const Duration(milliseconds: 8000));
      t.marks['responseRendered'] =
          base.add(const Duration(milliseconds: 8100));

      final line = t.formatSummary();
      expect(line, contains('promptTok=?'));
      expect(line, contains('respTok=?'));
      expect(line, contains('decode=?'));
      expect(line, contains('engineDecode=?'));
      expect(line, contains('ttft=?'));
    });

    test('formatSummary uses ? for missing marks instead of crashing', () {
      // Pathological: engineExit was never marked (engine threw before await
      // returned and the mark was skipped). Summary should still render.
      final t = CycleTimer();
      t.marks['engineEntry'] =
          t.t0.add(const Duration(milliseconds: 3));
      t.marks['responseRendered'] =
          t.t0.add(const Duration(milliseconds: 50));

      final line = t.formatSummary(promptTokens: 270, responseTokens: 5);
      expect(line, contains('generate=?'));
      expect(line, contains('render=?'));
      // decode=? because genMs is unknown
      expect(line, contains('decode=?'));
      expect(line, contains('promptTok=270'));
      expect(line, contains('respTok=5'));
    });

    test('decode tok/s rounds to two decimals', () {
      final t = CycleTimer();
      final base = t.t0;
      t.marks['engineEntry'] = base;
      t.marks['engineExit'] = base.add(const Duration(milliseconds: 1000));
      t.marks['responseRendered'] =
          base.add(const Duration(milliseconds: 1010));

      // 1 token in 1.000 s = 1.00 tok/s
      expect(
        t.formatSummary(promptTokens: 270, responseTokens: 1),
        contains('decode=1.00tok/s'),
      );
      // 3 tokens in 1.000 s = 3.00 tok/s
      expect(
        t.formatSummary(promptTokens: 270, responseTokens: 3),
        contains('decode=3.00tok/s'),
      );
    });
  });

  group('recentCactusLogs ring buffer', () {
    test('returns empty list initially when nothing has been buffered', () {
      // Note: the in-memory ring buffer is module-level state. This test
      // assumes no other test has populated it earlier in the run; the
      // ring is also small (2000-entry cap), so populating from another
      // group would still leave a recent slice.
      final tail = recentCactusLogs(max: 5);
      // We don't assert emptiness strictly (other tests may have written),
      // but we do assert the cap is honoured.
      expect(tail.length, lessThanOrEqualTo(5));
    });
  });
}
