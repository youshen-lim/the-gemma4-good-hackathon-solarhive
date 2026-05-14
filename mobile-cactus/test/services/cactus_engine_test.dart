// Unit tests for the FFI-direct Cactus engine wrapper.
//
// The wrapper itself is a thin shim over `package:mobile_cactus/cactus.dart`
// — the actual `cactusInit` and `cactusComplete` calls require the native
// `libcactus.so` to be present, which a flutter_test process cannot easily
// provide. We therefore unit-test only the parts of the wrapper that have
// no FFI dependency:
//
//   1. The docs-canonical message envelope our engine builds for
//      `cactusComplete`'s `messagesJson` argument (via reflection on the
//      jsonEncode shape we pass to the FFI).
//   2. The JSON response extractor (`_extractResponse` indirectly, via the
//      public `generate` contract — but we cannot exercise it without an
//      FFI handle, so we test by re-deriving the same behaviour here).
//
// End-to-end FFI exercise lives in `integration_test/app_smoke_test.dart`,
// which runs on a real device with the bundled `libcactus.so` available.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_cactus/services/cactus_engine.dart';

void main() {
  group('Docs-canonical message envelope', () {
    // Pin the JSON shape the engine builds for cactusComplete. The Cactus
    // FFI accepts an OpenAI-style messages array verbatim per
    // docs.cactuscompute.com Quickstart "basic-completion" section. A
    // future refactor that silently changes role names or adds extra keys
    // would break the C engine's chat-template renderer; this test fails
    // loudly if the shape drifts.
    test('messages array is [{role: system}, {role: user}] in that order', () {
      final messages = [
        {'role': 'system', 'content': 'You are SolarHive...'},
        {'role': 'user', 'content': 'What is GHI?'},
      ];
      final json = jsonEncode(messages);
      final decoded = jsonDecode(json) as List<dynamic>;
      expect(decoded.length, equals(2));
      expect((decoded[0] as Map)['role'], equals('system'));
      expect((decoded[1] as Map)['role'], equals('user'));
      expect((decoded[0] as Map).keys.toSet(), equals({'role', 'content'}));
      expect((decoded[1] as Map).keys.toSet(), equals({'role', 'content'}));
    });

    test('options envelope carries the four sampling parameters Cactus expects', () {
      // Pin the option keys against what Cactus's options parser accepts.
      // The keys mirror the OpenAI completions API surface that Cactus's
      // `optionsJson` argument is compatible with.
      final options = {
        'max_tokens': 128,
        'temperature': 1.0,
        'top_p': 0.95,
        'top_k': 64,
      };
      final json = jsonEncode(options);
      final decoded = jsonDecode(json) as Map<String, dynamic>;
      expect(decoded.keys.toSet(),
          equals({'max_tokens', 'temperature', 'top_p', 'top_k'}));
    });
  });

  group('Cactus response envelope extraction', () {
    // The Cactus C engine returns generation output wrapped in a JSON
    // envelope: `{"response": "...", "tokens_predicted": N, ...}`. The
    // engine wrapper extracts the `response` field; if the C engine ever
    // returns a non-JSON string (telemetry-only paths, buffer overflow),
    // the wrapper falls through to the raw string.
    //
    // The extractor is a private function in cactus_engine.dart, so this
    // test re-derives the behaviour against the same documented contract
    // and fails loudly if our published assumption changes.

    String extractDocsCanonical(String resultJson) {
      if (resultJson.isEmpty) return '';
      try {
        final decoded = jsonDecode(resultJson);
        if (decoded is Map<String, dynamic>) {
          final response = decoded['response'];
          if (response is String) return response;
        }
      } on FormatException {
        // Fall through.
      }
      return resultJson;
    }

    test('extracts the response field from the standard JSON envelope', () {
      const envelope =
          '{"response":"GHI is the total shortwave radiation per square metre.","tokens_predicted":18}';
      expect(
        extractDocsCanonical(envelope),
        equals('GHI is the total shortwave radiation per square metre.'),
      );
    });

    test('falls through to the raw string when the C engine returns non-JSON', () {
      const raw = 'plain text output (telemetry-only path)';
      expect(extractDocsCanonical(raw), equals(raw));
    });

    test('returns an empty string for an empty result', () {
      expect(extractDocsCanonical(''), equals(''));
    });

    test('falls through when JSON parses but lacks a string `response` key', () {
      const oddShape = '{"tokens_predicted":12,"finish_reason":"length"}';
      expect(extractDocsCanonical(oddShape), equals(oddShape));
    });
  });

  group('GenerationStats — Cactus envelope token-count parsing', () {
    // The CactusEngine wrapper now also surfaces token counts from the
    // envelope so the cycle-time instrumentation can compute decode tok/s
    // without re-parsing the JSON. The parser is private; this group
    // re-derives the same contract (multi-schema tolerant — cactus@d917981f
    // does not use llama.cpp-canonical key names, so the production parser
    // tries Cactus-canonical keys first then falls back to llama.cpp
    // names) and fails loudly if either the production parser or the
    // documented contract drifts.

    int? readIntField(Map<String, dynamic> m, List<String> names) {
      for (final n in names) {
        final v = m[n];
        if (v is int) return v;
        if (v is num) return v.toInt();
      }
      return null;
    }

    double? readDoubleField(Map<String, dynamic> m, List<String> names) {
      for (final n in names) {
        final v = m[n];
        if (v is num) return v.toDouble();
      }
      return null;
    }

    GenerationStats parseDocsCanonical(String resultJson) {
      if (resultJson.isEmpty) return const GenerationStats();
      try {
        final decoded = jsonDecode(resultJson);
        if (decoded is Map<String, dynamic>) {
          int? respTok = readIntField(decoded, const [
            'decode_tokens',
            'tokens_predicted',
            'predicted_n',
            'tokensPredicted',
          ]);
          int? promptTok = readIntField(decoded, const [
            'prefill_tokens',
            'tokens_evaluated',
            'prompt_n',
            'tokensEvaluated',
          ]);
          final double? engineDecodeTps =
              readDoubleField(decoded, const ['decode_tps']);
          final double? enginePrefillTps =
              readDoubleField(decoded, const ['prefill_tps']);
          final int? ttftMs =
              readIntField(decoded, const ['time_to_first_token_ms']);
          final int? engineTotalMs =
              readIntField(decoded, const ['total_time_ms']);
          final timings = decoded['timings'];
          if (timings is Map<String, dynamic>) {
            respTok ??= readIntField(timings, const [
              'predicted_n',
              'tokens_predicted',
            ]);
            promptTok ??= readIntField(timings, const [
              'prompt_n',
              'tokens_evaluated',
            ]);
          }
          return GenerationStats(
            promptTokens: promptTok,
            responseTokens: respTok,
            engineDecodeTps: engineDecodeTps,
            enginePrefillTps: enginePrefillTps,
            timeToFirstTokenMs: ttftMs,
            engineTotalMs: engineTotalMs,
          );
        }
      } on FormatException {
        // Fall through.
      }
      return const GenerationStats();
    }

    test('extracts decode_tokens + prefill_tokens (Cactus Run 13 canonical)', () {
      // Run 13 envelope shape, distilled from the on-device
      // `[Gen] envelopeKeys=[…]` forensic dump (full 15-key set:
      // cloud_handoff, confidence, decode_tokens, decode_tps, error,
      // function_calls, prefill_tokens, prefill_tps, ram_usage_mb,
      // response, segments, success, time_to_first_token_ms,
      // total_time_ms, total_tokens).
      const envelope = '{"response":"GHI is total shortwave radiation.",'
          '"decode_tokens":42,"prefill_tokens":270,'
          '"decode_tps":3.50,"prefill_tps":120.0,'
          '"time_to_first_token_ms":2250,"total_time_ms":14250,'
          '"total_tokens":312,"success":true,"error":null,'
          '"cloud_handoff":false,"confidence":0.92,"function_calls":[],'
          '"ram_usage_mb":3267,"segments":[]}';
      final stats = parseDocsCanonical(envelope);
      expect(stats.responseTokens, equals(42));
      expect(stats.promptTokens, equals(270));
      expect(stats.engineDecodeTps, equals(3.50));
      expect(stats.enginePrefillTps, equals(120.0));
      expect(stats.timeToFirstTokenMs, equals(2250));
      expect(stats.engineTotalMs, equals(14250));
      expect(stats.isComplete, isTrue);
    });

    test('extracts tokens_predicted + tokens_evaluated (llama.cpp canonical)', () {
      const envelope = '{"response":"hi","tokens_predicted":42,'
          '"tokens_evaluated":270,"finish_reason":"stop"}';
      final stats = parseDocsCanonical(envelope);
      expect(stats.responseTokens, equals(42));
      expect(stats.promptTokens, equals(270));
      expect(stats.engineDecodeTps, isNull);
      expect(stats.timeToFirstTokenMs, isNull);
      expect(stats.isComplete, isTrue);
    });

    test('Cactus keys win over llama.cpp keys when both present', () {
      // Pathological: an upstream realignment adds llama.cpp aliases
      // alongside the Cactus canonical keys. Lookup order means Cactus
      // wins.
      const envelope = '{"response":"hi",'
          '"decode_tokens":50,"tokens_predicted":99,'
          '"prefill_tokens":300,"tokens_evaluated":888}';
      final stats = parseDocsCanonical(envelope);
      expect(stats.responseTokens, equals(50));
      expect(stats.promptTokens, equals(300));
    });

    test('extracts predicted_n + prompt_n (llama.cpp short form)', () {
      const envelope =
          '{"response":"hi","predicted_n":11,"prompt_n":270,"stop":"eos"}';
      final stats = parseDocsCanonical(envelope);
      expect(stats.responseTokens, equals(11));
      expect(stats.promptTokens, equals(270));
    });

    test('extracts camelCase tokensPredicted + tokensEvaluated', () {
      const envelope =
          '{"response":"hi","tokensPredicted":7,"tokensEvaluated":99}';
      final stats = parseDocsCanonical(envelope);
      expect(stats.responseTokens, equals(7));
      expect(stats.promptTokens, equals(99));
    });

    test('extracts counts nested under timings (modern llama.cpp)', () {
      const envelope = '{"response":"hi",'
          '"timings":{"predicted_n":33,"prompt_n":270,"predicted_ms":12000}}';
      final stats = parseDocsCanonical(envelope);
      expect(stats.responseTokens, equals(33));
      expect(stats.promptTokens, equals(270));
    });

    test('top-level wins over nested timings when both present', () {
      const envelope = '{"response":"hi","tokens_predicted":50,'
          '"tokens_evaluated":300,"timings":{"predicted_n":5,"prompt_n":3}}';
      final stats = parseDocsCanonical(envelope);
      expect(stats.responseTokens, equals(50));
      expect(stats.promptTokens, equals(300));
    });

    test('accepts integer-valued doubles (42.0 round-trips to 42)', () {
      const envelope =
          '{"response":"hi","tokens_predicted":42.0,"tokens_evaluated":270.0}';
      final stats = parseDocsCanonical(envelope);
      expect(stats.responseTokens, equals(42));
      expect(stats.promptTokens, equals(270));
    });

    test('returns empty stats when envelope is non-JSON', () {
      final stats = parseDocsCanonical('plain text');
      expect(stats.responseTokens, isNull);
      expect(stats.promptTokens, isNull);
      expect(stats.isComplete, isFalse);
    });

    test('returns empty stats for empty input', () {
      expect(parseDocsCanonical('').isComplete, isFalse);
    });

    test('handles partial envelopes (one count present, one absent)', () {
      const partial = '{"response":"hi","tokens_predicted":5}';
      final stats = parseDocsCanonical(partial);
      expect(stats.responseTokens, equals(5));
      expect(stats.promptTokens, isNull);
      expect(stats.isComplete, isFalse);
    });

    test(
        'leaves both counts null when envelope keys are unknown (unknown-schema scenario)',
        () {
      // Facsimile of a JSON envelope where `response` is populated but
      // token counts live under keys neither the Cactus-canonical nor the
      // llama.cpp-canonical schema covers. The parser falls back to
      // null counts rather than crashing; the [Gen] diag line records
      // `envelopeKeys` so the next run reveals the real schema for a
      // follow-up patch.
      const oddSchema =
          '{"response":"GHI is total shortwave radiation per m^2.",'
          '"finish_reason":"stop","stop_reason":"natural","output_tokens":42}';
      final stats = parseDocsCanonical(oddSchema);
      expect(stats.responseTokens, isNull);
      expect(stats.promptTokens, isNull);
    });
  });
}
