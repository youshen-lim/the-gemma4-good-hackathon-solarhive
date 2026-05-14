// Drift detectors for sampling parameters + system prompt.
//
// These tests pin the on-device inference constants to the Kaggle Gemma 4
// defaults referenced in `solarhive_inference.py`. A future change that
// silently moves on-device away from those defaults will fail these tests
// and force an explicit re-evaluation rather than a quiet behavioural
// drift between runtimes.

import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_cactus/services/inference_constants.dart';

void main() {
  group('Sampling parameters -- Kaggle Gemma 4 defaults', () {
    test('temperature is 1.0 (Kaggle default)', () {
      expect(kKaggleTemperature, equals(1.0));
    });

    test('top-p is 0.95 (Kaggle default)', () {
      expect(kKaggleTopP, equals(0.95));
    });

    test('top-k is 64 (Kaggle default)', () {
      expect(kKaggleTopK, equals(64));
    });

    test('default max-new-tokens is a phone-friendly budget (between 128 and 1024)', () {
      // The cloud agentic loop uses 1024; on-device aims for tighter
      // first-token-to-completion latency. 512 is the current setting.
      expect(kDefaultMaxNewTokens, greaterThanOrEqualTo(128));
      expect(kDefaultMaxNewTokens, lessThanOrEqualTo(1024));
    });
  });

  group('System prompt -- on-device variant of the cloud SYSTEM_PROMPT', () {
    test('keeps SolarHive identity verbatim', () {
      expect(
        kSolarHiveSystemPrompt,
        contains('You are SolarHive, an AI energy advisor'),
      );
    });

    test('keeps the community facts (12 homes / Ann Arbor / rooftop solar + battery)', () {
      expect(kSolarHiveSystemPrompt, contains('12 homes'));
      expect(kSolarHiveSystemPrompt, contains('Ann Arbor, Michigan'));
      expect(kSolarHiveSystemPrompt, contains('rooftop solar'));
      expect(kSolarHiveSystemPrompt, contains('shared battery storage'));
    });

    test('keeps the response-length guidance verbatim', () {
      expect(kSolarHiveSystemPrompt, contains('3-5 sentences'));
    });

    test(
        'omits the cloud-only "call the available tools" instruction '
        '(no tools wired on-device)', () {
      // The cloud SYSTEM_PROMPT instructs the model to call tools. The
      // on-device variant must NOT carry that instruction or the model
      // will emit tool-call attempts the on-device tier cannot satisfy.
      expect(
        kSolarHiveSystemPrompt.toLowerCase(),
        isNot(contains('call the available tools')),
      );
      expect(
        kSolarHiveSystemPrompt.toLowerCase(),
        isNot(contains('call tools')),
      );
    });

    test('reframes "actual data" -> "reasonable assumptions" '
        '(no live API access on-device)', () {
      // The cloud prompt says "reference actual data". On-device has no
      // live API access; the prompt asks for "reasonable assumptions"
      // instead, so the model does not hallucinate live numbers.
      expect(kSolarHiveSystemPrompt, isNot(contains('reference actual data')));
      expect(kSolarHiveSystemPrompt, contains('reasonable assumptions'));
    });

    test('is doubled (mirrors cloud SYSTEM_PROMPT "Repeat to Improve" pattern)', () {
      // The cloud `SYSTEM_PROMPT = _UNIFIED_SYSTEM_BODY + "\n\n" + _UNIFIED_SYSTEM_BODY`
      // applies "Repeat to Improve" (Leviathan, Kalman & Matias 2024,
      // arXiv:2512.14982). On-device mirrors this verbatim per the
      // post-success polish: doubling the system prompt is ~+77 MB of
      // KV cache (negligible on the validation device's 12 GB RAM, with
      // the measured 6+ GiB system free during inference) and brings the
      // mobile tier into byte-for-byte prompt parity with the cloud.
      final identityCount =
          'SolarHive, an AI energy advisor'.allMatches(kSolarHiveSystemPrompt).length;
      expect(identityCount, equals(2),
          reason:
              'On-device prompt should contain the SolarHive identity exactly twice '
              '(doubled body separated by "\\n\\n", per Repeat-to-Improve).');
    });

    test('explicitly forbids LaTeX/markdown formatting in output', () {
      // Defensive directive paired with the _stripLatex post-processor in
      // cactus_engine.dart. The on-device smoke test had model emit
      // `$\text{W/m}^2$` which displays as garbled characters in the chat
      // UI; this directive is the in-prompt half of the fix.
      expect(kSolarHiveSystemPrompt.toLowerCase(),
          contains('do not use latex'));
    });
  });
}
