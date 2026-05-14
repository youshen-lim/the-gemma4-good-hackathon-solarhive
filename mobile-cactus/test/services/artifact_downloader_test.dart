import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_cactus/services/artifact_downloader.dart';

void main() {
  group('ArtifactDownloader constants', () {
    test('kHfRepoId points to the canonical SolarHive Cactus repository', () {
      expect(kHfRepoId, equals('Truthseeker87/solarhive-e4b-cactus'));
    });

    test('kHfApiBase + kHfResolveBase are the documented HF Hub endpoints', () {
      expect(kHfApiBase, equals('https://huggingface.co/api/models'));
      expect(kHfResolveBase, equals('https://huggingface.co'));
    });

    test('kHfToken defaults to empty when --dart-define=HF_TOKEN is not passed', () {
      // This test runs without --dart-define, so the constant should be
      // the empty-string default. CI / local-dev runs that pass a token
      // override this assertion via `flutter test --dart-define=HF_TOKEN=hf_xxx`.
      expect(kHfToken, equals(''));
    });

    test('kMinExpectedFileCount is high enough to catch the 1000-page-cap bug', () {
      // The HF tree API caps each page at 1000 entries. The full
      // SolarHive Cactus artifact has ~2,084 runtime + 4 curated = 2,088
      // entries. The threshold must be above 1000 so the unfollowed-
      // pagination case fails loudly, and below ~2,084 so a small
      // upstream rename/addition does not falsely fail. 1500 is the
      // conservative midpoint.
      expect(kMinExpectedFileCount, greaterThan(1000));
      expect(kMinExpectedFileCount, lessThan(2050));
    });
  });

  group('ArtifactDownloader URL composition (smoke check)', () {
    test('list endpoint composes to the expected tree URL', () {
      final url = '$kHfApiBase/$kHfRepoId/tree/main';
      expect(
        url,
        equals('https://huggingface.co/api/models/Truthseeker87/solarhive-e4b-cactus/tree/main'),
      );
    });

    test('per-file resolve endpoint composes to the expected URL', () {
      final filePath = 'config.txt';
      final url = '$kHfResolveBase/$kHfRepoId/resolve/main/$filePath';
      expect(
        url,
        equals('https://huggingface.co/Truthseeker87/solarhive-e4b-cactus/resolve/main/config.txt'),
      );
    });
  });

  group('parseNextLink (HF tree pagination)', () {
    test('returns null on an empty header', () {
      expect(parseNextLink(''), isNull);
    });

    test('extracts the next URL from a single-entry header', () {
      const header = '<https://huggingface.co/api/models/foo/bar/tree/main?cursor=abc>; rel="next"';
      expect(
        parseNextLink(header),
        equals('https://huggingface.co/api/models/foo/bar/tree/main?cursor=abc'),
      );
    });

    test('finds the next URL among multiple comma-separated entries', () {
      const header =
          '<https://huggingface.co/api/models/foo/bar?cursor=p>; rel="prev", '
          '<https://huggingface.co/api/models/foo/bar?cursor=n>; rel="next"';
      expect(
        parseNextLink(header),
        equals('https://huggingface.co/api/models/foo/bar?cursor=n'),
      );
    });

    test('returns null when only rel="prev" is present', () {
      const header = '<https://huggingface.co/api/models/foo/bar?cursor=p>; rel="prev"';
      expect(parseNextLink(header), isNull);
    });

    test('tolerates whitespace variants in the header', () {
      const header =
          '<https://huggingface.co/api/models/foo/bar?cursor=n>;rel="next"'; // no space after ;
      expect(
        parseNextLink(header),
        equals('https://huggingface.co/api/models/foo/bar?cursor=n'),
      );
    });

    test('tolerates unquoted rel value (lenient match)', () {
      // RFC 5988 mandates quoted rel values, but some servers/proxies
      // strip quotes. The classifier is permissive on this one detail.
      const header = '<https://huggingface.co/api/models/foo/bar?cursor=n>; rel=next';
      expect(
        parseNextLink(header),
        equals('https://huggingface.co/api/models/foo/bar?cursor=n'),
      );
    });

    test('preserves cursor parameters with special characters in the URL', () {
      // HF cursors are base64-encoded JSON; can contain `=`, `+`, `/`.
      const header = '<https://huggingface.co/api/models/foo/bar?cursor=eyJfaWQiOiI2N2YifQ==>; rel="next"';
      expect(
        parseNextLink(header),
        equals('https://huggingface.co/api/models/foo/bar?cursor=eyJfaWQiOiI2N2YifQ=='),
      );
    });
  });
}
