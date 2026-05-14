// First-launch artifact downloader.
//
// Downloads the SolarHive Cactus artifact (~6.94 GB across ~2,084 files)
// from `Truthseeker87/solarhive-e4b-cactus` to the app's documents
// directory. Uses dio for streaming downloads with per-file progress
// callbacks so the LoadingScreen can render an aggregate progress bar.
//
// HuggingFace Hub does not offer a single-archive endpoint for an entire
// repo. We resolve the file list via the API tree endpoint (which carries
// per-file `size` ground truth) and download each file individually.
// LFS-tracked files are served from the LFS resolve endpoint transparently.
//
// HF tree pagination: the `/api/models/<repo>/tree/<rev>` endpoint
// returns at most 1000 entries per page. Subsequent pages live at the
// URL given by the `Link: <next-url>; rel="next"` header. We follow
// that chain in `_listAllFiles()` so the full ~2,084-file inventory is
// fetched, not just the first page.
//
// SIZE-VERIFIED RESUMABLE DOWNLOAD: each file's HF-reported size is
// captured during list. The resumable check compares local file size to
// the expected size and re-downloads any mismatch. This handles the
// failure mode where the wireless adb / Wi-Fi drops mid-write, leaving
// a partial-write file the simple `length() > 0` check would have
// preserved as "complete".

import 'dart:io';
import 'package:dio/dio.dart';

const String kHfRepoId = 'Truthseeker87/solarhive-e4b-cactus';
const String kHfApiBase = 'https://huggingface.co/api/models';
const String kHfResolveBase = 'https://huggingface.co';

/// HF read token for accessing the SolarHive Cactus model repository.
/// Compile-time constant read from `--dart-define=HF_TOKEN=hf_xxx` at
/// `flutter run` / `flutter build` time. Empty string (default) means
/// anonymous access; pass a read-scope token only while the repository
/// is private. Once the repository is publicly accessible, no token
/// is required:
///   flutter run --dart-define=HF_TOKEN=hf_xxxxxxxxxxxx
const String kHfToken = String.fromEnvironment('HF_TOKEN', defaultValue: '');

/// Lower-bound sanity threshold for the listed file count. The Cactus
/// converter produces ~2,084 runtime files; the HF repo carries 4 more
/// curated card files for a total around 2,088. If the listed count
/// drops below this threshold we have a strong signal that pagination
/// is being silently truncated (the original bug was an exact 1,000
/// cap from the HF tree API's unfollowed `Link: rel="next"` header).
const int kMinExpectedFileCount = 1500;

/// One file in the HF repo, as returned by the tree API. The `size`
/// field is authoritative — we use it both to detect partial-write
/// truncation in the resumable check and to expose ground-truth sizes
/// to the artifact audit.
class HfFileEntry {
  final String path;
  final int size;
  const HfFileEntry({required this.path, required this.size});
}

/// Parses an HTTP `Link` header and returns the URL associated with
/// `rel="next"`, or `null` if no next link is present. Pure function;
/// runs offline. Examples:
///
///   parseNextLink('<https://x?cursor=abc>; rel="next"')
///     -> 'https://x?cursor=abc'
///
///   parseNextLink('<https://x?cursor=p>; rel="prev", <https://x?cursor=n>; rel="next"')
///     -> 'https://x?cursor=n'
///
///   parseNextLink('<https://x?cursor=p>; rel="prev"')
///     -> null
///
///   parseNextLink('')
///     -> null
String? parseNextLink(String linkHeader) {
  if (linkHeader.isEmpty) return null;
  // Link header can carry multiple comma-separated entries; the
  // angle-bracketed URL precedes the rel parameter.
  final entries = linkHeader.split(',');
  final pattern = RegExp(r'<([^>]+)>\s*;\s*rel\s*=\s*"?next"?');
  for (final entry in entries) {
    final match = pattern.firstMatch(entry);
    if (match != null) return match.group(1);
  }
  return null;
}

class ArtifactDownloader {
  final Dio _dio;
  final Directory destDir;
  final void Function(double overall, String currentFile, int filesDone, int filesTotal)?
      onProgress;

  ArtifactDownloader({required this.destDir, this.onProgress})
      : _dio = Dio(BaseOptions(
          headers: kHfToken.isNotEmpty ? {'Authorization': 'Bearer $kHfToken'} : null,
          connectTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(minutes: 30),
        ));

  /// Lists every file in the repo with its expected size, following HF
  /// Link-header pagination until exhausted. Throws on network or auth
  /// failure.
  Future<List<HfFileEntry>> _listAllFiles() async {
    final files = <HfFileEntry>[];
    String? url = '$kHfApiBase/$kHfRepoId/tree/main';
    Map<String, dynamic>? queryParams = {'recursive': 'true'};
    var pageCount = 0;

    while (url != null) {
      pageCount += 1;
      final resp = await _dio.get<List<dynamic>>(
        url,
        queryParameters: queryParams,
      );
      final entries = resp.data ?? const [];
      for (final raw in entries) {
        if (raw is! Map<String, dynamic>) continue;
        if (raw['type'] != 'file') continue;
        final path = raw['path'] as String?;
        if (path == null) continue;
        // HF tree response uses `size` for direct files and may use
        // `lfs.size` for LFS-tracked files. Try both, default 0.
        final directSize = raw['size'];
        final lfsBlock = raw['lfs'];
        final lfsSize = lfsBlock is Map<String, dynamic> ? lfsBlock['size'] : null;
        final size = (directSize is int)
            ? directSize
            : (lfsSize is int ? lfsSize : 0);
        files.add(HfFileEntry(path: path, size: size));
      }

      // Headers are case-insensitive in HTTP/1.1; dio lowercases them.
      final linkHeader = resp.headers.value('link') ?? '';
      final next = parseNextLink(linkHeader);
      if (next == null) break;

      url = next;
      // The next URL already carries the cursor + recursive in its
      // query string; do not duplicate.
      queryParams = null;

      // Defensive cap to avoid an infinite-pagination bug becoming an
      // infinite loop. The current artifact takes 3 pages.
      if (pageCount > 20) {
        throw StateError(
            'HF tree pagination exceeded 20 pages; aborting to avoid '
            'an infinite loop. Last page yielded ${files.length} files.');
      }
    }

    return files;
  }

  /// Downloads the entire artifact. Resumable across launches via
  /// SIZE-VERIFIED check: existing files whose local size equals the
  /// HF-reported size are skipped; existing files with a size mismatch
  /// are deleted and re-fetched. Files with HF-reported size 0 (rare,
  /// usually empty marker files) are skipped if local exists.
  ///
  /// Throws if the listed file count drops below `kMinExpectedFileCount`,
  /// surfacing pagination or auth issues loudly rather than silently
  /// proceeding with a partial artifact.
  Future<void> downloadAll() async {
    final files = await _listAllFiles();

    if (files.length < kMinExpectedFileCount) {
      throw StateError(
          'Listed only ${files.length} files (expected >= $kMinExpectedFileCount). '
          'Likely cause: HF tree pagination truncated, repo permissions changed, '
          'or repo is empty. Aborting before partial download.');
    }

    final total = files.length;
    var done = 0;

    for (final entry in files) {
      final filePath = entry.path;
      final expectedSize = entry.size;
      final localPath = '${destDir.path}/$filePath';
      final localFile = File(localPath);

      // SIZE-VERIFIED resumable check. Three cases:
      //   - File exists, size matches expected → skip (resumable hit)
      //   - File exists, size mismatch (partial write or stale) → delete + re-fetch
      //   - File missing → fetch
      // For HF-reported size 0 files, fall back to "exists + non-empty"
      // since we have no ground truth to compare against.
      if (await localFile.exists()) {
        final localSize = await localFile.length();
        if (expectedSize > 0) {
          if (localSize == expectedSize) {
            done += 1;
            onProgress?.call(done / total, filePath, done, total);
            continue;
          } else {
            // Mismatch — log to stderr (visible in flutter_run.log) and
            // delete the partial file so the dio.download below writes
            // a fresh one.
            stderr.writeln(
                '[ArtifactDownloader] size mismatch on $filePath: '
                'local=$localSize expected=$expectedSize → re-fetching');
            await localFile.delete();
          }
        } else {
          // Size unknown from HF metadata; preserve any non-empty file.
          if (localSize > 0) {
            done += 1;
            onProgress?.call(done / total, filePath, done, total);
            continue;
          }
        }
      }

      await localFile.parent.create(recursive: true);

      final url = '$kHfResolveBase/$kHfRepoId/resolve/main/$filePath';
      await _dio.download(
        url,
        localPath,
        onReceiveProgress: (received, expected) {
          if (expected > 0) {
            final fileFraction = received / expected;
            final overall = (done + fileFraction) / total;
            onProgress?.call(overall, filePath, done, total);
          }
        },
      );

      // Post-download size verification — if dio.download finished but
      // the file is short of expected size (e.g., connection dropped
      // mid-stream and dio reported success on partial bytes), surface
      // it loudly so the issue isn't silently masked.
      if (expectedSize > 0) {
        final actualSize = await localFile.length();
        if (actualSize != expectedSize) {
          throw StateError(
              'Post-download size mismatch on $filePath: '
              'local=$actualSize expected=$expectedSize. '
              'Connection likely dropped mid-stream. Retry the download.');
        }
      }

      done += 1;
      onProgress?.call(done / total, filePath, done, total);
    }
  }
}
