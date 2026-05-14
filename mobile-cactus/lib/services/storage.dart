// App-local storage path resolver for the Cactus artifact.
//
// PATH CONVENTION:
//
//   final modelPath = '${(await getApplicationDocumentsDirectory()).path}/models/solarhive-e4b';
//
// The Cactus FFI loader (`cactusInit` in `lib/cactus.dart`) accepts any
// absolute path to a well-formed artifact directory — there is no slug
// machinery and no curated-registry resolution. We pick `models/solarhive-e4b/`
// purely as a tidy filesystem convention; the directory name is not load-bearing
// and can change without touching the engine code.

import 'dart:io';
import 'package:path_provider/path_provider.dart';

/// Filesystem path component for the SolarHive on-device artifact. Used
/// only to build a deterministic local path; the FFI loader does not look
/// this name up against any registry.
const String kSolarHiveArtifactDirName = 'solarhive-e4b';

/// Resolves the on-device directory where the SolarHive INT4 artifact is
/// downloaded and read from. Layout: `{appDocs}/models/solarhive-e4b/`.
/// Created on first call if it does not exist.
Future<Directory> resolveArtifactDir() async {
  final base = await getApplicationDocumentsDirectory();
  final artifactDir = Directory('${base.path}/models/$kSolarHiveArtifactDirName');
  if (!await artifactDir.exists()) {
    await artifactDir.create(recursive: true);
  }
  return artifactDir;
}

/// Heuristic completeness check. Returns `true` if the artifact directory
/// looks fully downloaded. The Cactus converter produces ~2,084 runtime
/// files (text decoder INT4 weights + audio Conformer FP16 + vision
/// encoder FP16 + tokenizer + chat template). The HF repo additionally
/// carries a small set of curated card files (README, license, header
/// image, .gitattributes), bringing the total to ~2,088. We accept
/// >= 2,000 entries as "looks complete" to tolerate minor count variance
/// across Cactus versions and HF repo metadata changes.
Future<bool> isArtifactComplete(Directory dir) async {
  if (!await dir.exists()) return false;
  final entries = await dir.list().toList();
  return entries.length >= 2000;
}
