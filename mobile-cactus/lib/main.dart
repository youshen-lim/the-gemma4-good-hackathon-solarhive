// SolarHive Cactus mobile app entry point.
//
// Boot flow:
//   1. App boots into LoadingScreen.
//   2. LoadingScreen verifies the artifact directory at
//      `${appDocs}/models/solarhive-e4b/`. If the cache is complete
//      (>= 2000 entries — see `isArtifactComplete` in storage.dart),
//      it routes immediately to ChatScreen with no download. Otherwise
//      it runs the resumable HF download (paginated; ~6.94 GB across
//      ~2,084 files) and routes to ChatScreen on completion.
//   3. ChatScreen drives the Cactus inference + cloud-routing probe.
//
// Always routing through LoadingScreen on boot keeps cache state as the
// single source of truth: a stale "first-launch-done" sentinel cannot
// override an incomplete download.

import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';

void main() {
  runApp(const SolarHiveCactusApp());
}

class SolarHiveCactusApp extends StatelessWidget {
  const SolarHiveCactusApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SolarHive — Cactus Mobile',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFF59E0B)),
        useMaterial3: true,
      ),
      home: const LoadingScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}
