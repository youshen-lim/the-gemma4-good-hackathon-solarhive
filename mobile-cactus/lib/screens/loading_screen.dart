// First-launch artifact-download progress UI.
//
// Renders a LinearProgressIndicator + status text while the
// ArtifactDownloader streams the ~6.94 GB Cactus artifact from
// `Truthseeker87/solarhive-e4b-cactus`. Routes to ChatScreen once the
// artifact is complete (or immediately, if a previous run already
// populated the cache).

import 'package:flutter/material.dart';

import '../services/artifact_downloader.dart';
import '../services/storage.dart';
import 'chat_screen.dart';

class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key});

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen> {
  double _progress = 0.0;
  String _status = 'Preparing artifact download...';
  String _currentFile = '';
  int _filesDone = 0;
  int _filesTotal = 0;
  String? _error;

  @override
  void initState() {
    super.initState();
    _startDownload();
  }

  Future<void> _startDownload() async {
    try {
      final dir = await resolveArtifactDir();
      // Cache is the single source of truth: if the artifact directory
      // already meets the completeness threshold, skip the download
      // entirely and route to ChatScreen.
      if (await isArtifactComplete(dir)) {
        if (mounted) _routeToChat();
        return;
      }

      setState(() => _status = 'Downloading SolarHive Cactus artifact...');
      final dl = ArtifactDownloader(
        destDir: dir,
        onProgress: (overall, currentFile, done, total) {
          if (!mounted) return;
          setState(() {
            _progress = overall.clamp(0.0, 1.0);
            _currentFile = currentFile;
            _filesDone = done;
            _filesTotal = total;
            _status = 'Downloading $done of $total files...';
          });
        },
      );
      await dl.downloadAll();
      if (mounted) _routeToChat();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  void _routeToChat() {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const ChatScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'SolarHive',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              const Text(
                'First launch: downloading the on-device AI model.',
                style: TextStyle(fontSize: 16),
              ),
              const SizedBox(height: 4),
              Text(
                'About 6.94 GB on Wi-Fi recommended. One-time only — '
                'subsequent launches are instant.',
                style: TextStyle(fontSize: 13, color: Colors.grey[700]),
              ),
              const SizedBox(height: 32),
              if (_error != null)
                Text(
                  'Download failed:\n$_error',
                  style: const TextStyle(color: Colors.red),
                )
              else ...[
                LinearProgressIndicator(value: _progress),
                const SizedBox(height: 12),
                Text(_status, style: const TextStyle(fontSize: 14)),
                const SizedBox(height: 4),
                Text(
                  _currentFile.isEmpty ? '' : 'Current: $_currentFile',
                  style: TextStyle(fontSize: 11, color: Colors.grey[600]),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
