import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_cactus/services/storage.dart';

void main() {
  group('Storage path resolver', () {
    test('kSolarHiveArtifactDirName is the on-disk filesystem name we use', () {
      expect(kSolarHiveArtifactDirName, equals('solarhive-e4b'));
    });

    test('isArtifactComplete returns false for a non-existent directory', () async {
      final fakeDir = Directory(
        '${Directory.systemTemp.path}/mobile_cactus_test_does_not_exist_${DateTime.now().microsecondsSinceEpoch}',
      );
      expect(await fakeDir.exists(), isFalse);
      expect(await isArtifactComplete(fakeDir), isFalse);
    });

    test('isArtifactComplete returns false for an empty directory', () async {
      final tmp = await Directory.systemTemp.createTemp('mobile_cactus_empty_');
      try {
        expect(await isArtifactComplete(tmp), isFalse);
      } finally {
        await tmp.delete(recursive: true);
      }
    });

    test(
        'isArtifactComplete returns false when entry count is below the >= 2000 threshold',
        () async {
      final tmp = await Directory.systemTemp.createTemp('mobile_cactus_partial_');
      try {
        for (var i = 0; i < 50; i++) {
          await File('${tmp.path}/dummy_$i.weights').writeAsString('x');
        }
        expect(await isArtifactComplete(tmp), isFalse);
      } finally {
        await tmp.delete(recursive: true);
      }
    });

    test(
        'isArtifactComplete returns true when entry count meets the >= 2000 threshold',
        () async {
      final tmp = await Directory.systemTemp.createTemp('mobile_cactus_full_');
      try {
        // Use 2010 to clear the threshold without writing the full 2,084
        // runtime artifact + 4 curated card files.
        for (var i = 0; i < 2010; i++) {
          await File('${tmp.path}/dummy_$i').writeAsString('');
        }
        expect(await isArtifactComplete(tmp), isTrue);
      } finally {
        await tmp.delete(recursive: true);
      }
    });
  });
}
