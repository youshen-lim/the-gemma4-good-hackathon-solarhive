// Integration smoke test for the SolarHive Cactus app.
//
// Requires a connected device (physical or ARM64 AVD) with the SolarHive
// Cactus artifact already downloaded into app-local storage. On a fresh
// install the LoadingScreen will run for ~5-15 minutes downloading the
// ~6.94 GB artifact; this test waits up to 30 minutes for that path before
// asserting the Generate flow.
//
// Run with:
//   flutter test integration_test/app_smoke_test.dart \
//     --dart-define=HF_TOKEN=hf_xxxxxxxxxxxx
//
// On CI without a device, this file is excluded by default; only
// `flutter test` (no integration_test/ path) runs the unit + widget tests.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:mobile_cactus/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('SolarHive Cactus on-device smoke', () {
    testWidgets('app boots and reaches a launchable state within 30 minutes',
        (tester) async {
      app.main();
      // Allow the boot flow time to either: (a) skip download because the
      // artifact is cached, or (b) complete the first-launch download.
      await tester.pumpAndSettle(const Duration(minutes: 30));

      // Either the LoadingScreen finished and we're on ChatScreen, or
      // ChatScreen booted directly. We assert visible elements of the
      // post-boot state rather than naming the screen.
      expect(find.text('SolarHive'), findsOneWidget);
    }, timeout: const Timeout(Duration(minutes: 35)));

    testWidgets('Send button drives an inference end-to-end on-device',
        (tester) async {
      app.main();
      await tester.pumpAndSettle(const Duration(minutes: 30));

      // Reach ChatScreen if not already there. The bottom text field
      // and send icon button are rendered unconditionally on ChatScreen,
      // so their presence confirms we are post-LoadingScreen.
      expect(find.byType(TextField), findsOneWidget);
      expect(find.byIcon(Icons.send), findsOneWidget);

      // Type a prompt and tap send. This may take several seconds on a
      // physical device and several minutes on a QEMU-translated ARM64
      // AVD on an x86 host. We do not assert on the response text (the
      // model is sampled, so any verbatim-match would be flaky); we only
      // assert that the user's prompt has been added to the chat history
      // (proves the send path fired).
      await tester.enterText(
          find.byType(TextField), 'What is solar GHI?');
      await tester.pump();
      await tester.tap(find.byIcon(Icons.send));
      await tester.pumpAndSettle(const Duration(minutes: 15));

      expect(find.text('What is solar GHI?'), findsOneWidget);
    }, timeout: const Timeout(Duration(minutes: 50)));
  });
}
