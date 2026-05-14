import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:mobile_cactus/screens/loading_screen.dart';

void main() {
  group('LoadingScreen render', () {
    testWidgets('renders the SolarHive heading on first build', (tester) async {
      await tester.pumpWidget(const MaterialApp(home: LoadingScreen()));
      await tester.pump();
      expect(find.text('SolarHive'), findsOneWidget);
    });

    testWidgets('renders the one-time download disclosure copy', (tester) async {
      await tester.pumpWidget(const MaterialApp(home: LoadingScreen()));
      await tester.pump();
      expect(
        find.text('First launch: downloading the on-device AI model.'),
        findsOneWidget,
      );
    });

    testWidgets('renders a LinearProgressIndicator while no error is set', (tester) async {
      await tester.pumpWidget(const MaterialApp(home: LoadingScreen()));
      await tester.pump();
      expect(find.byType(LinearProgressIndicator), findsOneWidget);
    });
  });
}
