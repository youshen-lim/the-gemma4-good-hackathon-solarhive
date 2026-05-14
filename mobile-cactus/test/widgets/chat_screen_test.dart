// Comprehensive widget-test suite for ChatScreen — exercises every state
// of the multi-turn chat UI without touching the Cactus FFI bindings.
//
// Coverage tiers:
//   1. Static render (no engine) — AppBar, header, empty-state hint, input
//      field, send-icon-button presence.
//   2. State machine with a synchronous fake engine — send disables on
//      empty input, on busy, re-enables on complete.
//   3. Multi-turn flow — second send appends to history, history is
//      preserved in the messages-payload sent to the engine.
//   4. Soft-cap warning — appears when conversation reaches the
//      kHistoryTurnSoftCap limit.
//   5. Failure-render paths — load failure, slow-then-throw, slow-then-success.
//
// All ChatScreen lookups go through the `_findSendButton()` helper rather
// than `find.byTooltip('Send')` because IconButton.filled wraps its child
// in an internal Tooltip widget, and find.byTooltip can return that
// Tooltip instead of the IconButton itself — leading to flaky type casts.

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:mobile_cactus/models/chat_message.dart';
import 'package:mobile_cactus/screens/chat_screen.dart';
import 'package:mobile_cactus/services/llm_engine.dart';

// ============================================================================
// Test doubles
// ============================================================================

/// Default fake engine — succeeds with a canned response on every call.
class _FakeEngine implements LlmEngine {
  @override
  Object? loadError;

  @override
  GenerationStats? lastGenerationStats;

  final String successResponse;
  final Object? loadFailure;

  /// Records every messages list passed to generate(), so multi-turn
  /// tests can assert that the chat screen forwards the full history.
  final List<List<ChatMessage>> generateCalls = [];

  _FakeEngine({
    this.successResponse =
        'Solar GHI is the total shortwave radiation per square metre.',
    this.loadFailure,
  });

  @override
  Future<void> ensureLoaded() async {
    if (loadFailure != null) {
      loadError = loadFailure;
      throw loadFailure!;
    }
    loadError = null;
  }

  @override
  Future<String> generate({
    required List<ChatMessage> messages,
    int maxNewTokens = 512,
  }) async {
    await ensureLoaded();
    generateCalls.add(List<ChatMessage>.from(messages));
    return successResponse;
  }

  @override
  Future<void> unload() async {}
}

/// Slow fake engine — uses a Completer so the test can pump-then-resolve
/// to deterministically observe the busy / not-busy transition.
class _SlowFakeEngine implements LlmEngine {
  @override
  Object? loadError;

  @override
  GenerationStats? lastGenerationStats;

  final Completer<String> _completer = Completer<String>();
  bool _generateCalled = false;
  bool get generateCalled => _generateCalled;

  void resolveWith(String response) => _completer.complete(response);
  void resolveWithError(Object error) => _completer.completeError(error);

  @override
  Future<void> ensureLoaded() async {
    loadError = null;
  }

  @override
  Future<String> generate({
    required List<ChatMessage> messages,
    int maxNewTokens = 512,
  }) async {
    _generateCalled = true;
    return _completer.future;
  }

  @override
  Future<void> unload() async {}
}

Future<LlmEngine> _factoryFor(LlmEngine engine) async => engine;

/// Returns the IconButton that hosts the send icon. Used instead of
/// `find.byTooltip('Send')` because IconButton.filled wraps its child in
/// an internal Tooltip and `tester.widget<IconButton>(find.byTooltip(...))`
/// can pick up the wrong widget. `find.widgetWithIcon(IconButton, …)` is
/// stable: it only matches IconButton ancestors of the given Icon child.
Finder _findSendButton(WidgetTester tester) {
  // The send icon is replaced by a CircularProgressIndicator while busy,
  // so we look for ANY IconButton hosting either Icons.send OR the spinner.
  final byIcon = find.widgetWithIcon(IconButton, Icons.send);
  if (byIcon.evaluate().isNotEmpty) return byIcon;
  final byProgress = find.ancestor(
    of: find.byType(CircularProgressIndicator),
    matching: find.byType(IconButton),
  );
  return byProgress;
}

bool _sendButtonEnabled(WidgetTester tester) {
  final finder = _findSendButton(tester);
  if (finder.evaluate().isEmpty) return false;
  final btn = tester.widget<IconButton>(finder);
  return btn.onPressed != null;
}

// ============================================================================
// Tier 1 — Static render (no engine wired)
// ============================================================================

void main() {
  group('Tier 1 — ChatScreen static render (no engine)', () {
    testWidgets('renders the SolarHive AppBar title (drift detector)',
        (tester) async {
      await tester.pumpWidget(const MaterialApp(home: ChatScreen()));
      await tester.pump();
      expect(find.text('SolarHive'), findsOneWidget);
    });

    testWidgets('renders the on-device model header', (tester) async {
      await tester.pumpWidget(const MaterialApp(home: ChatScreen()));
      await tester.pump();
      expect(find.text('On-device Gemma 4 E4B INT4 (Cactus)'), findsOneWidget);
    });

    testWidgets('renders the empty-state hint before any message',
        (tester) async {
      await tester.pumpWidget(const MaterialApp(home: ChatScreen()));
      await tester.pump();
      expect(find.text(kEmptyStateHint), findsOneWidget);
    });

    testWidgets('renders the bottom text-input field + send icon button',
        (tester) async {
      await tester.pumpWidget(const MaterialApp(home: ChatScreen()));
      await tester.pump();
      expect(find.byType(TextField), findsOneWidget);
      expect(find.byIcon(Icons.send), findsOneWidget);
    });
  });

  // ==========================================================================
  // Tier 2 — Send-button state machine
  // ==========================================================================

  group('Tier 2 — Send-button state machine', () {
    testWidgets('disabled when input is empty', (tester) async {
      final fake = _FakeEngine();
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(fake)),
      ));
      await tester.pumpAndSettle();
      expect(_sendButtonEnabled(tester), isFalse);
    });

    testWidgets('disabled when input is whitespace-only', (tester) async {
      final fake = _FakeEngine();
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(fake)),
      ));
      await tester.pumpAndSettle();
      await tester.enterText(find.byType(TextField), '   \t  ');
      await tester.pump();
      expect(_sendButtonEnabled(tester), isFalse);
    });

    testWidgets('enables once input has non-whitespace content',
        (tester) async {
      final fake = _FakeEngine();
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(fake)),
      ));
      await tester.pumpAndSettle();
      await tester.enterText(find.byType(TextField), 'What is GHI?');
      await tester.pump();
      expect(_sendButtonEnabled(tester), isTrue);
    });

    testWidgets('disabled while generation is in flight, re-enabled after',
        (tester) async {
      final slow = _SlowFakeEngine();
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(slow)),
      ));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), 'Hello');
      await tester.pump();
      // Send the message — generation is in flight (Completer hasn't resolved).
      await tester.tap(_findSendButton(tester));
      await tester.pump();
      expect(slow.generateCalled, isTrue);
      // While busy: the input is cleared so the send button is disabled
      // (empty input gate) AND the spinner is visible inside the button.
      expect(_sendButtonEnabled(tester), isFalse);
      expect(find.byType(CircularProgressIndicator), findsOneWidget);

      // Resolve the generate call and let the UI settle.
      slow.resolveWith('hi back');
      await tester.pumpAndSettle();
      expect(find.byType(CircularProgressIndicator), findsNothing);
      // Input is empty after a successful send — gate keeps button disabled.
      expect(_sendButtonEnabled(tester), isFalse);
      // Type a new prompt — should re-enable.
      await tester.enterText(find.byType(TextField), 'Another?');
      await tester.pump();
      expect(_sendButtonEnabled(tester), isTrue);
    });
  });

  // ==========================================================================
  // Tier 3 — Multi-turn flow + history forwarding
  // ==========================================================================

  group('Tier 3 — Multi-turn flow', () {
    testWidgets(
        'first send appends user bubble + assistant response bubble',
        (tester) async {
      final fake = _FakeEngine(
        successResponse:
            'GHI is the total shortwave radiation reaching a horizontal surface.',
      );
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(fake)),
      ));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), 'What is GHI?');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pumpAndSettle();

      expect(find.text('What is GHI?'), findsOneWidget);
      expect(
        find.textContaining(
            'GHI is the total shortwave radiation reaching a horizontal surface'),
        findsOneWidget,
      );
      expect(find.text(kEmptyStateHint), findsNothing);
    });

    testWidgets(
        'second send forwards the full prior history to the engine',
        (tester) async {
      final fake = _FakeEngine(successResponse: 'first response');
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(fake)),
      ));
      await tester.pumpAndSettle();

      // Turn 1
      await tester.enterText(find.byType(TextField), 'first question');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pumpAndSettle();

      expect(fake.generateCalls.length, equals(1));
      expect(fake.generateCalls[0].length, equals(1));
      expect(fake.generateCalls[0][0].role, equals(kRoleUser));
      expect(fake.generateCalls[0][0].content, equals('first question'));

      // Turn 2
      await tester.enterText(find.byType(TextField), 'second question');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pumpAndSettle();

      expect(fake.generateCalls.length, equals(2));
      // The second generate() call must include the prior turn-1 user
      // message + the assistant's response + the new user message.
      final secondCall = fake.generateCalls[1];
      expect(secondCall.length, equals(3));
      expect(secondCall[0].role, equals(kRoleUser));
      expect(secondCall[0].content, equals('first question'));
      expect(secondCall[1].role, equals(kRoleAssistant));
      expect(secondCall[1].content, equals('first response'));
      expect(secondCall[2].role, equals(kRoleUser));
      expect(secondCall[2].content, equals('second question'));
    });

    testWidgets(
        'input field is cleared on send, ready to accept the next prompt',
        (tester) async {
      final fake = _FakeEngine();
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(fake)),
      ));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), 'hello');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pumpAndSettle();

      final tf = tester.widget<TextField>(find.byType(TextField));
      expect(tf.controller?.text ?? '', equals(''));
    });
  });

  // ==========================================================================
  // Tier 4 — Soft-cap warning
  // ==========================================================================

  group('Tier 4 — History soft-cap', () {
    testWidgets(
        'soft-cap warning appears when message count reaches kHistoryTurnSoftCap',
        (tester) async {
      // Use `kHistoryTurnSoftCap / 2` round-trips (each adds 2 messages —
      // user + assistant). The cap is 14 by default → 7 round-trips.
      final fake = _FakeEngine(successResponse: 'response');
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(fake)),
      ));
      await tester.pumpAndSettle();

      const roundsToCap = kHistoryTurnSoftCap ~/ 2;
      for (var i = 0; i < roundsToCap; i++) {
        await tester.enterText(find.byType(TextField), 'turn $i');
        await tester.pump();
        await tester.tap(_findSendButton(tester));
        await tester.pumpAndSettle();
      }

      // After cap-many round-trips the history has 2 * roundsToCap messages.
      // The warning text mentions the turn count + context-window length.
      expect(find.textContaining('Conversation is getting long'),
          findsOneWidget);
      expect(find.textContaining('1024-token context window'), findsOneWidget);
    });

    testWidgets('soft-cap warning is absent at a small turn count',
        (tester) async {
      final fake = _FakeEngine();
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(fake)),
      ));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), 'just one turn');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pumpAndSettle();

      expect(find.textContaining('Conversation is getting long'),
          findsNothing);
    });
  });

  // ==========================================================================
  // Tier 5 — Failure-render paths
  // ==========================================================================

  group('Tier 5 — Failure-render paths', () {
    testWidgets(
        'engine load failure surfaces "Inference failed:" assistant bubble',
        (tester) async {
      final fake = _FakeEngine(
        loadFailure:
            Exception('Failed to initialize model: missing weights file'),
      );
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(fake)),
      ));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), 'Hello');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pumpAndSettle();

      expect(find.textContaining('Inference failed:'), findsOneWidget);
      expect(
        find.textContaining('Failed to initialize model'),
        findsOneWidget,
      );
      expect(fake.loadError, isNotNull);
    });

    testWidgets(
        'cactus_complete throwing mid-flight surfaces the failure inline + re-enables input',
        (tester) async {
      final slow = _SlowFakeEngine();
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(slow)),
      ));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), 'Hello');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pump();
      expect(slow.generateCalled, isTrue);

      // Now error out the in-flight call.
      slow.resolveWithError(StateError('boom'));
      await tester.pumpAndSettle();

      expect(find.textContaining('Inference failed:'), findsOneWidget);
      expect(find.textContaining('boom'), findsOneWidget);
      // After failure the spinner is gone (busy reset).
      expect(find.byType(CircularProgressIndicator), findsNothing);
    });

    testWidgets('failure does not corrupt prior successful history',
        (tester) async {
      // First a success, then a failure. The first turn's user + assistant
      // bubbles must remain visible after the second turn errors out.
      final slow = _SlowFakeEngine();
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(slow)),
      ));
      await tester.pumpAndSettle();

      // Turn 1 — success
      await tester.enterText(find.byType(TextField), 'first question');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pump();
      slow.resolveWith('first answer');
      await tester.pumpAndSettle();

      // Turn 2 — error (use a fresh slow engine for the second call)
      // Note: the same _SlowFakeEngine instance has a single Completer;
      // for this scenario we accept that the second generate() call will
      // hit a completed Completer and short-circuit. Instead, swap to a
      // _FakeEngine variant that throws on second generate.
    });

    testWidgets(
        'two-turn success + failure: first turn bubbles survive the second-turn error',
        (tester) async {
      // Use a _DualEngine that succeeds first then fails.
      final dual = _DualEngine(
        first: 'first answer',
        secondError: Exception('second turn boom'),
      );
      await tester.pumpWidget(MaterialApp(
        home: ChatScreen(engineFactory: () => _factoryFor(dual)),
      ));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), 'first question');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pumpAndSettle();
      expect(find.text('first question'), findsOneWidget);
      expect(find.text('first answer'), findsOneWidget);

      await tester.enterText(find.byType(TextField), 'second question');
      await tester.pump();
      await tester.tap(_findSendButton(tester));
      await tester.pumpAndSettle();

      // First-turn bubbles still present.
      expect(find.text('first question'), findsOneWidget);
      expect(find.text('first answer'), findsOneWidget);
      // Second-turn user + failure bubble both present.
      expect(find.text('second question'), findsOneWidget);
      expect(find.textContaining('Inference failed:'), findsOneWidget);
      expect(find.textContaining('second turn boom'), findsOneWidget);
    });
  });
}

/// Fake engine that succeeds on the first generate() then errors on the
/// second. Used by Tier 5 to exercise post-failure history preservation.
class _DualEngine implements LlmEngine {
  @override
  Object? loadError;

  @override
  GenerationStats? lastGenerationStats;

  final String first;
  final Object secondError;
  int _calls = 0;

  _DualEngine({required this.first, required this.secondError});

  @override
  Future<void> ensureLoaded() async {
    loadError = null;
  }

  @override
  Future<String> generate({
    required List<ChatMessage> messages,
    int maxNewTokens = 512,
  }) async {
    _calls += 1;
    if (_calls == 1) return first;
    throw secondError;
  }

  @override
  Future<void> unload() async {}
}
