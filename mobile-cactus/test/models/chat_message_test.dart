// Unit tests for ChatMessage — the multi-turn chat UI's data record.
//
// Pure Dart, no Flutter context required. Pins:
// - role constants
// - construction defaults (timestamp auto-populated when omitted)
// - isUser / isAssistant role helpers
// - toJsonShape() output matches the OpenAI-style message envelope
//   that cactus_engine.dart serialises into messagesJson for cactus_complete

import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_cactus/models/chat_message.dart';

void main() {
  group('ChatMessage role constants', () {
    test('kRoleUser is the OpenAI-canonical "user"', () {
      expect(kRoleUser, equals('user'));
    });

    test('kRoleAssistant is the OpenAI-canonical "assistant"', () {
      expect(kRoleAssistant, equals('assistant'));
    });
  });

  group('ChatMessage construction', () {
    test('defaults timestamp to now when omitted', () {
      final before = DateTime.now();
      final m = ChatMessage(role: kRoleUser, content: 'hi');
      final after = DateTime.now();
      expect(m.timestamp.isAfter(before.subtract(const Duration(seconds: 1))), isTrue);
      expect(m.timestamp.isBefore(after.add(const Duration(seconds: 1))), isTrue);
    });

    test('honours an explicit timestamp', () {
      final t = DateTime(2026, 5, 8, 12, 34, 56);
      final m = ChatMessage(role: kRoleUser, content: 'hi', timestamp: t);
      expect(m.timestamp, equals(t));
    });

    test('preserves the content verbatim', () {
      const text = 'What is solar GHI?';
      final m = ChatMessage(role: kRoleUser, content: text);
      expect(m.content, equals(text));
    });
  });

  group('Role helpers', () {
    test('isUser is true when role == kRoleUser', () {
      expect(ChatMessage(role: kRoleUser, content: 'hi').isUser, isTrue);
      expect(ChatMessage(role: kRoleUser, content: 'hi').isAssistant, isFalse);
    });

    test('isAssistant is true when role == kRoleAssistant', () {
      expect(ChatMessage(role: kRoleAssistant, content: 'hi').isAssistant, isTrue);
      expect(ChatMessage(role: kRoleAssistant, content: 'hi').isUser, isFalse);
    });

    test('isUser and isAssistant are both false on an unknown role', () {
      // Defensive: future code might introduce roles like "tool"; the
      // existing helpers should not match.
      final m = ChatMessage(role: 'system', content: 'x');
      expect(m.isUser, isFalse);
      expect(m.isAssistant, isFalse);
    });
  });

  group('toJsonShape — OpenAI-canonical envelope', () {
    test('emits exactly {role, content} keys, no timestamp leak', () {
      final m = ChatMessage(role: kRoleUser, content: 'What is GHI?');
      final shape = m.toJsonShape();
      expect(shape.keys.toSet(), equals({'role', 'content'}));
      expect(shape['role'], equals('user'));
      expect(shape['content'], equals('What is GHI?'));
    });

    test('preserves the role string exactly (no canonicalisation)', () {
      // The role helpers (isUser/isAssistant) are a Dart-side convenience;
      // toJsonShape must round-trip the original role string so the C
      // engine sees what the caller intended.
      final m = ChatMessage(role: 'arbitrary-future-role', content: 'x');
      expect(m.toJsonShape()['role'], equals('arbitrary-future-role'));
    });
  });
}
