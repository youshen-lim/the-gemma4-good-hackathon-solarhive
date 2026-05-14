// Chat message data class for the multi-turn UI.
//
// Plain immutable record — `role` is one of `kRoleUser` / `kRoleAssistant`,
// `content` is the rendered text. Timestamps are captured at construction
// time and used only for ordering / debugging (the model doesn't see them).
//
// We don't model the `system` role here — the system prompt is injected by
// `CactusEngine.generate()` before sending to the C runtime so it stays
// anchored to the on-device variant in `inference_constants.dart` regardless
// of UI state.

const String kRoleUser = 'user';
const String kRoleAssistant = 'assistant';

class ChatMessage {
  final String role;
  final String content;
  final DateTime timestamp;

  ChatMessage({
    required this.role,
    required this.content,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  bool get isUser => role == kRoleUser;
  bool get isAssistant => role == kRoleAssistant;

  Map<String, String> toJsonShape() => {'role': role, 'content': content};
}
