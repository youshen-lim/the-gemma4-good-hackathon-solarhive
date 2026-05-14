// Abstract LLM engine interface.
//
// `CactusEngine` is the only concrete implementation today; the interface
// exists to let widget tests inject a fake engine that does not touch the
// Cactus FFI bindings (loading `libcactus.so` from a unit-test process is
// brittle).

import '../models/chat_message.dart';
import 'cactus_engine.dart' show GenerationStats;

export 'cactus_engine.dart' show GenerationStats;

/// The minimum public surface a Cactus-style LLM engine must expose for
/// `ChatScreen` to drive it.
abstract class LlmEngine {
  /// The raw exception thrown by the most recent failing load call, if any.
  /// `null` when no load has been attempted or the most recent attempt
  /// succeeded.
  Object? get loadError;

  /// Token-count stats from the most recent `generate()` call (output +
  /// prompt tokens, parsed from the engine's response envelope).
  /// Default `null` so fake engines used in widget tests don't have to
  /// fabricate stats — `CactusEngine` overrides it to surface real
  /// counts that the cycle-time instrumentation reads back.
  GenerationStats? get lastGenerationStats => null;

  /// Idempotent. Loads the model on first call. Reraises the underlying
  /// exception on failure so the caller's exception flow is preserved;
  /// the same exception is also stored on [loadError] for UI inspection
  /// after the throw is handled.
  Future<void> ensureLoaded();

  /// Multi-turn generation. The caller passes the full conversation
  /// history (alternating user / assistant turns); the engine prepends
  /// the SolarHive system prompt internally so it stays anchored to the
  /// on-device variant in `inference_constants.dart` regardless of UI
  /// state. Implementations should call `ensureLoaded()` internally;
  /// callers can rely on the engine being ready after a successful return.
  Future<String> generate({
    required List<ChatMessage> messages,
    int maxNewTokens,
  });

  /// Releases any FFI handle held by the implementation.
  Future<void> unload();
}
