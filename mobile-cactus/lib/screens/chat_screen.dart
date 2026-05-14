// Multi-turn chat screen.
//
// Renders a scrollable message history (user-right / assistant-left bubbles)
// + a bottom text-input field. The orange diagnostic card from the
// single-prompt era is retained — on any inference failure it surfaces
// audit summary + memory state + the cactus C engine log tail inline so
// post-mortem analysis is in-screen.

import 'package:flutter/material.dart';

import '../models/chat_message.dart';
import '../services/cactus_engine.dart';
import '../services/diagnostics.dart' as diag;
import '../services/inference_constants.dart' show kDefaultMaxNewTokens;
import '../services/llm_engine.dart';
import '../services/storage.dart';

/// Builds the engine instance the screen drives. Defaults to a real
/// `CactusEngine` resolved against the app-local artifact directory; tests
/// inject a fake-engine factory to exercise the failure-render paths.
typedef EngineFactory = Future<LlmEngine> Function();

/// Soft cap on the number of message turns held in history. Each turn is
/// roughly 50–150 tokens of prompt+response; with `kDefaultContextSize=1024`
/// and a doubled system prompt of ~270 tokens, we have ~750 tokens of
/// rolling-history budget. 14 turns ≈ 7 user + 7 assistant exchanges; the
/// banner warns the user before they hit the budget.
const int kHistoryTurnSoftCap = 14;

/// Hint text in the empty state, doubles as a click-to-fill suggestion.
const String kEmptyStateHint =
    'Ask SolarHive — for example: "What is solar GHI?"';

class ChatScreen extends StatefulWidget {
  /// Optional engine factory. When omitted, the screen creates a real
  /// `CactusEngine` against the on-device artifact directory.
  final EngineFactory? engineFactory;

  const ChatScreen({super.key, this.engineFactory});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  LlmEngine? _engine;
  final List<ChatMessage> _messages = [];
  final TextEditingController _input = TextEditingController();
  final ScrollController _scroll = ScrollController();
  bool _busy = false;
  bool _failed = false;

  @override
  void initState() {
    super.initState();
    _initEngine();
  }

  Future<void> _initEngine() async {
    final factory = widget.engineFactory ?? _defaultFactory;
    final engine = await factory();
    if (!mounted) return;
    setState(() => _engine = engine);
  }

  Future<LlmEngine> _defaultFactory() async {
    final dir = await resolveArtifactDir();
    return CactusEngine(artifactDir: dir);
  }

  @override
  void dispose() {
    _engine?.unload();
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    // T0 — first line of _send is as close to the tap event as Dart sees;
    // the Material IconButton ripple has already started but no async work
    // has been scheduled yet.
    final cycle = diag.CycleTimer();
    final text = _input.text.trim();
    if (text.isEmpty || _engine == null || _busy) return;

    final userMessage = ChatMessage(role: kRoleUser, content: text);
    setState(() {
      _messages.add(userMessage);
      _input.clear();
      _busy = true;
      _failed = false;
    });
    _scrollToEnd();

    cycle.mark('engineEntry');
    try {
      final response = await _engine!.generate(
        messages: List<ChatMessage>.unmodifiable(_messages),
        maxNewTokens: kDefaultMaxNewTokens,
      );
      cycle.mark('engineExit');
      if (!mounted) return;
      setState(() {
        _messages.add(ChatMessage(role: kRoleAssistant, content: response));
        _failed = false;
      });
    } catch (e) {
      cycle.mark('engineExit');
      if (!mounted) return;
      setState(() {
        _messages.add(ChatMessage(
            role: kRoleAssistant, content: 'Inference failed:\n$e'));
        _failed = true;
      });
    } finally {
      if (mounted) {
        setState(() => _busy = false);
        _scrollToEnd();
      }
      // Stamp T_render after the engine has built + submitted the next
      // frame. `addPostFrameCallback` fires once the frame is composited
      // and handed to the GPU — the closest "pixels scheduled" mark
      // Flutter exposes without platform plumbing. The summary line is
      // written from inside the callback so prompt/response token counts
      // come from the same generate() that produced the response.
      WidgetsBinding.instance.addPostFrameCallback((_) async {
        cycle.mark('responseRendered');
        final stats = _engine?.lastGenerationStats;
        await cycle.writeSummary(
          promptTokens: stats?.promptTokens,
          responseTokens: stats?.responseTokens,
          engineDecodeTps: stats?.engineDecodeTps,
          timeToFirstTokenMs: stats?.timeToFirstTokenMs,
        );
      });
    }
  }

  void _scrollToEnd() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.animateTo(
        _scroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  Widget _buildBubble(BuildContext context, ChatMessage m) {
    final theme = Theme.of(context);
    final isUser = m.isUser;
    final bg = isUser
        ? theme.colorScheme.primaryContainer
        : theme.colorScheme.surfaceContainerHigh;
    final fg = isUser
        ? theme.colorScheme.onPrimaryContainer
        : theme.colorScheme.onSurface;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.78,
        ),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Text(
          m.content,
          style: TextStyle(fontSize: 15, color: fg),
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Text(
          kEmptyStateHint,
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 15,
            color: Theme.of(context).colorScheme.outline,
          ),
        ),
      ),
    );
  }

  /// On failure, render the most recent ~20 cactus C engine + Dart
  /// diagnostic log lines plus the artifact audit summary. Placed under
  /// the message list so a long failure card doesn't push the input out
  /// of view.
  Widget? _buildDiagnosticTail() {
    if (!_failed) return null;
    final lines = diag.recentCactusLogs(max: 20);
    final auditReport =
        (_engine is CactusEngine) ? (_engine as CactusEngine).lastAudit : null;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.orange.shade50,
        border: Border.all(color: Colors.orange),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (auditReport != null) ...[
            Text(
              'Cache: ${auditReport.totalGiBSummary}',
              style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: Colors.orange.shade900),
            ),
            if (auditReport.issues.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(
                '${auditReport.issues.length} audit issue(s):',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: Colors.orange.shade900),
              ),
              for (final issue in auditReport.issues.take(5))
                Text('  • $issue',
                    style: TextStyle(
                        fontSize: 10, color: Colors.orange.shade900)),
              if (auditReport.issues.length > 5)
                Text('  + ${auditReport.issues.length - 5} more',
                    style: TextStyle(
                        fontSize: 10, color: Colors.orange.shade900)),
            ],
            const Divider(height: 12),
          ],
          Text(
            'Cactus log tail (last ${lines.length}):',
            style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: Colors.orange.shade900),
          ),
          if (lines.isEmpty)
            Text('  (no log lines captured)',
                style: TextStyle(fontSize: 10, color: Colors.orange.shade900))
          else
            for (final line in lines)
              Text(line,
                  style: TextStyle(
                      fontSize: 10,
                      fontFamily: 'monospace',
                      color: Colors.orange.shade900)),
          const SizedBox(height: 4),
          Text(
            'Full log on device: app_flutter/solarhive_diag.log '
            '(adb shell run-as com.solarhive.mobile_cactus cat ...)',
            style: TextStyle(
                fontSize: 10,
                fontStyle: FontStyle.italic,
                color: Colors.orange.shade900),
          ),
        ],
      ),
    );
  }

  /// Soft history-cap warning. Renders only when the conversation is
  /// approaching the contextSize=1024 budget so users get a heads-up
  /// before responses start losing the early turns.
  Widget? _buildHistoryWarning() {
    if (_messages.length < kHistoryTurnSoftCap) return null;
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        border: Border.all(color: Colors.amber),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        'Conversation is getting long (${_messages.length} turns). '
        'The model has a 1024-token context window; older turns may start '
        'getting truncated. Consider starting a fresh chat for a new topic.',
        style: TextStyle(fontSize: 11, color: Colors.amber.shade900),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final diagTail = _buildDiagnosticTail();
    final historyWarning = _buildHistoryWarning();
    final canSend =
        _engine != null && !_busy && _input.text.trim().isNotEmpty;

    return Scaffold(
      appBar: AppBar(title: const Text('SolarHive')),
      body: SafeArea(
        child: Column(
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(16, 12, 16, 8),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'On-device Gemma 4 E4B INT4 (Cactus)',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ),
            ),
            Expanded(
              child: _messages.isEmpty
                  ? _buildEmptyState()
                  : ListView.builder(
                      controller: _scroll,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      itemCount: _messages.length,
                      itemBuilder: (ctx, i) =>
                          _buildBubble(ctx, _messages[i]),
                    ),
            ),
            if (historyWarning != null) historyWarning,
            if (diagTail != null) diagTail,
            const Divider(height: 1),
            Padding(
              padding: const EdgeInsets.all(8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Expanded(
                    child: TextField(
                      controller: _input,
                      maxLines: 4,
                      minLines: 1,
                      enabled: _engine != null && !_busy,
                      decoration: InputDecoration(
                        hintText: _busy
                            ? 'Generating...'
                            : 'Type a question for SolarHive',
                        border: const OutlineInputBorder(),
                        isDense: true,
                      ),
                      onChanged: (_) => setState(() {}),
                      onSubmitted: (_) {
                        if (canSend) _send();
                      },
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    icon: _busy
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                            ),
                          )
                        : const Icon(Icons.send),
                    onPressed: canSend ? _send : null,
                    tooltip: 'Send',
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
