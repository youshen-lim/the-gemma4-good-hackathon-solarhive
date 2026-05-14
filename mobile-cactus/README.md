# SolarHive — Cactus Flutter Mobile App

**On-device Gemma 4 E4B inference via the [Cactus Flutter SDK](https://pub.dev/packages/cactus) on Android (and iOS, pending empirical validation).** Loads the SolarHive fine-tuned Cactus INT4 multimodal artifact from [`Truthseeker87/solarhive-e4b-cactus`](https://huggingface.co/Truthseeker87/solarhive-e4b-cactus) on first launch (~6.94 GB, one-time, cached to app-local storage), then runs all subsequent inference fully on-device with no network round-trips except the explicit cloud (🛰️) and microgrid hub (📡) escalations the user opts into.

Companion to:

- [`solarhive_e4b_cactus.ipynb`](../solarhive_e4b_cactus.ipynb) — the convert + smoke-test notebook
- [`Truthseeker87/solarhive-e4b-cactus`](https://huggingface.co/Truthseeker87/solarhive-e4b-cactus) — the HF repo this app downloads from
- [`solarhive_e4b_litert_v3.1.ipynb`](../solarhive_e4b_litert_v3.1.ipynb) — LiteRT-LM Python alternative
- [`web-litert/`](../web-litert/) — LiteRT browser companion on WebGPU

---

## Project layout

| File | Role |
|---|---|
| `pubspec.yaml` | Flutter project metadata + dependencies (`cactus`, `dio`, `path_provider`, `shared_preferences`) |
| `lib/main.dart` | App entry. Reads first-launch sentinel from `shared_preferences`; routes to `LoadingScreen` or `ChatScreen` |
| `lib/screens/loading_screen.dart` | First-launch download progress UI (LinearProgressIndicator + status text) |
| `lib/screens/chat_screen.dart` | Single-prompt smoke-test screen (future iterations replace this with the multi-turn chat UI) |
| `lib/services/artifact_downloader.dart` | HF Hub download via `dio` — lists repo files, streams each with progress callbacks; resumable across launches |
| `lib/services/cactus_engine.dart` | Cactus Flutter SDK wrapper + cloud-routing probe |
| `lib/services/storage.dart` | App-local artifact path resolver |
| `android/app/src/main/AndroidManifest.xml` | INTERNET + ACCESS_NETWORK_STATE permissions |

---

## Setup

### Prerequisites

- **Flutter SDK ≥ 3.22.0** — install per [official Flutter docs](https://docs.flutter.dev/get-started/install)
- **Android Studio** with Android SDK + emulator (for AVD development)
- **Physical Android device or ARM64 Android emulator.** Cactus's runtime requires ARM-compatible execution. On an ARM-host machine (Apple Silicon Mac, ARM Linux, ARM-based Windows), Android emulators run ARM64 with native acceleration. On x86 hosts, an `arm64-v8a` AVD will work via QEMU translation (slower but functional). An x86 AVD will fail at model load because `libcactus.so` has no x86 build.

Verify your install:

```bash
flutter doctor -v
```

All checks should pass before proceeding.

### One-time project initialisation

This scaffold ships the SolarHive-specific files (`lib/`, `pubspec.yaml`, `android/app/src/main/AndroidManifest.xml`) but not the full platform folder tree. Run `flutter create` once to generate the rest of the platform scaffolding without overwriting the source files:

```bash
cd mobile-cactus
flutter create . --org com.solarhive --project-name mobile_cactus --platforms android,ios
flutter pub get
```

`flutter create .` is safe to run on a directory with existing files — it adds platform scaffolding without touching `lib/`, `pubspec.yaml`, or the existing AndroidManifest entries. The `--platforms android,ios` flag scopes initial creation to mobile targets only.

### Run on Android emulator

```bash
flutter emulators --launch Pixel_7_API_34   # adjust name to match your AVD
flutter run
```

On first launch:

1. App boots into `LoadingScreen` and starts the ~6.94 GB download from `Truthseeker87/solarhive-e4b-cactus` (~5–15 min on Wi-Fi)
2. Once download completes, app routes to `ChatScreen`
3. Tap **Generate** to run the smoke-test prompt
4. The routing-strategy probe runs on the first generation call; the result is shown in the small status text below the response area

On subsequent launches, the artifact is already cached, so the app boots straight to `ChatScreen`.

### Run on a physical Android device

Connect a device with USB debugging enabled and ≥ 8 GB of free storage:

```bash
adb devices            # confirms the device is authorised
flutter run            # picks the connected device automatically
```

Physical devices give native ARM execution speed and are recommended for any user-facing demonstration.

---

## Architecture notes

### FFI-direct loading per Cactus docs

This app uses the docs-canonical Cactus Flutter integration recipe — every page on [docs.cactuscompute.com](https://docs.cactuscompute.com/latest/) (Quickstart Flutter tab, Flutter SDK page, Android page, Finetuning §6 *"Use in Android app"*) prescribes the same pattern: load the model from an absolute file path via `cactusInit`, drive completions with JSON-string messages and options. Verbatim from the [Quickstart Flutter tab](https://docs.cactuscompute.com/latest/docs/quickstart/#__tabbed_2_5):

```dart
import 'cactus.dart';
final model = cactusInit('/path/to/model', null, false);
final messages = '[{"role":"user","content":"What is capital of France?"}]';
final resultJson = cactusComplete(model, messages, null, null, null);
```

[`lib/cactus.dart`](lib/cactus.dart) is a verbatim copy of [`cactus-compute/cactus/flutter/cactus.dart`](https://github.com/cactus-compute/cactus/blob/main/flutter/cactus.dart) — same FFI bindings the docs reference. [`lib/services/cactus_engine.dart`](lib/services/cactus_engine.dart) wraps these calls with the SolarHive system prompt + Kaggle-recommended Gemma 4 sampling defaults; [`artifact_downloader.dart`](lib/services/artifact_downloader.dart) streams the ~6.94 GB Cactus artifact from [`Truthseeker87/solarhive-e4b-cactus`](https://huggingface.co/Truthseeker87/solarhive-e4b-cactus) into `${appDocs}/models/solarhive-e4b/` on first launch; on subsequent launches the directory is reused and `cactusInit` accepts it directly.

**Why FFI rather than the pub.dev OO API.** The published [`cactus: 1.3.0` package](https://pub.dev/packages/cactus) (publisher `cactuscompute.com`, verified) exposes a higher-level OO surface (`CactusLM`, `CactusInitParams`, `CompletionMode`) but **resolves models by short slug name only, against the curated [`Cactus-Compute/`](https://huggingface.co/Cactus-Compute) HF organisation**. SolarHive's fine-tune is a third-party repo, so the OO loader cannot resolve it without a slug-pre-population workaround. The FFI surface accepts any well-formed Cactus artifact directory by absolute path, which is exactly what `cactus convert` produces on the cloud side and what we ship at [`Truthseeker87/solarhive-e4b-cactus`](https://huggingface.co/Truthseeker87/solarhive-e4b-cactus). The pub.dev `cactus` dependency stays in [`pubspec.yaml`](pubspec.yaml) **purely as the binary provider** — Flutter's plugin manifest delivers `libcactus.so` (arm64-v8a) to `android/app/src/main/jniLibs/arm64-v8a/` automatically; we never `import 'package:cactus/cactus.dart'` from `lib/`.

### Cloud-routing strategy

The on-device tier escalates real-time-data queries to the SolarHive cloud tier via a 🛰️ emoji embedded in the model's output. The chat UI surfaces a routing-status line under the response area documenting the cloud handoff path; later iterations render an "Ask cloud for deeper analysis" button that POSTs the query + on-device response context to [`huggingface.co/spaces/Truthseeker87/solarhive`](https://huggingface.co/spaces/Truthseeker87/solarhive) (Gemma 4 26B A4B fine-tune) and renders the cloud response as a follow-up bubble. Microgrid hub routing (📡) follows the same pattern against a configured Ollama endpoint that serves the E4B fine-tune from the community microgrid hub.

We do not use the pub.dev `cactus` package's `CompletionMode.hybrid` — that mode falls back to OpenRouter, which does not host the SolarHive 26B A4B fine-tune. A roll-own POST against our HF Space is the only way to route to the actual SolarHive cloud target.

### Inference approach: on-device narrow tier vs. cloud agentic loop

The SolarHive cloud inference pipeline (`solarhive_inference.py`) runs a full agentic loop with five tools (OpenWeatherMap, EIA grid status, NREL PVWatts baseline, Open-Meteo solar production, battery state simulator), up to three loop rounds, and a two-message tool-result format matching the model's training distribution. The on-device tier in this app is intentionally narrower: a single-prompt → single-response call, no tools, no agentic loop. Real-time queries that need tool data emit 🛰️ on-device and the chat UI offers a cloud-handoff button that POSTs to the SolarHive HF Space — that's the entire point of the routing UX, and it's what keeps the on-device tier fast on phone hardware.

What stays consistent across the two tiers:

- **Sampling parameters** are pinned to the Kaggle-recommended Gemma 4 defaults (`temperature=1.0`, `top_p=0.95`, `top_k=64`) in [`lib/services/inference_constants.dart`](lib/services/inference_constants.dart). The same fine-tune family driving the same sampling regime keeps on-device behaviour comparable to the cloud benchmarks the project publishes.
- **System prompt identity** matches the cloud prompt verbatim (SolarHive identity, 12-home Ann Arbor community, rooftop solar + shared battery, 3-5 sentence response-length guidance). The on-device variant in `inference_constants.dart` drops the cloud-only "call the available tools" sentence (no tools wired on-device) and reframes "actual data" → "reasonable assumptions" (no live API access on-device). It is a single body rather than the doubled cloud prompt; the "Repeat to Improve" technique is a cloud-side latency trade we do not pay on phone hardware.

What is intentionally divergent:

- No tool-calling on-device. Routing escalation handles tool-needing queries via the 🛰️ emoji + cloud handoff path.
- No agentic multi-round loop on-device. Single prompt → single response keeps phone-side latency bounded.

Drift detectors in [`test/services/inference_constants_test.dart`](test/services/inference_constants_test.dart) pin the sampling values, the system prompt identity + community facts + length guidance, the absence of the "call tools" instruction, and the single-body shape. A change that silently moves on-device away from these alignment points fails those tests rather than drifting quietly.

---

## Known limitations

1. **HF token while the repo is private.** While `Truthseeker87/solarhive-e4b-cactus` access is restricted, `artifact_downloader.dart`'s `kHfToken` reads `String.fromEnvironment('HF_TOKEN', defaultValue: '')` at compile time. Pass a read-scope token at run/build time:

   ```bash
   flutter run --dart-define=HF_TOKEN=hf_xxxxxxxxxxxx
   ```

   The dio `Authorization` header is wired conditionally — empty string (default) sends no header for anonymous access; non-empty value sends `Bearer <token>`.

2. **iOS path unverified.** The current scaffold targets Android. The `--platforms android,ios` flag in `flutter create` is forward-looking — iOS validation is pending empirical testing on a physical device.

3. **Resumable download is heuristic.** Existing non-empty files in the artifact dir are skipped on subsequent launches. This works for partial-download recovery but does not validate file integrity (no SHA check). A future enhancement could call `HfApi.list_repo_files` for SHA-based validation.

4. **Platform minimum versions.** Per the Cactus pub.dev page: **iOS 12.0+** and **Android API 24+**. After running `flutter create .`, verify `android/app/build.gradle` has `minSdkVersion >= 24` and `ios/Podfile` has `platform :ios, '12.0'` (or higher).

---

## Testing

The app ships with a three-tier test harness mirroring the Python test conventions used elsewhere in the SolarHive repository.

### Tier 1 — unit tests (no device required)

Pure-Dart logic exercised with `flutter_test`. Runs on any machine with a Flutter SDK:

```bash
flutter test test/services/
```

| File | What it covers |
|---|---|
| `test/services/storage_test.dart` | `kSolarHiveArtifactDirName` constant; `isArtifactComplete()` boundary cases (non-existent dir, empty dir, partial download, full download) |
| `test/services/artifact_downloader_test.dart` | HF Hub URL constants; `kHfToken` `--dart-define` default; URL composition for the tree + per-file resolve endpoints |
| `test/services/cactus_engine_test.dart` | Pinned shape of the docs-canonical `messagesJson` envelope (`role: system`/`user` order + key set) and the `optionsJson` sampling-parameter set; response-extractor behaviour against the Cactus C engine's `{"response": "..."}` JSON envelope |

### Tier 2 — widget tests (no device required)

Renders individual screens in a headless test environment via `flutter_test`:

```bash
flutter test test/widgets/
```

| File | What it covers |
|---|---|
| `test/widgets/loading_screen_test.dart` | LoadingScreen renders the SolarHive heading, the one-time-download disclosure, and a LinearProgressIndicator on first build |
| `test/widgets/chat_screen_test.dart` | ChatScreen renders the SolarHive AppBar (drift detector — catches dev-phase labels leaking into user-facing titles), the on-device model header, the Generate button, the smoke-test placeholder, and asserts the routing-probe line is absent before a probe runs |

### Tier 3 — integration tests (device required)

Full end-to-end smoke test on a connected Android device or ARM64 AVD. The artifact must already be downloaded into app-local storage (or the test will block for up to 30 minutes waiting for the first-launch download to complete):

```bash
flutter test integration_test/app_smoke_test.dart \
  --dart-define=HF_TOKEN=hf_xxxxxxxxxxxx
```

| File | What it covers |
|---|---|
| `integration_test/app_smoke_test.dart` | App boot reaches a launchable state; tapping Generate drives a full inference and surfaces the routing-probe line |

### Run all non-device tests

```bash
flutter test               # unit + widget; skips integration_test/
```



Per the [Cactus repository](https://github.com/cactus-compute/cactus)'s recommended attribution:

```bibtex
@software{cactus,
  title  = {Cactus: AI Inference Engine for Phones & Wearables},
  author = {Ndubuaku, Henry and Cactus Team},
  url    = {https://github.com/cactus-compute/cactus},
  year   = {2025}
}
```

*Built with Gemma 4 in Ann Arbor, Michigan.*

*Gemma is a trademark of Google LLC.*
