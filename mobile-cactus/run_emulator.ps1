# SolarHive -- on-device smoke-test launcher for the mobile-cactus app.
#
# 1. Verify $env:HF_TOKEN_RUNTIME is set
# 2. Detect physical Android device via adb (Path A -- preferred)
# 3. If no physical device, boot SolarHive_Pixel7_ARM64 AVD (Path B fallback)
# 4. flutter run --dart-define=HF_TOKEN=$env:HF_TOKEN_RUNTIME
#
# Companion: setup_android_sdk.ps1 must complete first (one-time setup).

$ErrorActionPreference = 'Continue'

# --- Constants ---
$AvdName    = "SolarHive_Pixel7_ARM64"
$ProjectDir = "D:\YSL\projects\mobile-cactus"
$LogFile    = "$ProjectDir\flutter_run.log"

# --- Stage 0: token check ---
Write-Host "=== Stage 0: pre-flight ===" -ForegroundColor Cyan

# Background PowerShell sessions do not auto-load User env -- pull JAVA_HOME
# from the persisted User-level env (set by setup_android_sdk.ps1) so the
# Gradle build inside `flutter run` finds java without manual export.
if (-not $env:JAVA_HOME) {
    $userJavaHome = [System.Environment]::GetEnvironmentVariable('JAVA_HOME', 'User')
    if ($userJavaHome -and (Test-Path $userJavaHome)) {
        $env:JAVA_HOME = $userJavaHome
        $env:Path = "$env:JAVA_HOME\bin;$env:Path"
        Write-Host "  Loaded JAVA_HOME from user env: $env:JAVA_HOME"
    } else {
        Write-Host "  ERROR: JAVA_HOME not set in this session and no User-level " `
                   "JAVA_HOME found. Run .\setup_android_sdk.ps1 first." -ForegroundColor Red
        exit 1
    }
}

if (-not $env:HF_TOKEN_RUNTIME) {
    Write-Host "  ERROR: `$env:HF_TOKEN_RUNTIME not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Set for this session via:"
    Write-Host '    $env:HF_TOKEN_RUNTIME = "hf_xxxxxxxxxxx"'
    Write-Host ""
    Write-Host "  Generate a read-scope token at https://huggingface.co/settings/tokens"
    Write-Host "  (Truthseeker87/solarhive-e4b-cactus is private until submission day.)"
    exit 1
}
$tokenPreview = $env:HF_TOKEN_RUNTIME.Substring(0, [Math]::Min(7, $env:HF_TOKEN_RUNTIME.Length))
Write-Host "  Token: $tokenPreview..." -ForegroundColor Green

if (-not (Test-Path $env:ANDROID_HOME)) {
    Write-Host "  ERROR: `$env:ANDROID_HOME ('$env:ANDROID_HOME') not found" -ForegroundColor Red
    Write-Host "  Run .\setup_android_sdk.ps1 first" -ForegroundColor Yellow
    exit 1
}
$adb = "$env:ANDROID_HOME\platform-tools\adb.exe"
if (-not (Test-Path $adb)) {
    Write-Host "  ERROR: adb.exe not found at $adb" -ForegroundColor Red
    Write-Host "  Run .\setup_android_sdk.ps1 first" -ForegroundColor Yellow
    exit 1
}

# --- Stage 1: device detection ---
Write-Host ""
Write-Host "=== Stage 1: device detection ===" -ForegroundColor Cyan
& $adb start-server 2>&1 | Out-Null
$adbDevicesRaw = & $adb devices
Write-Host "  adb devices output:"
$adbDevicesRaw | ForEach-Object { Write-Host "    $_" }

# Lines look like "<serial>\tdevice" -- physical devices have non-emulator-N serials
$deviceLines = $adbDevicesRaw | Where-Object { $_ -match '^\S+\s+device$' }
$physicalDevices = $deviceLines | Where-Object { $_ -notmatch '^emulator-' }
$emulatorDevices = $deviceLines | Where-Object { $_ -match '^emulator-' }

$usingEmulator = $false
if ($physicalDevices) {
    Write-Host "  [OK] Physical device(s) detected -- Path A" -ForegroundColor Green
    $physicalDevices | ForEach-Object { Write-Host "    $_" }
} elseif ($emulatorDevices) {
    Write-Host "  Existing emulator detected -- re-using" -ForegroundColor Green
    $emulatorDevices | ForEach-Object { Write-Host "    $_" }
} else {
    Write-Host "  No device detected -- will boot AVD (Path B)" -ForegroundColor Yellow
    $usingEmulator = $true
}

# --- Stage 2: boot emulator if needed ---
if ($usingEmulator) {
    Write-Host ""
    Write-Host "=== Stage 2: boot AVD $AvdName ===" -ForegroundColor Cyan

    $emulatorBin = "$env:ANDROID_HOME\emulator\emulator.exe"
    if (-not (Test-Path $emulatorBin)) {
        Write-Host "  ERROR: emulator.exe not found at $emulatorBin" -ForegroundColor Red
        Write-Host "  Re-run .\setup_android_sdk.ps1" -ForegroundColor Yellow
        exit 1
    }

    $avds = & $emulatorBin -list-avds
    if ($avds -notcontains $AvdName) {
        Write-Host "  ERROR: AVD '$AvdName' not found." -ForegroundColor Red
        Write-Host "  Available AVDs:"
        if ($avds) { $avds | ForEach-Object { Write-Host "    $_" } }
        else { Write-Host "    (none)" }
        Write-Host "  Re-run .\setup_android_sdk.ps1" -ForegroundColor Yellow
        exit 1
    }

    Write-Host "  Launching $AvdName (background process)"
    Start-Process -FilePath $emulatorBin `
                  -ArgumentList "-avd", $AvdName, "-no-snapshot-save" `
                  -WindowStyle Minimized

    Write-Host "  Waiting for boot... (typical: 60-180s; longer on x86 hosts running ARM emulation via QEMU)"
    $bootTimeout = 600  # 10 min
    $start = Get-Date
    $booted = $false
    while (((Get-Date) - $start).TotalSeconds -lt $bootTimeout) {
        Start-Sleep -Seconds 10
        $bootProp = & $adb shell getprop sys.boot_completed 2>$null
        if ($bootProp -match '1') {
            $booted = $true
            break
        }
        $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 0)
        Write-Host "    [${elapsed}s] still booting..."
    }

    if (-not $booted) {
        Write-Host "  ERROR: AVD did not boot within ${bootTimeout}s" -ForegroundColor Red
        exit 1
    }
    Write-Host "  [OK] AVD booted" -ForegroundColor Green

    # Sometimes adb sees boot_completed=1 before the launcher is fully ready
    Start-Sleep -Seconds 5
}

# --- Stage 3: flutter run ---
Write-Host ""
Write-Host "=== Stage 3: flutter run ===" -ForegroundColor Cyan
Set-Location $ProjectDir

Write-Host "  Logging to: $LogFile"
Write-Host ""
Write-Host "  Watch for:" -ForegroundColor Yellow
Write-Host "    1. LoadingScreen -- download progress (~2,084 files / 6.94 GB)"
Write-Host "       Wi-Fi: 5-15 min. If 401/403: token issue (expired? wrong scope?)"
Write-Host ""
Write-Host "    2. Route to ChatScreen after download"
Write-Host ""
Write-Host "    3. Tap 'Generate' button on emulator/device"
Write-Host "       - If 'Failed to initialize model context' --> pivot to Option C"
Write-Host "         (FFI direct via package:cactus/src/services/bindings.dart)"
Write-Host "       - Coherent text response = Option A loader path confirmed [OK]"
Write-Host ""
Write-Host "    4. Routing-probe line below response should read:"
Write-Host "         'Hybrid mode: available (OpenRouter-only) | Cloud-routing path: ...'"
Write-Host ""
Write-Host "  Hot-reload keys (foreground): r=reload, R=restart, q=quit"
Write-Host ""

flutter run --dart-define=HF_TOKEN=$env:HF_TOKEN_RUNTIME 2>&1 | Tee-Object -FilePath $LogFile
