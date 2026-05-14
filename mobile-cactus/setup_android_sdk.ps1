# SolarHive -- Android SDK + AVD bootstrap for the mobile-cactus app.
#
# Idempotent -- re-runs detect existing state and skip completed stages.
#
# Stage 1: Download Android cmdline-tools to $ANDROID_HOME (D:\YSL\Android\Sdk)
# Stage 2: Install platform-tools, android-34 platform, ARM64 system image,
#          emulator, build-tools 34.0.0 (~5 GB)
# Stage 3: Accept licenses
# Stage 4: Create SolarHive_Pixel7_ARM64 AVD (config.ini disk bumped to 16 GB
#          so the 6.94 GB Cactus artifact + Android OS overhead fit)
# Stage 5: Verification (flutter doctor + flutter emulators)
#
# Companion: run_emulator.ps1 launches flutter run after this completes.

$ErrorActionPreference = 'Continue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# --- Constants (match $env:ANDROID_HOME / $env:ANDROID_AVD_HOME conventions) ---
$SdkRoot       = "D:\YSL\Android\Sdk"
$AvdHome       = "D:\YSL\Android\avd"
$AvdName       = "SolarHive_Pixel7_ARM64"
$ApiLevel      = "android-34"
$SystemImage   = "system-images;android-34;google_apis;arm64-v8a"
$DeviceProfile = "pixel_7"
$DiskSizeGB    = 16

# Stable Google CDN URL. If retired, find current at:
# https://developer.android.com/studio#command-line-tools-only
$CmdlineToolsUrl = "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip"

$tmpDir              = "D:\YSL\tmp"
$cmdlineToolsZip     = "$tmpDir\cmdline-tools.zip"
$cmdlineToolsRoot    = "$SdkRoot\cmdline-tools"
$sdkmanager          = "$cmdlineToolsRoot\latest\bin\sdkmanager.bat"
$avdmanager          = "$cmdlineToolsRoot\latest\bin\avdmanager.bat"

# --- Stage 0: pre-flight ---
Write-Host "=== Stage 0: pre-flight ===" -ForegroundColor Cyan
$d = Get-PSDrive D -ErrorAction SilentlyContinue
if (-not $d) {
    Write-Host "  ERROR: D: drive not found" -ForegroundColor Red
    exit 1
}
$freeGB = [math]::Round($d.Free / 1GB, 1)
if ($freeGB -lt 25) {
    Write-Host "  WARN: Only $freeGB GB free on D: -- recommend 25+ GB for full chain" -ForegroundColor Yellow
} else {
    Write-Host "  D: free space: $freeGB GB OK" -ForegroundColor Green
}

if ($env:ANDROID_HOME -ne $SdkRoot) {
    Write-Host "  WARN: `$env:ANDROID_HOME = '$env:ANDROID_HOME' but script targets '$SdkRoot'" -ForegroundColor Yellow
    Write-Host "        Setting for this session only."
    $env:ANDROID_HOME = $SdkRoot
}
if ($env:ANDROID_AVD_HOME -ne $AvdHome) {
    Write-Host "  WARN: `$env:ANDROID_AVD_HOME = '$env:ANDROID_AVD_HOME' but script targets '$AvdHome'" -ForegroundColor Yellow
    $env:ANDROID_AVD_HOME = $AvdHome
}

# --- Stage 1: cmdline-tools ---
if (Test-Path $sdkmanager) {
    Write-Host ""
    Write-Host "=== Stage 1: cmdline-tools (already present, skipping) ===" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "=== Stage 1: download + extract cmdline-tools ===" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

    if (-not (Test-Path $cmdlineToolsZip)) {
        Write-Host "  Downloading from $CmdlineToolsUrl"
        Write-Host "  ~120 MB, 1-3 min on broadband"
        try {
            Invoke-WebRequest -Uri $CmdlineToolsUrl -OutFile $cmdlineToolsZip -UseBasicParsing
        } catch {
            Write-Host "  Download failed: $_" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  ZIP already at $cmdlineToolsZip -- re-using"
    }

    $extractTemp = "$tmpDir\cmdline-tools-extract"
    if (Test-Path $extractTemp) { Remove-Item $extractTemp -Recurse -Force }
    Write-Host "  Extracting to $extractTemp"
    Expand-Archive -Path $cmdlineToolsZip -DestinationPath $extractTemp -Force

    # Google's tools insist on $SDK_ROOT/cmdline-tools/latest/ -- exact layout
    New-Item -ItemType Directory -Force -Path $cmdlineToolsRoot | Out-Null
    if (Test-Path "$cmdlineToolsRoot\latest") {
        Remove-Item "$cmdlineToolsRoot\latest" -Recurse -Force
    }
    Move-Item -Path "$extractTemp\cmdline-tools" -Destination "$cmdlineToolsRoot\latest"
    Remove-Item $extractTemp -Recurse -Force

    if (Test-Path $sdkmanager) {
        Write-Host "  sdkmanager installed at $sdkmanager" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: sdkmanager not found at $sdkmanager after extract" -ForegroundColor Red
        exit 1
    }
}

# --- Stage 1.5: JDK (sdkmanager is a Java app and needs a JDK) ---
$JdkRoot = "D:\YSL\jdk-17"
$JdkUrl  = "https://aka.ms/download-jdk/microsoft-jdk-17-windows-x64.zip"
$jdkBin  = Get-ChildItem -Path $JdkRoot -Directory -ErrorAction SilentlyContinue |
           Select-Object -First 1 |
           ForEach-Object { Join-Path $_.FullName "bin\java.exe" }

if ($jdkBin -and (Test-Path $jdkBin)) {
    Write-Host ""
    Write-Host "=== Stage 1.5: JDK (already present, skipping) ===" -ForegroundColor Green
    Write-Host "  java.exe: $jdkBin"
    $env:JAVA_HOME = Split-Path -Parent (Split-Path -Parent $jdkBin)
    $env:Path = "$env:JAVA_HOME\bin;$env:Path"
} else {
    Write-Host ""
    Write-Host "=== Stage 1.5: download + extract Microsoft OpenJDK 17 ===" -ForegroundColor Cyan
    Write-Host "  ~180 MB, 1-3 min on broadband"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
    $jdkZip = "$tmpDir\jdk-17.zip"
    if (-not (Test-Path $jdkZip)) {
        try {
            Invoke-WebRequest -Uri $JdkUrl -OutFile $jdkZip -UseBasicParsing
        } catch {
            Write-Host "  JDK download failed: $_" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  ZIP already at $jdkZip -- re-using"
    }

    if (Test-Path $JdkRoot) { Remove-Item $JdkRoot -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $JdkRoot | Out-Null
    Write-Host "  Extracting to $JdkRoot"
    Expand-Archive -Path $jdkZip -DestinationPath $JdkRoot -Force

    # Microsoft JDK ZIP unpacks to jdk-17.x.x.x-hotspot/ inside $JdkRoot
    $jdkSub = Get-ChildItem -Path $JdkRoot -Directory | Select-Object -First 1
    if (-not $jdkSub) {
        Write-Host "  ERROR: JDK extract did not produce a subdirectory in $JdkRoot" -ForegroundColor Red
        exit 1
    }
    $env:JAVA_HOME = $jdkSub.FullName
    $env:Path = "$env:JAVA_HOME\bin;$env:Path"
    Write-Host "  JAVA_HOME set to: $env:JAVA_HOME" -ForegroundColor Green

    # Confirm java works
    $javaVersion = & "$env:JAVA_HOME\bin\java.exe" -version 2>&1 | Select-Object -First 1
    Write-Host "  $javaVersion"
}

# Persist JAVA_HOME for future sessions (User scope, no admin needed)
$existingJavaHome = [System.Environment]::GetEnvironmentVariable('JAVA_HOME', 'User')
if ($existingJavaHome -ne $env:JAVA_HOME) {
    [System.Environment]::SetEnvironmentVariable('JAVA_HOME', $env:JAVA_HOME, 'User')
    Write-Host "  Persisted JAVA_HOME to user environment ($env:JAVA_HOME)" -ForegroundColor Green
}

# --- Helper: license-accept via Start-Process -RedirectStandardInput ---
# (PowerShell's pipe to a Java native binary sends UTF-16, which sdkmanager
# does not read correctly. Reading from an ASCII file via -RedirectStandardInput
# bypasses that.)
$yFile = "$tmpDir\android_yes.txt"
[System.IO.File]::WriteAllText($yFile, (("y`r`n") * 100), [System.Text.UTF8Encoding]::new($false))

function Invoke-LicenseAccept {
    Write-Host "  Accepting licenses via Start-Process -RedirectStandardInput..."
    $p = Start-Process -FilePath $sdkmanager -ArgumentList "--licenses" `
                       -RedirectStandardInput $yFile `
                       -RedirectStandardOutput "$tmpDir\sdkmgr_lic.out" `
                       -RedirectStandardError  "$tmpDir\sdkmgr_lic.err" `
                       -NoNewWindow -Wait -PassThru
    if (Test-Path "$tmpDir\sdkmgr_lic.out") {
        $tail = Get-Content "$tmpDir\sdkmgr_lic.out" -Tail 5
        $tail | ForEach-Object { Write-Host "    $_" }
    }
    return $p.ExitCode
}

# --- Stage 2: pre-accept licenses + install packages ---
Write-Host ""
Write-Host "=== Stage 2a: pre-accept SDK base licenses ===" -ForegroundColor Cyan
Invoke-LicenseAccept | Out-Null

Write-Host ""
Write-Host "=== Stage 2b: install SDK packages (first pass) ===" -ForegroundColor Cyan
Write-Host "  Packages:"
Write-Host "    - platform-tools"
Write-Host "    - platforms;$ApiLevel"
Write-Host "    - $SystemImage"
Write-Host "    - emulator"
Write-Host "    - build-tools;34.0.0"
Write-Host "  ~5 GB total, 5-15 min depending on bandwidth"
Write-Host ""

& $sdkmanager `
    "platform-tools" `
    "platforms;$ApiLevel" `
    $SystemImage `
    "emulator" `
    "build-tools;34.0.0"

if ($LASTEXITCODE -ne 0) {
    Write-Host "  First-pass sdkmanager exit code $LASTEXITCODE -- some packages may have been blocked by license prompts" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Stage 3: accept any newly-surfaced licenses ===" -ForegroundColor Cyan
Invoke-LicenseAccept | Out-Null

Write-Host ""
Write-Host "=== Stage 3b: install SDK packages (second pass to fill gaps) ===" -ForegroundColor Cyan
& $sdkmanager `
    "platform-tools" `
    "platforms;$ApiLevel" `
    $SystemImage `
    "emulator" `
    "build-tools;34.0.0"

if ($LASTEXITCODE -ne 0) {
    Write-Host "  Second-pass sdkmanager exited with code $LASTEXITCODE" -ForegroundColor Red
    Write-Host "  Common causes: network proxy, Defender exclusion not effective, expired URL" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

# Verify the system image actually landed (Stage 4 needs it)
$sysImagePath = "$SdkRoot\system-images\android-34\google_apis\arm64-v8a"
if (-not (Test-Path $sysImagePath)) {
    Write-Host "  ERROR: System image not found at $sysImagePath" -ForegroundColor Red
    Write-Host "  Listing installed packages:"
    & $sdkmanager --list_installed
    exit 1
}
Write-Host "  System image present: $sysImagePath" -ForegroundColor Green

# --- Stage 4: create AVD ---
Write-Host ""
Write-Host "=== Stage 4: create AVD $AvdName ===" -ForegroundColor Cyan
# When -p is passed to `avdmanager create avd`, the on-disk directory uses
# the literal -p path (no `.avd` extension). avdmanager also writes a
# sibling `<name>.ini` pointer at the parent dir.
$avdPath = "$AvdHome\$AvdName"
if (Test-Path $avdPath) {
    Write-Host "  AVD already exists at $avdPath -- skipping create" -ForegroundColor Green
} else {
    New-Item -ItemType Directory -Force -Path $AvdHome | Out-Null
    Write-Host "  Creating $AvdName with system image $SystemImage"

    # Multiple 'no' + empty lines to satisfy any sequence of interactive prompts
    # (e.g., "Do you wish to create a custom hardware profile? [no]" plus any
    # follow-up). cmd /c with file-piped stdin avoids both PowerShell's UTF-16
    # stdin encoding for native exes AND Start-Process's quoting of the
    # semicolon-bearing system-image path.
    $noFile = "$tmpDir\android_no.txt"
    [System.IO.File]::WriteAllText($noFile, ("no`r`n" * 10), [System.Text.UTF8Encoding]::new($false))

    $avdLog = "$tmpDir\avdmgr_run.log"
    $avdCmdLine = 'type "{0}" | "{1}" create avd -n {2} -k "{3}" -d {4} -p "{5}" --force' -f `
                  $noFile, $avdmanager, $AvdName, $SystemImage, $DeviceProfile, "$AvdHome\$AvdName"
    Write-Host "  cmd: $avdCmdLine"
    & cmd /c "$avdCmdLine > `"$avdLog`" 2>&1"
    $avdExit = $LASTEXITCODE

    if (Test-Path $avdLog) {
        Write-Host "  avdmanager output:"
        Get-Content $avdLog | ForEach-Object { Write-Host "    $_" }
    }

    if ($avdExit -ne 0) {
        Write-Host "  avdmanager exit code $avdExit" -ForegroundColor Red
    }

    if (-not (Test-Path $avdPath)) {
        Write-Host "  ERROR: AVD create did not produce $avdPath" -ForegroundColor Red
        exit 1
    }

    # Confirm the sibling .ini pointer also exists (sanity check)
    if (-not (Test-Path "$AvdHome\$AvdName.ini")) {
        Write-Host "  WARN: Expected pointer file $AvdHome\$AvdName.ini not found" -ForegroundColor Yellow
    }
}

# --- Stage 4b: AVD config tuning (idempotent, runs whether AVD was just
# created or already existed). Defaults from the Pixel 7 device profile are
# too small for the 6.94 GB Cactus artifact and on-device LLM workload.
$configIni = "$avdPath\config.ini"
if (Test-Path $configIni) {
    $tuneTargets = @{
        'disk.dataPartition.size' = "${DiskSizeGB}G"  # default 800M -> 16G
        'hw.ramSize'              = "4096M"            # default 1536M -> 4G
        'vm.heapSize'             = "512M"             # default 228M  -> 512M
        'sdcard.size'             = "2048 MB"          # default 512 MB -> 2 GB
    }
    $cfg = Get-Content $configIni
    foreach ($key in $tuneTargets.Keys) {
        $val = $tuneTargets[$key]
        $found = $false
        $cfg = $cfg | ForEach-Object {
            if ($_ -match "^$([regex]::Escape($key))=") {
                $found = $true
                "${key}=${val}"
            } else {
                $_
            }
        }
        if (-not $found) {
            $cfg += "${key}=${val}"
        }
    }
    $cfg | Set-Content $configIni -Encoding ASCII
    Write-Host "  AVD config.ini tuned: $($tuneTargets.Keys -join ', ')" -ForegroundColor Green
} else {
    Write-Host "  WARN: $configIni not found -- AVD config not tuned" -ForegroundColor Yellow
}

# --- Stage 5: verification ---
Write-Host ""
Write-Host "=== Stage 5: verification ===" -ForegroundColor Cyan
flutter doctor 2>&1 | Select-String -Pattern "Android|Connected device" -Context 0,3
Write-Host ""
Write-Host "  Available emulators:"
flutter emulators 2>&1

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Next: .\run_emulator.ps1 to launch the on-device smoke test"
Write-Host ""
Write-Host "  Before running, set the HF read-scope token for this session:"
Write-Host '    $env:HF_TOKEN_RUNTIME = "hf_xxxxxxxxxxxxxxxxxxxxx"'
Write-Host ""
Write-Host "  Generate at: https://huggingface.co/settings/tokens"
Write-Host "  (Read-only scope is sufficient -- the repo is private until submission day.)"
