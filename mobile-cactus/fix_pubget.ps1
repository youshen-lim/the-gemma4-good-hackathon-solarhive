# SolarHive — flutter pub get unblocker for mobile-cactus
#
# Stage 1: Bypass flutter.bat by calling dart.exe directly. flutter.bat acquires
#          a global lock (flutter.bat.lock) that may be the source of the hang.
#          dart pub get does the same dependency resolution without the wrapper.
#
# Stage 2: If Stage 1 hangs, fall back to sideload of cactus 1.3.0 from the
#          pre-extracted tarball at D:\YSL\tmp\cactus-1.3.0.

$ErrorActionPreference = 'Continue'

$proj = "D:\YSL\projects\mobile-cactus"
$flutterBin = "D:\YSL\flutter\bin"
$dartExe = "$flutterBin\cache\dart-sdk\bin\dart.exe"

if (-not (Test-Path $dartExe)) {
    Write-Host "ERROR: dart.exe not found at $dartExe" -ForegroundColor Red
    exit 1
}

Write-Host "=== Cleanup zombie processes ===" -ForegroundColor Cyan
Get-Process -Name dart, flutter -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("  Killing PID {0} ({1})" -f $_.Id, $_.Name)
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
$lock = "$flutterBin\cache\flutter.bat.lock"
if (Test-Path $lock) {
    Remove-Item $lock -Force
    Write-Host "  Removed flutter.bat.lock"
}

Set-Location $proj
$env:PUB_CACHE = "D:\YSL\pub-cache"
$env:Path = "$flutterBin;$env:Path"

Write-Host ""
Write-Host "=== STAGE 1: dart pub get (bypassing flutter.bat) ===" -ForegroundColor Yellow

$logOut = "$proj\dart_pubget.log"
$logErr = "$proj\dart_pubget.err"
if (Test-Path $logOut) { Remove-Item $logOut -Force }
if (Test-Path $logErr) { Remove-Item $logErr -Force }

$start = Get-Date
$proc = Start-Process -FilePath $dartExe `
    -ArgumentList "pub", "get", "--verbose" `
    -WorkingDirectory $proj `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -PassThru -NoNewWindow

$timeoutSec = 300
$pollIntervalSec = 5
$noProgressTicks = 0
$lastSize = 0
$hung = $false

while (-not $proc.HasExited) {
    $elapsed = ((Get-Date) - $start).TotalSeconds
    if ($elapsed -gt $timeoutSec) {
        Write-Host ("  TIMEOUT after {0}s - killing dart" -f $timeoutSec) -ForegroundColor Red
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $hung = $true
        break
    }

    $currentSize = if (Test-Path $logOut) { (Get-Item $logOut).Length } else { 0 }
    if ($currentSize -eq $lastSize) {
        $noProgressTicks++
    } else {
        $noProgressTicks = 0
        $lastSize = $currentSize
    }

    Write-Host ("  [{0}s] dart PID {1}, log {2} bytes, idle ticks: {3}" -f [math]::Round($elapsed,0), $proc.Id, $currentSize, $noProgressTicks)

    if ($noProgressTicks -ge 18 -and $currentSize -eq 0) {
        Write-Host "  HUNG at 0 bytes for 90s - killing dart" -ForegroundColor Red
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $hung = $true
        break
    }

    Start-Sleep -Seconds $pollIntervalSec
}

$elapsedFinal = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)

if (-not $hung -and $proc.ExitCode -eq 0) {
    Write-Host ""
    Write-Host ("=== STAGE 1 SUCCESS in {0}s ===" -f $elapsedFinal) -ForegroundColor Green
    Write-Host ""
    Write-Host "Tail of log:"
    Get-Content $logOut -Tail 20
    Write-Host ""
    Write-Host "=== Verification ==="
    $cactus = Get-ChildItem "$env:PUB_CACHE\hosted\pub.dev" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "cactus-*" }
    if ($cactus) {
        Write-Host ("  cactus package: {0}" -f $cactus.FullName) -ForegroundColor Green
    }
    if (Test-Path "$proj\.dart_tool") { Write-Host "  .dart_tool: created" -ForegroundColor Green }
    if (Test-Path "$proj\pubspec.lock") { Write-Host "  pubspec.lock: created" -ForegroundColor Green }
    exit 0
}

if (-not $hung) {
    Write-Host ""
    Write-Host ("=== STAGE 1 FAILED with exit code {0} ===" -f $proc.ExitCode) -ForegroundColor Red
    Write-Host "STDOUT:"
    if (Test-Path $logOut) { Get-Content $logOut -Raw }
    Write-Host ""
    Write-Host "STDERR:"
    if (Test-Path $logErr) { Get-Content $logErr -Raw }
    Write-Host ""
    Write-Host "Read the error above. Stage 2 only helps with hangs, not real errors."
    exit 1
}

Write-Host ""
Write-Host "=== STAGE 2: Sideload cactus 1.3.0 from pre-extracted tarball ===" -ForegroundColor Yellow

$cactusSource = "D:\YSL\tmp\cactus-1.3.0"
$cactusDest = "D:\YSL\pub-cache\hosted\pub.dev\cactus-1.3.0"

if (-not (Test-Path $cactusSource)) {
    Write-Host ("ERROR: {0} not found." -f $cactusSource) -ForegroundColor Red
    Write-Host "Re-run the tarball download first:"
    Write-Host '  Invoke-WebRequest "https://pub.dev/api/archives/cactus-1.3.0.tar.gz" -OutFile D:\YSL\tmp\cactus-1.3.0.tar.gz'
    Write-Host '  tar -xzf D:\YSL\tmp\cactus-1.3.0.tar.gz -C D:\YSL\tmp\cactus-1.3.0'
    exit 1
}

Write-Host ("Copying {0} -> {1}" -f $cactusSource, $cactusDest)
if (Test-Path $cactusDest) {
    Write-Host "  Destination exists - removing first"
    Remove-Item $cactusDest -Recurse -Force
}
New-Item -ItemType Directory -Path "D:\YSL\pub-cache\hosted\pub.dev" -Force | Out-Null
Copy-Item -Path $cactusSource -Destination $cactusDest -Recurse -Force
Write-Host "  cactus 1.3.0 sideloaded"

Write-Host ""
Write-Host "=== Retry dart pub get (cactus pre-staged, fetching transitive deps) ==="
$logOut2 = "$proj\dart_pubget_stage2.log"
$logErr2 = "$proj\dart_pubget_stage2.err"
if (Test-Path $logOut2) { Remove-Item $logOut2 -Force }
if (Test-Path $logErr2) { Remove-Item $logErr2 -Force }

$proc2 = Start-Process -FilePath $dartExe `
    -ArgumentList "pub", "get" `
    -WorkingDirectory $proj `
    -RedirectStandardOutput $logOut2 `
    -RedirectStandardError $logErr2 `
    -PassThru -NoNewWindow -Wait

$exitCode = $proc2.ExitCode
Write-Host ("Exit code: {0}" -f $exitCode)
Write-Host ""
Write-Host "STDOUT:"
if (Test-Path $logOut2) { Get-Content $logOut2 -Raw } else { "(empty)" }
Write-Host ""
Write-Host "STDERR:"
if (Test-Path $logErr2) { Get-Content $logErr2 -Raw } else { "(empty)" }

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "=== STAGE 2 SUCCESS - sideload + retry worked ===" -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "=== STAGE 2 FAILED ===" -ForegroundColor Red
Write-Host @'

If the second attempt also hung, pub itself cannot reach pub.dev despite
the Test-NetConnection passing. Possible causes:
  - Corporate proxy / VPN intercepting TLS
  - HTTPS_PROXY env var needed (check: $env:HTTPS_PROXY)
  - Defender exclusion not effective for child processes
  - Some other AV product running alongside Defender

Last resort: download all transitive dep tarballs manually from pub.dev.
Packages needed: dio, path_provider, shared_preferences, cupertino_icons,
flutter_lints, plus their transitive deps (~30-50 total).
'@
exit 1
