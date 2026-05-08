# VideoTranslator Studio — Windows installer (PowerShell)
#
# One-line install (spusti v PowerShell ako Administrator nie je potrebný):
#   iwr -useb https://raw.githubusercontent.com/Vojtech-Pet/videotranslator-studio/main/install.ps1 | iex
#
# Override:
#   $env:VTS_DIR = "$HOME\Apps\vts"; iwr -useb ... | iex

$ErrorActionPreference = "Stop"

$VTS_DIR = if ($env:VTS_DIR) { $env:VTS_DIR } else { "$HOME\videotranslator-studio" }
$VTS_PY_VERSION = if ($env:VTS_PY_VERSION) { $env:VTS_PY_VERSION } else { "3.10" }
$VTS_ENV_NAME = if ($env:VTS_ENV_NAME) { $env:VTS_ENV_NAME } else { "videotranslator" }
$MINIFORGE_DIR = if ($env:MINIFORGE_DIR) { $env:MINIFORGE_DIR } else { "$HOME\miniforge3" }
$REPO_URL = "https://github.com/Vojtech-Pet/videotranslator-studio.git"

function Info($msg)  { Write-Host "[install] $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "[install] $msg" -ForegroundColor Yellow }
function Err($msg)   { Write-Host "[install] $msg" -ForegroundColor Red; exit 1 }

# ── 1. Sanity ─────────────────────────────────────────────────────────────────
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Err "Git nie je nainstalovany. Stiahni z https://git-scm.com/download/win"
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Warn "ffmpeg nie je v PATH. Doporuceny: winget install ffmpeg"
}

# ── 2. Miniforge3 ─────────────────────────────────────────────────────────────
$condaExe = "$MINIFORGE_DIR\Scripts\conda.exe"
if (-not (Test-Path $condaExe)) {
    Info "Instalujem Miniforge3 do $MINIFORGE_DIR"
    $installer = "$env:TEMP\miniforge3-installer.exe"
    Invoke-WebRequest "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile $installer -UseBasicParsing
    Start-Process -FilePath $installer -ArgumentList "/InstallationType=JustMe","/RegisterPython=0","/S","/D=$MINIFORGE_DIR" -Wait
    Remove-Item $installer
}

# ── 3. Conda env ──────────────────────────────────────────────────────────────
$envExists = & $condaExe env list | Select-String -Pattern "^$VTS_ENV_NAME\s"
if ($envExists) {
    Info "Env '$VTS_ENV_NAME' uz existuje, preskakujem"
} else {
    Info "Vytvaram conda env '$VTS_ENV_NAME' (Python $VTS_PY_VERSION)"
    & $condaExe create -y -n $VTS_ENV_NAME "python=$VTS_PY_VERSION" pip
}

$envPython = "$MINIFORGE_DIR\envs\$VTS_ENV_NAME\python.exe"

# ── 4. Repo ───────────────────────────────────────────────────────────────────
if (Test-Path "$VTS_DIR\.git") {
    Info "Aktualizujem repo v $VTS_DIR"
    git -C $VTS_DIR pull --ff-only
} else {
    Info "Klonujem repo do $VTS_DIR"
    New-Item -ItemType Directory -Force -Path (Split-Path $VTS_DIR) | Out-Null
    git clone $REPO_URL $VTS_DIR
}

# ── 5. Python deps ────────────────────────────────────────────────────────────
Info "Instalujem Python dependencies (5-15 min)"
& $envPython -m pip install --upgrade pip wheel
& $envPython -m pip install -r "$VTS_DIR\requirements.txt"

# ── 6. CUDA libs ──────────────────────────────────────────────────────────────
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Info "GPU detekovane, instalujem CUDA dependencies"
    try {
        & $envPython -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
    } catch {
        Warn "CUDA libs install zlyhal — VTS bude bezat na CPU"
    }
} else {
    Warn "Ziadne NVIDIA GPU — VTS bude bezat na CPU (pomalsie)"
}

# ── 7. Start Menu shortcut ────────────────────────────────────────────────────
$startMenu = [Environment]::GetFolderPath("StartMenu") + "\Programs"
$shortcutPath = "$startMenu\VideoTranslator Studio.lnk"
$WshShell = New-Object -ComObject WScript.Shell
$shortcut = $WshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "$VTS_DIR\launch_studio.ps1"
$shortcut.Arguments = ""
$shortcut.WorkingDirectory = $VTS_DIR
$shortcut.IconLocation = "$VTS_DIR\assets\icon.ico,0"
$shortcut.Description = "EN-SK video dubbing pipeline"
$shortcut.Save()
Info "Start Menu shortcut vytvoreny: $shortcutPath"

# ── 8. Launcher .env ──────────────────────────────────────────────────────────
@"
VTS_PYTHON=$envPython
VTS_CONDA_ROOT=$MINIFORGE_DIR
VTS_PYTHON_ENV=$VTS_ENV_NAME
"@ | Out-File -FilePath "$VTS_DIR\.env" -Encoding utf8

# ── 9. Paths config ───────────────────────────────────────────────────────────
$pathsCfg = "$env:USERPROFILE\.config\videotranslator"
New-Item -ItemType Directory -Force -Path $pathsCfg | Out-Null

Info "Hotovo!"
Write-Host ""
Write-Host "Spustenie:"
Write-Host "  Start Menu -> VideoTranslator Studio" -ForegroundColor Green
Write-Host "  alebo: $VTS_DIR\launch_studio.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "API kluce (volitelne, pre cloud preklad):"
Write-Host "  setx OPENAI_API_KEY sk-..."
Write-Host "  setx HF_API_KEY hf_..."
