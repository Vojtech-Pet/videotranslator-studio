# VideoTranslator Studio launcher (Windows PowerShell)
# Override: $env:VTS_PYTHON, $env:VTS_CONDA_ROOT, $env:VTS_PYTHON_ENV

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Load .env if present
$envFile = Join-Path $ScriptDir ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#=]+)=(.+)$") {
            Set-Item -Path "Env:$($Matches[1].Trim())" -Value $Matches[2].Trim()
        }
    }
}

# Locate Python interpreter
$python = $null
if ($env:VTS_PYTHON -and (Test-Path $env:VTS_PYTHON)) {
    $python = $env:VTS_PYTHON
} else {
    $envName = if ($env:VTS_PYTHON_ENV) { $env:VTS_PYTHON_ENV } else { "videotranslator" }
    $candidates = @(
        $env:VTS_CONDA_ROOT,
        "$HOME\miniforge3",
        "$HOME\anaconda3",
        "$HOME\miniconda3",
        "C:\miniforge3",
        "C:\ProgramData\miniforge3"
    )
    foreach ($root in $candidates) {
        if ($root -and (Test-Path "$root\envs\$envName\python.exe")) {
            $python = "$root\envs\$envName\python.exe"
            break
        }
    }
    if (-not $python -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $python = (Get-Command python).Source
        Write-Host "[launch] WARN: conda env not found, using system python: $python" -ForegroundColor Yellow
    }
}

if (-not $python) {
    Write-Host "[launch] ERROR: Python not found." -ForegroundColor Red
    Write-Host "  Run install.ps1 first or set `$env:VTS_PYTHON" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Set-Location $ScriptDir
& $python "$ScriptDir\VideoTranslator_studio.py" $args
