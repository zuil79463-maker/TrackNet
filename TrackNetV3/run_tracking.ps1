param(
    [string]$Camera = 'realsense',
    [string]$Serial,
    [int]$Width = 1280,
    [int]$Height = 720,
    [int]$Fps = 30,
    [ValidateSet('auto','dshow','msmf')][string]$Backend = 'auto',
    [ValidateSet('auto','torch','onnx')][string]$Engine = 'auto',
    [string]$Csv,
    [switch]$Headless,
    [int]$MaxResults = 0,
    [switch]$ListDevices
)
$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) { throw 'Missing .venv. See README.md.' }
$predictArgs = @((Join-Path $PSScriptRoot 'predict.py'), '--source', $Camera,
    '--backend', $Backend, '--engine', $Engine, '--max-results', $MaxResults,
    '--width', $Width, '--height', $Height, '--fps', $Fps)
if ($Serial) { $predictArgs += @('--serial', $Serial) }
if ($ListDevices) { $predictArgs += '--list-devices' }
if ($Csv) { $predictArgs += @('--csv', $Csv) }
if ($Headless) { $predictArgs += '--headless' }
& $pythonExe @predictArgs
if ($LASTEXITCODE -ne 0) { throw "Camera tracking failed with exit code $LASTEXITCODE" }
