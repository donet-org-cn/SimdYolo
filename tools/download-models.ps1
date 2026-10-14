param([string]$Directory = '.cache/models', [switch]$Vision)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Directory | Out-Null
$models = @('yolov8n', 'yolov9t', 'yolov10n', 'yolo11n')
if ($Vision) { $models += @('yolov8n-seg', 'yolov8n-pose', 'yolo11n-seg', 'yolo11n-pose') }
foreach ($model in $models) {
    $destination = Join-Path $Directory "$model.pt"
    if (-not (Test-Path -LiteralPath $destination)) {
        Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/$model.pt" -OutFile $destination
    }
    Get-FileHash -LiteralPath $destination -Algorithm SHA256 | Select-Object Path, Hash
}
