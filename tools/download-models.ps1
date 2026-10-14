param([string]$Directory = '.cache/models', [switch]$Vision)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Directory | Out-Null
$models = @('yolov8n', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo26n')
if ($Vision) { $models += @('yolov8n-seg', 'yolov8n-pose', 'yolo11n-seg', 'yolo11n-pose', 'yolo26n-seg', 'yolo26n-pose') }
foreach ($model in $models) {
    $destination = Join-Path $Directory "$model.pt"
    if (-not (Test-Path -LiteralPath $destination)) {
        $release = if ($model.StartsWith('yolo26')) { 'v8.4.0' } else { 'v8.3.0' }
        Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/$release/$model.pt" -OutFile $destination
    }
    Get-FileHash -LiteralPath $destination -Algorithm SHA256 | Select-Object Path, Hash
}
