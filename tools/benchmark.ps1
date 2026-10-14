param([string]$ModelDirectory = '.cache', [int]$Size = 640, [int]$Iterations = 10, [string[]]$Backends = @('cpu', 'vulkan'), [string]$Output = 'artifacts/benchmarks.json')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    dotnet build -c Release
    if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
    $records = @()
    foreach ($model in @('yolov8n', 'yolov9t', 'yolov10n', 'yolo11n')) {
        foreach ($backend in $Backends) {
            $lines = & dotnet 'src/SimdYolo.Cli/bin/Release/net10.0/SimdYolo.Cli.dll' bench (Join-Path $ModelDirectory "$model.pt") $backend $Size $Iterations
            if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $model / $backend" }
            $record = (($lines | Select-Object -Skip 1) -join [Environment]::NewLine) | ConvertFrom-Json
            $record | Add-Member -NotePropertyName Backend -NotePropertyValue $lines[0]
            $records += $record
            Write-Host "$model / $backend : $($record.MedianMs) ms"
        }
    }
    $destination = [IO.Path]::GetFullPath($Output)
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    ConvertTo-Json -InputObject $records -Depth 8 | Set-Content -LiteralPath $destination -Encoding UTF8
} finally { Pop-Location }
