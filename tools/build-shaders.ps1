param([string]$Compiler = 'glslangValidator')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
foreach ($shader in (Get-ChildItem -LiteralPath (Join-Path $projectRoot 'shaders') -Filter '*.comp' | Sort-Object Name)) {
    $outputPath = [System.IO.Path]::ChangeExtension($shader.FullName, '.spv')
    & $Compiler -V --target-env vulkan1.0 -o $outputPath $shader.FullName
    if ($LASTEXITCODE -ne 0) { throw 'SPIR-V compilation failed.' }
}
