# 固定旧版主库，先完成全部构建，再串行执行新进程 ABBA 对照；Python 仅为开发计时控制器。
param(
    [Parameter(Mandatory = $true)][string]$BaselineDirectory,
    [string]$OutputDirectory = '.cache/startup-0.11.3',
    [string]$RuntimeIdentifier = 'win-x64'
)
$ErrorActionPreference = 'Stop'
$repository = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repository
$baseline = (Resolve-Path -LiteralPath $BaselineDirectory).Path
if (!(Test-Path -LiteralPath (Join-Path $baseline 'SimdYolo.dll'))) { throw '旧版目录缺少 SimdYolo.dll。' }
$destination = [IO.Path]::GetFullPath((Join-Path $repository $OutputDirectory))
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$normal = Join-Path $destination 'il'
$ready = Join-Path $destination 'ready-to-run'
if ([IO.Path]::GetFullPath((Join-Path $baseline 'SimdYolo.dll')) -in @((Join-Path $normal 'SimdYolo.dll'), (Join-Path $ready 'SimdYolo.dll'))) { throw '基线不能放在本次构建输出目录。' }
dotnet build src/SimdYolo.Cli -c Release -o $normal --nologo
if ($LASTEXITCODE -ne 0) { throw '普通 IL 构建失败。' }
dotnet publish src/SimdYolo.Cli -c Release -r $RuntimeIdentifier -p:PublishProfile=FastStartup -o $ready --nologo
if ($LASTEXITCODE -ne 0) { throw 'ReadyToRun 发布失败。' }
$input = Join-Path $repository '.cache/image-validation/input.f32'
if (!(Test-Path -LiteralPath $input)) { throw '请先准备相同的640×640 RGB NCHW FP32真实图片输入，见文档。' }
$settings = @('--models', 'yolov8n', 'yolo26n-seg', 'yolo11n-pose', '--repeats', '2', '--duration', '3', '--warmup', '60', '--input', $input)
python tools/compare-lifecycle.py --baseline-dir $baseline --candidate-dir $normal @settings --output (Join-Path $destination 'source.json')
if ($LASTEXITCODE -ne 0) { throw '源码优化对照失败；保留已完成窗口。' }
python tools/compare-lifecycle.py --baseline-dir $normal --candidate-dir $ready @settings --output (Join-Path $destination 'publish.json')
if ($LASTEXITCODE -ne 0) { throw '同源码发布方式对照失败；保留已完成窗口。' }
Write-Output ('完成。原始首帧/末帧及哈希保存在：' + $destination)
