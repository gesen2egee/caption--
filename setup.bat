@echo off
setlocal EnableExtensions
chcp 65001 > nul
echo ==========================================
echo       Caption Tool 環境建置腳本
echo ==========================================

set "PROJECT_ROOT=%~dp0"
set "VENV_PY=%PROJECT_ROOT%venv\Scripts\python.exe"
set "RUNTIME_ONLY=0"
if /I "%~1"=="--refresh-runtime" set "RUNTIME_ONLY=1"

if "%RUNTIME_ONLY%"=="0" (
    echo [1/7] 檢查/建立虛擬環境...
    if not exist "%PROJECT_ROOT%venv" (
        echo 建立虛擬環境中...
        python -m venv "%PROJECT_ROOT%venv"
    ) else (
        echo 虛擬環境已存在。
    )
    call "%PROJECT_ROOT%venv\Scripts\activate"
    if errorlevel 1 goto :error

    echo [2/7] 優先安裝 GPU 版 PyTorch (CUDA 11.8)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 goto :error

    echo [3/7] 安裝 Python 依賴...
    pip install -r "%PROJECT_ROOT%requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple || pip install -r "%PROJECT_ROOT%requirements.txt"
    if errorlevel 1 goto :error

    echo [4/7] 安裝 Pilmoji (從 GitHub)...
    pip install git+https://github.com/jay3332/pilmoji.git
    if errorlevel 1 goto :error

    echo [5/7] 補齊 imgutils GPU 依賴...
    pip install dghs-imgutils[gpu] nvidia-cudnn-cu12 nvidia-cublas-cu12 -i https://pypi.tuna.tsinghua.edu.cn/simple || pip install dghs-imgutils[gpu] nvidia-cudnn-cu12 nvidia-cublas-cu12
    if errorlevel 1 goto :error
) else (
    echo [runtime] 使用 --refresh-runtime，跳過 Python 依賴安裝。
)

echo [6/7] 下載 stable-diffusion.cpp GPU Runtime...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$projectRoot = [System.IO.Path]::GetFullPath('%PROJECT_ROOT%');" ^
  "$runtimeRoot = Join-Path $projectRoot 'tasks\runtime\stable-diffusion-cpp';" ^
  "$downloadRoot = Join-Path $runtimeRoot '_download';" ^
  "New-Item -ItemType Directory -Force -Path $runtimeRoot, $downloadRoot | Out-Null;" ^
  "$headers = @{ 'User-Agent' = 'CaptionSetup' };" ^
  "$release = Invoke-RestMethod -Headers $headers -UseBasicParsing 'https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest';" ^
  "$asset = $release.assets | Where-Object { $_.name -eq 'cudart-sd-bin-win-cu12-x64.zip' } | Select-Object -First 1;" ^
  "if (-not $asset) { $asset = $release.assets | Where-Object { $_.name -like '*win-cuda12-x64.zip' } | Select-Object -First 1 };" ^
  "if (-not $asset) { throw '找不到 stable-diffusion.cpp Windows CUDA Release 資產。' };" ^
  "$versionDir = Join-Path $runtimeRoot $release.tag_name;" ^
  "$zipPath = Join-Path $downloadRoot $asset.name;" ^
  "if (-not (Test-Path (Join-Path $versionDir 'sd-server.exe'))) {" ^
  "  Write-Host ('Downloading ' + $asset.browser_download_url);" ^
  "  Invoke-WebRequest -Headers $headers -UseBasicParsing -Uri $asset.browser_download_url -OutFile $zipPath;" ^
  "  if (Test-Path $versionDir) { Remove-Item -Recurse -Force $versionDir };" ^
  "  New-Item -ItemType Directory -Force -Path $versionDir | Out-Null;" ^
  "  Expand-Archive -Path $zipPath -DestinationPath $versionDir -Force;" ^
  "} else { Write-Host ('stable-diffusion.cpp runtime already exists: ' + $versionDir) }"
if errorlevel 1 goto :error

echo [7/8] 下載 llama.cpp GPU Runtime...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$projectRoot = [System.IO.Path]::GetFullPath('%PROJECT_ROOT%');" ^
  "$runtimeRoot = Join-Path $projectRoot 'tasks\runtime\llama-b8848';" ^
  "$downloadRoot = Join-Path $runtimeRoot '_download';" ^
  "New-Item -ItemType Directory -Force -Path $runtimeRoot, $downloadRoot | Out-Null;" ^
  "$headers = @{ 'User-Agent' = 'CaptionSetup' };" ^
  "$release = Invoke-RestMethod -Headers $headers -UseBasicParsing 'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b8848';" ^
  "$cudart = $release.assets | Where-Object { $_.name -eq 'cudart-llama-bin-win-cuda-13.1-x64.zip' } | Select-Object -First 1;" ^
  "$binary = $release.assets | Where-Object { $_.name -eq 'llama-b8848-bin-win-cuda-13.1-x64.zip' } | Select-Object -First 1;" ^
  "if (-not $cudart -or -not $binary) { throw '找不到 llama.cpp Windows CUDA 13.1 Release 資產。' };" ^
  "foreach ($asset in @($cudart, $binary)) {" ^
  "  $zipPath = Join-Path $downloadRoot $asset.name;" ^
  "  Write-Host ('Downloading ' + $asset.browser_download_url);" ^
  "  Invoke-WebRequest -Headers $headers -UseBasicParsing -Uri $asset.browser_download_url -OutFile $zipPath;" ^
  "  Expand-Archive -Path $zipPath -DestinationPath $runtimeRoot -Force;" ^
  "}"
if errorlevel 1 goto :error

echo [8/8] 預下載 FLUX.2-klein 修圖資產...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$projectRoot = [System.IO.Path]::GetFullPath('%PROJECT_ROOT%');" ^
  "$modelRoot = Join-Path $projectRoot 'tasks\runtime\models\flux2-klein';" ^
  "New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null;" ^
  "$headers = @{ 'User-Agent' = 'CaptionSetup' };" ^
  "$downloads = @(" ^
  "  @{ Url='https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-BF16.gguf'; Target=(Join-Path $modelRoot 'flux-2-klein-4b-BF16.gguf') }," ^
  "  @{ Url='https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/ae.safetensors'; Target=(Join-Path $modelRoot 'ae.safetensors') }," ^
  "  @{ Url='https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf'; Target=(Join-Path $modelRoot 'Qwen3-4B-Q4_K_M.gguf') }" ^
  ");" ^
  "foreach ($item in $downloads) {" ^
  "  if (Test-Path $item.Target) { Write-Host ('Already cached: ' + $item.Target); continue };" ^
  "  Write-Host ('Downloading ' + $item.Url);" ^
  "  Invoke-WebRequest -Headers $headers -UseBasicParsing -Uri $item.Url -OutFile $item.Target;" ^
  "}"
if errorlevel 1 goto :error

echo 完成！
echo stable-diffusion.cpp runtime 已放在 tasks\runtime\stable-diffusion-cpp\
echo llama.cpp runtime 已放在 tasks\runtime\llama-b8848\
echo 模型已放在 tasks\runtime\models\flux2-klein\
if "%RUNTIME_ONLY%"=="1" exit /b 0
echo 請使用 run.bat 啟動程式。
pause
exit /b 0

:error
echo.
echo 安裝失敗，請查看上方錯誤訊息。
if "%RUNTIME_ONLY%"=="1" exit /b 1
pause
exit /b 1
