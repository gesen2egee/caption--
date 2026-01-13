@echo off
call venv\Scripts\activate

:: [GPU Fix] 強制將 pip 安裝的 NVIDIA DLL 路徑加入系統 PATH
set "SITE_PACKAGES=%~dp0venv\Lib\site-packages"
set "CUDNN_BIN=%SITE_PACKAGES%\nvidia\cudnn\bin"
set "CUBLAS_BIN=%SITE_PACKAGES%\nvidia\cublas\bin"
set "CUDNN_ROOT=%SITE_PACKAGES%\nvidia\cudnn"
set "CUBLAS_ROOT=%SITE_PACKAGES%\nvidia\cublas"

if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"
if exist "%CUBLAS_BIN%" set "PATH=%CUBLAS_BIN%;%PATH%"
:: 部分環境可能需要 lib 根目錄
if exist "%CUDNN_ROOT%" set "PATH=%CUDNN_ROOT%;%PATH%"
if exist "%CUBLAS_ROOT%" set "PATH=%CUBLAS_ROOT%;%PATH%"

python caption.py
pause
