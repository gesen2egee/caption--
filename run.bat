@echo off
setlocal

set "MODE=%~1"
if not defined MODE (
  set "MODE=qt"
)

if /I "%MODE%"=="qt" goto launch_qt
if /I "%MODE%"=="web" goto launch_web
if /I "%MODE%"=="shell" goto launch_shell
if /I "%MODE%"=="service" goto launch_service

echo [caption] Unknown mode: %MODE%
echo [caption] Usage: run.bat [qt^|web^|shell^|service]
pause
exit /b 1

:prepare_env
call venv\Scripts\activate

set "SITE_PACKAGES=%~dp0venv\Lib\site-packages"
set "CUDNN_BIN=%SITE_PACKAGES%\nvidia\cudnn\bin"
set "CUBLAS_BIN=%SITE_PACKAGES%\nvidia\cublas\bin"
set "CUDNN_ROOT=%SITE_PACKAGES%\nvidia\cudnn"
set "CUBLAS_ROOT=%SITE_PACKAGES%\nvidia\cublas"

if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"
if exist "%CUBLAS_BIN%" set "PATH=%CUBLAS_BIN%;%PATH%"
if exist "%CUDNN_ROOT%" set "PATH=%CUDNN_ROOT%;%PATH%"
if exist "%CUBLAS_ROOT%" set "PATH=%CUBLAS_ROOT%;%PATH%"
goto :eof

:launch_qt
call :prepare_env
python caption.py --ui=qt
goto finish

:launch_web
if not exist "%~dp0frontend\dist\index.html" (
  echo [caption] frontend\dist\index.html not found.
  echo [caption] Run "cd frontend && npm run build" first, or use "run.bat shell".
  pause
  exit /b 1
)
call :prepare_env
set "CAPTION_UI_MODE=web"
set "CAPTION_RUNTIME_HTTP=1"
set "CAPTION_RUNTIME_HTTP_HOST=127.0.0.1"
set "CAPTION_RUNTIME_HTTP_PORT=8765"
set "CAPTION_RUNTIME_AUTO_RELOAD_SERVICES=1"
python caption.py --ui=web --runtime-host=127.0.0.1 --runtime-port=8765 --open-browser --auto-reload-services
goto finish

:launch_shell
call :prepare_env
set "CAPTION_UI_MODE=shell"
set "CAPTION_RUNTIME_HTTP=1"
set "CAPTION_RUNTIME_HTTP_HOST=127.0.0.1"
set "CAPTION_RUNTIME_HTTP_PORT=8765"
set "CAPTION_RUNTIME_AUTO_RELOAD_SERVICES=1"
start "Caption Runtime Shell" cmd /k "cd /d %~dp0frontend && npm run dev"
start "" "http://127.0.0.1:5173/"
python caption.py --ui=shell --runtime-host=127.0.0.1 --runtime-port=8765 --auto-reload-services
goto finish

:launch_service
call :prepare_env
set "CAPTION_UI_MODE=service"
set "CAPTION_RUNTIME_HTTP=1"
set "CAPTION_RUNTIME_HTTP_HOST=127.0.0.1"
set "CAPTION_RUNTIME_HTTP_PORT=8765"
set "CAPTION_RUNTIME_AUTO_RELOAD_SERVICES=1"
python caption.py --ui=service --runtime-host=127.0.0.1 --runtime-port=8765 --auto-reload-services
goto finish

:finish
endlocal
pause
