@echo off
chcp 65001 > nul
echo ==========================================
echo       Caption Tool 更新腳本
echo ==========================================

echo [1/4] 執行 Git Pull...
git pull

echo [2/4] 啟動虛擬環境...
if not exist venv (
    echo 虛擬環境不存在，建立中...
    python -m venv venv
)
call venv\Scripts\activate
if errorlevel 1 goto :error

echo [3/4] 更新 Python 依賴套件...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || (
    echo 鏡像安裝失敗，改用官方 PyPI 重試...
    pip install -r requirements.txt
)
if errorlevel 1 goto :error

echo [4/4] 同步 stable-diffusion.cpp / llama.cpp GPU Runtime...
call "%~dp0setup.bat" --refresh-runtime
if errorlevel 1 goto :error

echo 更新完成！
pause
exit /b 0

:error
echo 更新失敗，請查看上方錯誤訊息。
pause
exit /b 1
