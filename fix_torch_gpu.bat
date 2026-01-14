@echo off
chcp 65001 >nul
echo 正在檢查並修復 venv 中的 PyTorch GPU 支援...

if not exist venv (
    echo [ERROR] 未找到 venv 資料夾，請確認您在正確的目錄下執行。
    pause
    exit /b
)

echo 啟動虛擬環境...
call venv\Scripts\activate

echo 移除現有的 torch torchvision torchaudio...
pip uninstall -y torch torchvision torchaudio

echo 正在安裝支援 CUDA 11.8 的 PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 安裝完成。請重新啟動程式並觀察輸出。
echo 可以在 Python 中輸入以下指令測試:
echo import torch; print(torch.cuda.is_available())
echo.
pause
