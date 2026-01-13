@echo off
chcp 65001 > nul
echo ==========================================
echo       Caption Tool 更新腳本
echo ==========================================

echo [1/3] 執行 Git Pull...
git pull

echo [2/3] 啟動虛擬環境...
if not exist venv (
    echo 虛擬環境不存在，建立中...
    python -m venv venv
)
call venv\Scripts\activate

echo [3/3] 更新 Python 依賴套件...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo 更新完成！
pause
