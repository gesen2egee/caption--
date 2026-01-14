@echo off
chcp 65001 > nul
echo ==========================================
echo       Caption Tool 環境建置腳本
echo ==========================================

echo [1/6] 檢查/建立虛擬環境...
if not exist venv (
    echo 建立虛擬環境中...
    python -m venv venv
) else (
    echo 虛擬環境已存在。
)
call venv\Scripts\activate

echo [1.5/6] 優先安裝 GPU 版 PyTorch (CUDA 12.1)...
echo 這一步驟非常重要，請耐心等待下載...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [2/6] 安裝基礎套件 (PyQt6, Pillow, OpenAI)...
pip install "numpy<2" PyQt6 Pillow natsort openai -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [3/6] 安裝 Pilmoji (從 GitHub)...
pip install git+https://github.com/jay3332/pilmoji.git

echo [4/6] 安裝 dghs-imgutils[gpu] (WD14/OCR) 及 GPU 依賴...
pip install dghs-imgutils[gpu] nvidia-cudnn-cu12 nvidia-cublas-cu12 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [5/6] 安裝去背與其他工具...
pip install transparent-background transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [6/6] 完成！
echo 請使用 run.bat 啟動程式。
pause
