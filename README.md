# Caption 神器 🖼️✨
<img width="2377" height="1245" alt="image" src="https://github.com/user-attachments/assets/800d3514-792a-4ace-b8cf-c0260602e7f1" />



AI 驅動的圖片標註工具，專為機器學習訓練資料集設計。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

## 功能特色

### 🏷️ 智慧標籤
- **WD14 Tagger** - 自動生成圖片標籤
- **Batch Tagger** - 批次處理整個資料夾
- **Batch Tagger to txt** - 批次將標籤直接寫入實體 `.txt` 檔案 (含過濾與格式化)
- **標籤管理** - 支援資料夾/Meta/自訂標籤
- **特徵標籤 (Character Tags)** - 自動識別並高亮 (紅框)，支援黑白名單過濾
- **中英對照** - 自動載入 Tags.csv 翻譯

### 🤖 LLM 描述生成
- **OpenRouter API** 整合 (支援各種 LLM 模型)
- **LLaMA.cpp Local Worker** - 支援 GGUF 本地模型 / Hugging Face URL
- **自然語言描述** - 生成英文句子 + 中文翻譯
- **Default/Custom Prompt** - 雙模板切換
- **NL 歷史** - 保留多次生成結果
- **Batch LLM to txt** - 批次將 NL 描述直接寫入實體 `.txt` 檔案

### 🎨 圖片處理工具
- **FLUX.2-klein Image Edit** - 使用 `stable-diffusion.cpp / sd-server` 做本地修圖
- **Remove Background** - 一鍵去背 (transparent_background)
- **Batch Unmask** - 批次去除含 `background` 標籤的圖片背景
- **Stroke Eraser** - 手繪橡皮擦，塗抹區域變透明
- **Mask Text (OCR)** - 自動偵測文字區塊並遮罩

### 📝 文字編輯
- **即時儲存** - 編輯 .txt 自動同步
- **Token 計數** - CLIP Tokenizer 精確計算
- **Find/Replace** - 支援正則表達式批次取代
- **智慧插入** - 游標位置插入標籤，自動格式化
- **多國語言 (I18n)** - 支援繁體中文與英文介面即時切換
- **日夜間模式 (Theme)** - 支援 Light / Dark 模式切換

---

## 安裝

### 1. 快速安裝 (Windows)
雙擊執行 `setup.bat` 即可自動建立虛擬環境並安裝所有依賴，另外會：

- 下載 `stable-diffusion.cpp` Windows GPU Release 到 `tasks/runtime/stable-diffusion-cpp/`
- 預下載 FLUX.2-klein 修圖所需資產到 `tasks/runtime/models/flux2-klein/`
- 保留 `llama.cpp / llama-server` 給 Qwen LLM 視覺描述使用

### 2. 啟動
雙擊執行 `run.bat`。

### 3. 更新
若需更新程式碼、依賴與 `stable-diffusion.cpp` runtime，請執行 `update.bat`。

---

### 手動安裝 (Advanced)
若不使用 bat 檔，請依序執行：
```bash
python -m venv venv
venv\Scripts\activate

# 基礎依賴
pip install PyQt6 Pillow natsort openai llama-cpp-python huggingface-hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# 若鏡像無 llama-cpp-python，可回退官方 PyPI
pip install llama-cpp-python huggingface-hub

# Pilmoji (Source fixed)
pip install git+https://github.com/jay3332/pilmoji.git

# Main Utils
pip install dghs-imgutils[gpu] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transparent-background transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 使用說明

### 基本流程
1. **File → Open Directory** 選擇圖片資料夾
2. 左右鍵/滾輪瀏覽圖片
3. 在 **TAGS** 分頁點選標籤加入 .txt
4. 或使用 **Auto Tag** / **Run LLM** 自動生成

### 快捷鍵
| 按鍵 | 功能 |
|------|------|
| `←` `→` | 上/下一張圖 |
| `PageUp` `PageDown` | 上/下一張圖 |
| `Delete` | 移動圖片到 no_used |
| 滾輪 (圖片區) | 瀏覽圖片 |

### 設定 (Settings)
- **LLM** - Provider、API Key、Model、Prompt 模板
- **LLaMA.cpp** - GGUF 模型路徑/URL、n_ctx、n_gpu_layers、max_tokens
- **LLaMA.cpp Vision** - 可選 mmproj 路徑，未填時會嘗試從同一個 Hugging Face repo 自動抓取
- **Image Edit / stable-diffusion.cpp** - `sd-server` URL、自動啟動、模型路徑、steps、guidance、seed、extra args
- **Tagger** - WD14 閾值、模型選擇
- **Text** - 英文強制小寫、自動格式化、Batch 寫入模式 (附加/覆寫)、資料夾觸發詞
- **Tags Filter** - 特徵標籤黑白名單 (Prefixes/Suffixes/Words)
- **Mask** - 預設透明度、格式、OCR 開關、舊圖移動時刪除對應 npz
- **UI (介面)** - 語言切換、日夜間模式切換

> LLaMA.cpp 預設 GGUF URL：  
> `https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q8_0.gguf`

> FLUX.2-klein 預設 Diffusion GGUF URL：  
> `https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/blob/main/flux-2-klein-4b-BF16.gguf`

> Image Edit 預設 `sd-server` URL：  
> `http://127.0.0.1:8001/v1`

## 本地 C++ 後端分工

- `Qwen LLM` 使用 `llama.cpp / llama-server`
  - 預設 URL: `http://127.0.0.1:8000/v1`
- `FLUX.2-klein 修圖` 使用 `stable-diffusion.cpp / sd-server`
  - 預設 URL: `http://127.0.0.1:8001/v1`

這樣兩套 server 會各自使用自己的 port，避免互相搶占。

如果你想接外部 server：

- 到 Settings 把 `Base URL` 改成你自己的 endpoint
- 程式會先檢查該 URL 是否已可用
- 若可用，就直接重用外部 server，不會強制重啟它
- 若不可用，且有開啟 autostart，才會啟動內建 managed server

---

## 檔案結構

```
your_dataset/
├── image1.webp
├── image1.txt           # 標註文字
├── image1.json          # Tagger/NL/Mask 記錄 (整合)
├── image1.boorutag      # (可選) Booru 元資料
├── .custom_tags.json    # 資料夾自訂標籤
├── no_used/             # 刪除的檔案
└── unmask/              # 去背/Mask 前的原圖
```

### JSON Sidecar 結構
```json
{
  "tagger_tags": "rating:general, 1girl, ...",
  "nl_pages": ["LLM 生成結果 1", "LLM 生成結果 2"],
  "masked_background": true,
  "masked_text": false
}
```

---

## 依賴套件

| 套件 | 用途 | 必要 |
|------|------|------|
| PyQt6 | GUI | ✅ |
| Pillow | 圖片處理 | ✅ |
| natsort | 自然排序 | ✅ |
| openai | LLM API | ✅ |
| imgutils | WD14 Tagger | ✅ |
| transparent-background | 去背 | ❌ |
| transformers | Token 計數 | ❌ |

---

## License

MIT License

---

## 特別感謝

- **[deepghs](https://github.com/deepghs)** / **[narugo1992](https://github.com/narugo1992)** / **[imgutils](https://github.com/deepghs/imgutils)**：提供強大的底層圖片處理工具。

Made with ❤️ for AI image training.
