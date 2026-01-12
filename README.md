# Caption 神器 🖼️✨

AI 驅動的圖片標註工具，專為機器學習訓練資料集設計。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

## 功能特色

### 🏷️ 智慧標籤
- **WD14 Tagger** - 自動生成圖片標籤
- **Batch Tagger** - 批次處理整個資料夾
- **標籤管理** - 支援資料夾/Meta/自訂標籤
- **中英對照** - 自動載入 Tags.csv 翻譯

### 🤖 LLM 描述生成
- **OpenRouter API** 整合 (支援各種 LLM 模型)
- **自然語言描述** - 生成英文句子 + 中文翻譯
- **Default/Custom Prompt** - 雙模板切換
- **NL 歷史** - 保留多次生成結果

### 🎨 圖片處理工具
- **Remove Background** - 一鍵去背 (transparent_background)
- **Batch Unmask** - 批次去除含 `background` 標籤的圖片背景
- **Stroke Eraser** - 手繪橡皮擦，塗抹區域變透明
- **Mask Text (OCR)** - 自動偵測文字區塊並遮罩

### 📝 文字編輯
- **即時儲存** - 編輯 .txt 自動同步
- **Token 計數** - CLIP Tokenizer 精確計算
- **Find/Replace** - 支援正則表達式批次取代
- **智慧插入** - 游標位置插入標籤，自動格式化

---

## 安裝

### 1. 建立虛擬環境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 安裝依賴
```bash
pip install PyQt6 Pillow natsort openai
pip install imgutils[ocr]       # WD14 Tagger + OCR
pip install transparent-background  # 去背功能 (選用)
pip install transformers        # Token 計數 (選用)
```

### 3. 執行
```bash
python caption.py
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
- **LLM** - API Key、Model、Prompt 模板
- **Tagger** - WD14 閾值、模型選擇
- **Text** - 英文強制小寫
- **Mask** - 預設透明度、格式、OCR 開關

---

## 檔案結構

```
your_dataset/
├── image1.webp
├── image1.txt           # 標註文字
├── image1.boorutag      # (可選) Booru 元資料
├── image1.tagger.txt    # Tagger 快取
├── image1.nl.txt        # LLM 生成歷史
├── .custom_tags.json    # 資料夾自訂標籤
├── no_used/             # 刪除的檔案
├── unmask/              # 去背前的原圖
└── masked/              # Mask 前的原圖
```

---

## 截圖

> (待補充)

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

## 作者

Made with ❤️ for AI image training.
