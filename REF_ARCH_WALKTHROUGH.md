# Caption 神器 重構導讀 (Services & Processors 門面與處理器架構)

本專案已成功從傳統的單一 Worker 模式重構為 **「Services & Processors」** 分層架構。這項改動提升了代碼的可維護性、可重用性，並簡化了 `caption.py` 的邏輯。

## 1. 架構核心概念

### A. Services (門面與底層服務)
位於 `lib/services/`，負責與底層模型 (WD14, LLM, OCR) 或外部 API 交互。
- `tagger.py`: 封裝 `imgutils` 的標籤識別。
- `llm.py`: 封裝 OpenAI 相容 API 的對話生成，包含圖片 Base64 編碼。
- `common.py`: 資源管理，如 `unload_all_models`。

### B. Processors (業務處理單元)
位於 `lib/processors/`，封裝了單一圖片的完整處理流程（輸入處理 + 調用 Service + 更新 Sidecar/結果）。
- `TaggerProcessor`: 執行標籤識別並更新 sidecar。
- `LLMProcessor`: 執行自然語言生成。
- `UnmaskProcessor`: 執行去背景。
- `TextMaskProcessor`: 執行文字偵測與遮蔽。
- `RestoreProcessor`: 執行還原。

### C. Workers (並行執行引擎)
位於 `lib/workers/`。
- `GenericBatchWorker`: 一個通用的執行緒類別，接受一組 `ImageContext` 和一個 `Processor`，並在後台逐一處理。現在不論是單張圖片還是批量操作，都統一使用這個引擎。

### D. Data & Context (數據封裝)
位於 `lib/data.py`。
- `ImageContext`: 封裝圖片路徑、Sidecar 數據與圖片對象緩存。這確保了所有處理器使用一致的數據格式。

## 2. 代碼清理與優化

- **`caption.py` 瘦身**: 移除了數千行冗餘的 Worker 定義與工具函式，將其遷移至 `lib/` 模組中。
- **統一 UI 反饋**: 單圖操作現在也享有進度條提示，且與批量操作共享相同的錯誤處理邏輯。
- **增強的 AppSettings**: 引入了類型安全的配置讀寫。

## 3. 如何擴充功能

若要新增功能（例如：美學評價 Aesthetic Scoring）：
1. 在 `lib/services/` 建立對應的 Service 封裝 API 或模型調用。
2. 在 `lib/processors/` 建立對應的 Processor 並繼承 `BaseProcessor`。
3. 在 `MainWindow` 中實例化 `GenericBatchWorker` 並傳入你的新 Processor 即可。

---
*重構完成時間：2026-01-16*
*維護者：Antigravity*
