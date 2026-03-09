# FLUX.2-klein stable-diffusion.cpp 整合

## 待辦
- [x] 補齊 image-process 專用設定欄位與預設值
- [x] 將 `image_flux2_klein_gguf_local` worker 改為 `sd-server` HTTP client
- [x] 保留舊 worker id 相容舊設定
- [x] 新增 stable-diffusion.cpp server 自動啟停與外部 URL 重用邏輯
- [x] 新增 diffusion / vae / llm 三組模型資產設定
- [x] 更新 Settings UI，加入 image-process server 與模型欄位
- [x] 更新安裝腳本，自動下載 stable-diffusion.cpp GPU Release 與模型資產
- [x] 更新 README，說明 `llama-server` 與 `sd-server` 的分工與 port
- [x] 執行語法/匯入驗證
- [x] 執行設定載入與 worker smoke test

## 驗證紀錄
- [x] `python -m compileall` 通過
- [x] 設定物件可正常載入新欄位
- [x] Image worker 可成功建立並解析設定
- [x] SettingsDialog 可建立並回存 image-process 新欄位
- [x] 假 `sd-server` `/v1/images/edits` smoke test 可完成覆寫原圖流程

## 審核
- compileall 通過，無語法錯誤
- smoke test 驗證了既有 `/v1/models` endpoint 可被重用，且 autostart 關閉時會回傳預期錯誤
- 假 `sd-server` 驗證了 multipart 上傳、回圖解碼、覆寫原圖可正常運作
- 未執行實際 `setup.bat` 下載與真實 `sd-server` 端到端修圖，因為會下載大型 runtime / 模型資產
