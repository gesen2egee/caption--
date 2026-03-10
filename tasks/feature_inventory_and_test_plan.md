# Feature Inventory And Test Plan

## Goal

整理目前專案已存在的功能面，並把後續全面測試與除錯流程收斂成一份可執行的基準。

這份文件面向三種使用方式：
- 手動測試
- Agent / headless runtime 測試
- 問題發生後的定位與除錯

## 1. 現有功能總表

### 1.1 使用者功能

#### A. 檔案與導覽
- 開啟資料夾
- 圖片列表載入與自然排序
- 上一張 / 下一張 / 第一張 / 最後一張
- 跳到指定 index
- 依 tags / txt 內容過濾
- 清除過濾
- 刪除目前圖片到 `no_used/`
- 顯示目前圖片、txt、json、raw backup 狀態

對應入口：
- PyQt UI
- runtime commands:
  - `selection.open_directory`
  - `selection.set_root_dir`
  - `selection.apply_filter`
  - `selection.clear_filter`
  - `selection.prev`
  - `selection.next`
  - `selection.first`
  - `selection.last`
  - `selection.jump_to_index`
  - `image.delete_current`

#### B. Tagger
- 單張 Auto Tag
- Batch Tagger
- Batch Tagger to txt
- Tagger tags 寫入 `.json`
- Tagger tags 寫入 `.txt`
- Character tag 過濾
- custom tags 疊加
- folder/meta tags 同步

對應入口：
- `action.run_tagger_current`
- `batch.run_tagger`
- `task.run_tagger`
- `task.run_tagger_loaded`

主要實作：
- `lib/pipeline/tasks/tagger_task.py`
- `lib/workers/tagger_imgutils_generic.py`
- `lib/workers/tagger_imgutils_tagging_local.py`

#### C. LLM / NL 描述
- 單張 Run LLM
- Batch Run LLM
- OpenRouter / 遠端 VLM 路徑
- `llama.cpp` 本地 GGUF 路徑
- default / custom prompt 切換
- NL page 歷史
- LLM 結果寫入 `.json`
- LLM 結果寫入 `.txt`
- context / tags prompt 注入
- thinking mode / sampling 參數

對應入口：
- `action.run_llm_current`
- `batch.run_llm`
- `task.run_llm`
- `task.run_llm_loaded`
- `prompt.use_default`
- `prompt.use_custom`
- `nl.prev_page`
- `nl.next_page`

主要實作：
- `lib/pipeline/tasks/llm_task.py`
- `lib/workers/vlm_openrouter_api.py`
- `lib/workers/llm_llama_cpp_local.py`

#### D. 圖片處理
- Image Process / FLUX.2-klein edit
- Remove Background / Unmask
- Mask Text (OCR)
- Restore from raw backup
- Stroke Eraser

對應入口：
- `action.run_image_process_current`
- `action.run_unmask_current`
- `action.run_mask_text_current`
- `action.run_restore_current`
- `action.run_stroke_eraser_current`
- `batch.run_image_process`
- `batch.run_unmask`
- `batch.run_mask_text`
- `batch.run_restore`

主要實作：
- `lib/pipeline/tasks/image_process_task.py`
- `lib/pipeline/tasks/unmask_task.py`
- `lib/pipeline/tasks/mask_text_task.py`
- `lib/pipeline/tasks/restore_task.py`
- `lib/workers/image_flux2_klein_gguf_local.py`
- `lib/workers/mask_transparent_background_local.py`
- `lib/workers/mask_text_local.py`
- `lib/workers/detect_imgutils_ocr_local.py`
- `lib/workers/image_restore_raw.py`

#### E. 文字編輯
- `.txt` 即時讀寫
- token 計數
- 自動格式化
- 自動小寫
- find / replace
- regex find / replace
- undo / redo
- 自訂 tag 加入

對應入口：
- `content.set_txt_content`
- `editor.find_replace`
- `editor.undo`
- `editor.redo`
- `tags.add_custom`

#### F. 設定與 UI
- UI language 切換
- theme 切換
- LLM 設定
- llama.cpp 設定
- image process 設定
- tagger 設定
- mask / OCR 設定
- worker runtime mode 設定
- prompt 文字設定
- view mode / active tab / control 值同步

對應入口：
- `settings.get`
- `settings.update`
- `settings.schema`
- `content.set_prompt_text`
- `content.set_image_process_prompt_text`
- `ui.set_view_mode`
- `ui.set_active_tab`
- `ui.set_control_value`

### 1.2 Runtime / Agent / Dev 功能

#### A. Runtime inspection
- runtime state
- runtime events
- task status
- capabilities
- agent manifest
- event profiles
- UI spec / override

對應 commands：
- `app.get_runtime_state`
- `app.get_runtime_events`
- `app.get_task_status`
- `app.get_capabilities`
- `app.get_agent_manifest`
- `app.get_event_profiles`
- `app.get_ui_spec`
- `app.get_ui_spec_override`

#### B. Runtime UI mutation
- UI spec patch
- UI spec override replace
- UI spec reset

對應 commands：
- `app.patch_ui_node`
- `app.update_ui_spec`
- `app.replace_ui_spec_override`
- `app.reset_ui_spec`

#### C. Bridge / streaming
- bridge start / stop / status
- SSE event stream
- event profile filtering

對應 commands：
- `bridge.start`
- `bridge.stop`
- `bridge.get_status`

#### D. Reload / hot update
- 列出 reloadable runtime services
- backend service reload
- backend auto-reload watcher
- worker service reload / stop
- reload policy recommendation

對應 commands：
- `backend.list_reloadables`
- `backend.get_reload_policy`
- `backend.reload_services`
- `backend.watch_reload_start`
- `backend.watch_reload_status`
- `backend.watch_reload_stop`
- `workers.services_status`
- `workers.services_reload`
- `workers.services_stop`

#### E. 測試 / 診斷
- smoke command
- runtime regression harness
- command tracking
- structured errors
- safe / development access gating

對應 commands：
- `test.smoke_command`
- `test.run_runtime_regression`

### 1.3 底層 worker / provider 面

目前 repo 實作的 worker 模組：
- `tagger_imgutils_generic`
- `tagger_imgutils_tagging_local`
- `vlm_openrouter_api`
- `llm_llama_cpp_local`
- `image_flux2_klein_gguf_local`
- `mask_transparent_background_local`
- `mask_text_local`
- `detect_imgutils_ocr_local`
- `image_restore_raw`
- `text_filter_lists`

注意：
- 真正「當下可用」的 worker 會受依賴、環境、啟動掃描狀態影響。
- 測試前應先跑：
  - `workers.scan`
  - `workers.list`

## 2. 測試分層設計

### L0. Static / build gate

目的：
- 防止語法錯、匯入錯、前端 build 壞掉

必跑：
- `python -m compileall lib`
- `cd frontend && npm run build`

通過條件：
- 無 syntax/import error
- 前端 build 成功

### L1. Runtime smoke

目的：
- 確認 runtime host、command registry、state、event、bridge 沒壞

必跑：
- `python scripts/runtime_regression.py --json`
- `test.smoke_command`
- `app.get_capabilities`
- `app.get_runtime_state`

通過條件：
- runtime regression 通過
- command 數量正常
- state section 齊全

### L2. Command smoke by feature area

目的：
- 每個功能區至少有一條 command path 可以在 headless 模式通過

範圍：
- selection
- content/editor
- tagger
- llm
- image_process
- unmask
- mask_text
- restore
- runtime reload
- worker service lifecycle

通過條件：
- command 可執行
- result 結構化
- event/state 同步正確

### L3. Real-model end-to-end

目的：
- 確認真模型、真檔案、副作用、sidecar、備份資料夾都正確

原則：
- 一律對暫存副本跑，不直接動原始資料夾
- 每類功能至少一張真圖

通過條件：
- `.txt` / `.json` / `unmask/` / `no_used/` 等輸出符合預期
- `task.image_done` 成功
- 無結構化錯誤

### L4. Reload / service lifecycle

目的：
- 確認 backend reload、worker restart、watcher defer 行為不會破壞執行中任務

通過條件：
- `backend.reload_services` 可成功
- `workers.services_reload` 可成功
- `workers.services_stop` 後可重新 invoke
- pid 會變
- active task 時 reload 會 defer

### L5. Fault injection / recovery

目的：
- 驗證錯誤處理是否可定位、可回報、可恢復

建議故障注入：
- 無效 worker name
- 無效 base URL
- server port 被占用
- OCR 關閉時跑 mask text
- 沒 raw backup 時 restore
- context size 不足
- prompt 缺 tags / confirm flag 缺失

通過條件：
- 都回穩定 `error_info.code`
- 不應只剩 free-form exception

## 3. 測試資料設計

### P0 最小資料集

用途：
- 命令 smoke / selection / editor / reload

內容：
- `a.png` + `a.txt`
- `b.png` + `b.txt`

已存在：
- `scripts/runtime_regression.py` 內建 fixture

### P1 真實單圖資料

用途：
- Tagger / LLM / command 追蹤 / sidecar 驗證

建議固定基準圖：
- `E:\\NE\\20_miss valentine\\Generated Image November 28, 2025 - 2_28AM.webp`

說明：
- 這張圖已實測過 tagger + llama.cpp local 路徑
- 測試時應先複製到 temp 目錄

### P2 OCR / Mask Text 資料

用途：
- OCR 偵測
- 文字遮罩
- 文字移除後輸出驗證

需求：
- 圖中有明顯對話框 / 字幕 / 嵌字

### P3 Background / Unmask 資料

用途：
- remove background
- batch unmask 條件判斷

需求：
- 一張含角色前景
- 一張明顯 scenery / background-only 圖

### P4 Restore / delete / backup 資料

用途：
- restore
- delete current
- raw backup / unmask folder 邏輯

需求：
- 先做過 unmask 或 stroke eraser，留下可 restore 狀態

## 4. 全面測試矩陣

| 範圍 | 入口 | 測試資料 | 關鍵斷言 | 常見故障 | 首要定位點 |
|---|---|---|---|---|---|
| Runtime 基線 | `test.run_runtime_regression` | P0 | command/state/event 正常 | command 少、state 缺 section | `lib/runtime/*` |
| 檔案開啟與導覽 | `selection.set_root_dir` 等 | P0 / P1 | image_count、current_index、filter 正常 | 篩選沒反映、索引錯 | `selection_service.py`, `navigation_mixin.py` |
| Txt 編輯 | `content.set_txt_content`, `editor.find_replace` | P0 | txt 寫回磁碟、find/replace 回結構化結果 | regex 行為錯、autosave 漏寫 | `editor_service.py` |
| Tagger 單張 | `action.run_tagger_current` | P1 | `.json` 有 `tagger_tags`、`.txt` 有 tags | worker 不可用、sidecar 不更新 | `tagger_task.py`, tagger workers |
| Tagger 批次 | `batch.run_tagger` | P1/P2/P3 folder | 批次完成、每張輸出一致 | skip/filter 邏輯錯 | `batch_service.py`, `tagger_task.py` |
| LLM 單張 | `action.run_llm_current` | P1 | `.json.nl_pages`、`llm_result`、`.txt` 正常 | server 未就緒、context 不足、prompt 注入錯 | `llm_task.py`, `llm_llama_cpp_local.py`, `vlm_openrouter_api.py` |
| LLM 批次 | `batch.run_llm` | P1/P2 folder | 批量 NL 正常、confirm 邏輯正確 | `{tags}` / 角色名確認卡住 | `batch_service.py`, `processing_service.py` |
| Image Process | `action.run_image_process_current` | P2 | 圖片被替換或輸出正確、raw backup 正常 | sd-server 不可用、prompt 沒帶入 | `image_process_task.py`, `image_flux2_klein_gguf_local.py` |
| Unmask | `action.run_unmask_current` | P3 | 背景去除成功、`unmask/` 備份正確 | worker 缺依賴、輸出透明錯 | `unmask_task.py`, `mask_transparent_background_local.py` |
| Mask Text | `action.run_mask_text_current` | P2 | OCR 偵測成功、遮罩輸出正確 | OCR 參數不對、無文字卻誤遮 | `mask_text_task.py`, `detect_imgutils_ocr_local.py`, `mask_text_local.py` |
| Restore | `action.run_restore_current` | P4 | 可恢復原圖 | `no_backup` | `restore_task.py`, `image_restore_raw.py` |
| Stroke Eraser | `action.run_stroke_eraser_current` | P4 + mask | 根據 mask 正確輸出透明區域 | mask base64 / path 解析錯 | `processing_service.py` |
| Delete current | `image.delete_current` | P4 | 檔案進 `no_used/`，sidecar 同步搬移 | 搬移一半、txt/json 遺漏 | `selection_service.py` |
| Settings | `settings.update` | P0 | app state / runtime state / task config 同步 | 設定寫入不生效 | `app_state.py`, `state_adapter.py`, `settings_mixin.py` |
| UI/runtime spec | `app.patch_ui_node` 等 | P0 | spec patch 可套用與 reset | override 殘留 | `ui_spec.py`, `runtime_api.py` |
| Worker lifecycle | `workers.services_*` | P0 | reload/stop/restart 正常 | pid 不變、service 卡死 | `service_manager.py`, `service_process.py` |
| Backend reload | `backend.reload_services` | P0 | reloadable service 可重載 | active task reload 污染狀態 | `service_reload.py`, `service_watch.py` |

## 5. 除錯 SOP

### Step 1. 先看 capability 與環境

先收集：
- `app.get_capabilities`
- `workers.list`
- `workers.services_status`
- `settings.get`
- `bridge.get_status`

目的：
- 先確認不是 worker 根本沒載入、mode 錯、access mode 錯

### Step 2. 用 temp copy 重現

規則：
- 永遠不要直接對原始資料夾做破壞性除錯
- 一律複製到 temp 目錄後再跑

特別適合：
- LLM
- image process
- unmask
- mask text
- restore
- delete current

### Step 3. 走 command，不走 widget

優先用：
- `action.*`
- `batch.*`
- `selection.*`
- `settings.*`

不要先從 PyQt 點點點開始追，因為那會把 UI 問題和 backend 問題混在一起。

### Step 4. 同時看三個面

每次出錯都收三份資訊：
- command result
- runtime state
- runtime events

最關鍵事件：
- `command.started`
- `command.finished`
- `command.failed`
- `task.started`
- `task.progress`
- `task.image_done`
- `task.batch_done`
- `worker.*`
- `backend.*`

### Step 5. 依錯誤層級分流

#### A. command/input 層
特徵：
- 直接在 command call 就失敗

優先檢查：
- command kwargs
- access mode
- structured error code

#### B. task orchestration 層
特徵：
- command 成功 start，但 task 中途失敗

優先檢查：
- `task.started`
- `task.image_done.error_info`
- task settings 組裝

#### C. worker/service 層
特徵：
- task 啟動了，但 worker 回失敗

優先檢查：
- `worker_runtime_mode`
- `workers.services_status`
- `worker error taxonomy`
- 對應 server / model log

#### D. file side-effect 層
特徵：
- 顯示成功，但 `.txt` / `.json` / 輸出圖不對

優先檢查：
- temp 目錄輸出
- sidecar 寫入
- `selection` state
- `tags` / `content` state

### Step 6. 套用 reload policy

不要盲目重開。

先問：
- `backend.get_reload_policy(changed_path=...)`

原則：
- `lib/runtime/*service.py` 變更：優先 `backend.reload_services`
- `lib/workers/*` 變更：`worker_runtime_mode=service` 時優先 `workers.services_reload`
- `lib/pipeline/tasks/*` 深層變更：目前仍優先 `restart_host`

## 6. 建議的固定測試套件

### P0 每次改動都跑
- `python -m compileall lib`
- `cd frontend && npm run build`
- `python scripts/runtime_regression.py --json`
- `python scripts/feature_smoke_matrix.py --image "<real_image>" --json`

### P1 每次改動 LLM / prompt / server 都跑
- P1 真圖 temp copy
- `action.run_tagger_current`
- `action.run_llm_current`
- 檢查：
  - `.json.tagger_tags`
  - `.json.nl_pages`
  - `.json.llm_result`
  - `.txt`
  - `task.image_done.error_info`

### P1 每次改動 mask / image process 都跑
- `action.run_image_process_current`
- `action.run_mask_text_current`
- `action.run_unmask_current`
- `action.run_restore_current`

### P2 每次改動 runtime / reload / worker service 都跑
- `backend.reload_services`
- `workers.services_reload`
- `workers.services_stop`
- `test.run_runtime_regression`

## 7. 目前自動化覆蓋與缺口

### 已有
- runtime surface regression
- selection/filter flow
- settings update
- ui spec patch/reset
- pipeline callback flow
- worker missing error taxonomy
- worker service lifecycle
- reload policy recommendation

### 還缺
- 真 Tagger E2E 自動回歸
- 真 LLM E2E 自動回歸
- OCR / Mask Text 真圖回歸
- Image Process 真 server 回歸
- delete / restore / raw backup 一致性回歸
- active task + deferred reload 回歸

## 8. 推薦下一步

若要把「全面測試」真正落地成可重複執行，下一階段最值得補三個自動化腳本：
- `scripts/feature_smoke_tagger_llm.py`
- `scripts/feature_smoke_mask_restore.py`
- `scripts/feature_reload_stress.py`

原則：
- 吃 `--image` 或 `--root-dir`
- 一律複製到 temp fixture
- 全部輸出 JSON summary
- summary 要包含：
  - input
  - settings snapshot
  - command result
  - task events
  - artifacts
  - structured errors

## 9. 現有逐項 smoke 腳本

已新增：
- `scripts/feature_smoke_matrix.py`

目前會逐項執行：
- `runtime_regression`
- `worker_inventory`
- `selection_editor`
- `tagger_llm_real_image`
- `delete_current_bundle`
- `heavy_feature_preflight`
- `unmask_real_image`
- `mask_text_restore_fixture`
- `stroke_restore_real_image`
- `image_process_real_image`

建議用法：
- `python scripts/feature_smoke_matrix.py --image "E:\\NE\\20_miss valentine\\Generated Image November 28, 2025 - 2_28AM.webp" --json`

目前已知結果：
- `tagger_llm_real_image`：通過
- `unmask_real_image`：通過
- `mask_text_restore_fixture`：通過
- `stroke_restore_real_image`：通過
- `image_process_real_image`：已改為使用假 `sd-server` 驗證 `/v1/models` + `/v1/images/edits` client/task 流程，因此不必等待大型 runtime 下載也能通過 smoke
