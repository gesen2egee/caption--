# Current Status Progress

更新時間：2026-04-07

## 總覽

這份文件是目前專案重構與 UI 對齊工作的狀態快照，目的是讓後續不必再從聊天紀錄回推目前做到哪裡。

## 1. 目前整體進度

### Runtime / Backend 重構
- 完成度：約 95%+
- 狀態：
  - 已有 headless runtime host
  - command / event / state / capabilities / agent manifest 已成形
  - backend reload / watch 已存在
  - worker service lifecycle 已存在
  - regression / feature smoke 已存在

### Web UI 對齊 Qt
- 完成度：約 75% 到 80%
- 狀態：
  - 主布局已固定為左圖右欄
  - 分頁、右欄 tagger、文字區、menu bar 都已對齊一大段
  - responsive 行為已補三段 breakpoint
  - 預覽區已修正窄寬度下的顯示不全
  - 右欄 tab/action row/文字區密度已再收一輪

### Agent / UI 擷取與比對
- 完成度：約 85%+
- 狀態：
  - web UI 內建 Agent 擷取面板
  - 可擷取工作區、目前分頁、全部分頁、預覽區、右欄
  - 後端會保存 PNG + JSON metadata
  - 支援 `window.__captionAgentCapture.*`
  - 已改成標準化 desktop baseline capture，較不受 live viewport / browser zoom 干擾

## 2. 這輪工作區的重點變更

### 前端
- `frontend/src/App.tsx`
  - 新增 Agent 擷取面板
  - 擷取摘要輸出
  - menu 入口
- `frontend/src/specRenderer.tsx`
  - 右欄 tagger / tab 結構更像 Qt
- `frontend/src/styles.css`
  - Qt-like 版面與 breakpoint 收斂
  - 右欄密度與文字編輯區調整
  - capture sandbox / normalized baseline styles
- `frontend/src/agentCapture.ts`
  - 新增 capture 核心
  - 支援全部分頁與標準化 baseline capture
- `frontend/runtime/client.ts`
  - 新增 `saveUiCaptures`
- `frontend/runtime/types.ts`
  - 新增 capture 型別

### 後端
- `lib/runtime/http_bridge.py`
  - 新增 `POST /captures/ui`
  - 保存 PNG / JSON 到 `output/ui-captures/`

### 文件
- `README.md`
  - 已補 Runtime / Regression / Web Agent 擷取說明
- `tasks/feature_inventory_and_test_plan.md`
  - 已補 Agent capture 與 visual diff 流程

## 3. 已有的實測基線

目前工作區裡已有最近的 web UI 基線圖：

- `output/playwright/web-ui-1600.png`
- `output/playwright/web-ui-1366.png`
- `output/playwright/web-ui-1180-v2.png`
- `output/playwright/web-ui-right-panel-refined.png`

這些圖可以當成：
- 版面回歸比對基準
- 後續細節優化前後的參考

## 4. 現在最值得做的事

### 必做
- 把 web UI 剩餘細節再往 Qt 靠：
  - menu / dropdown 的桌面手感
  - quick dialog header / footer
  - 小控制項高度一致性

### 建議做
- 做一個固定的 UI smoke / capture runner
  - 自動起 service
  - 自動載入測試資料夾
  - 自動輸出 1600 / 1366 / 1180 三組基準圖
  - 自動跑 `captureAllTabs()`

### 可延後
- 真正做 visual diff 報告
  - 比對上一版與本版
  - 自動標記新增 / 位移 / 裁切差異

## 5. 目前最大的剩餘風險

- README / 文件雖然已經補很多，但還需要持續整理，避免重構過程留下重複段落
- web UI 雖然骨架已穩，但仍有部分「像 Qt」主要靠 CSS 對齊，還沒做到完全 1:1
- Agent capture 已經能用，但缺一個固定的一鍵 runner 來量產基準圖

## 6. 建議之後的工作節奏

如果接下來繼續往前推，建議每一輪都做完整一包，而不是只做單點：

1. 調整一組 UI 細節
2. build
3. 產生 3 組 viewport 基準圖
4. 用 Agent capture 跑全部分頁
5. 更新 status 文件

這樣後面就不會再需要一直回頭追「到底哪一版比較好」。
