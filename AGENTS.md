# AGENTS.md - PhysicsNeMo EV-NSFnet PINN 專案指引

## 專案概述
- **EV-NSFnet**: 使用 PINNs + EVM 的熵黏性納維-史托克斯傅立葉網路
- **問題**: Re=5000 的方腔流動，使用雙神經網路架構
- **框架**: NVIDIA PhysicsNeMo 分散式訓練與最佳化
- **開發者**: opencode + GitHub Copilot

## 建構/測試/檢查指令
- **訓練**: `python physicsnemo_train.py` 或 `./run_training.sh [num_gpus]`
- **測試**: `python physicsnemo_test.py`
- **單一測試**: 使用 `pytest physicsnemo_test.py::test_function_name` 執行特定測試
- **程式碼檢查**: `black .` 和 `isort .` (可選開發工具)
- **安裝依賴**: `pip install -r requirements.txt`

## 角色扮演
在執行專案時，請扮演一個專業工程師的視角來分析程式碼，並給出階段性計畫的建議

## Git 規則
- 不要主動git
- 檢查是否存在.gitignore文件
- 被告知上傳至github時先執行```git status```查看狀況
- 上傳至github前請先更新 @README.md 文檔

## markdwon檔案原則（此處不包含AGENTS.md）
- README.md 中必須要標示本專案使用opencode+Github Copilot開發
- 說明檔案請盡可能簡潔明瞭
- 避免建立過多的markdown文件來描述專案
- markdown文件可以多使用emoji以及豐富排版來增加豐富度

## 程式建構規則
- 程式碼以邏輯清晰、精簡、易讀、高效這四點為主
- 將各種獨立功能獨立成一個定義函數或是api檔案，並提供api文檔
- 各api檔案需要有獨立性，避免循環嵌套
- 盡量避免大於3層的迴圈以免程式效率低下
- 使用註解在功能前面簡略說明
- 若程式有輸出需求，讓輸出能一目瞭然並使用'==='或是'---'來做分隔

## 程式碼風格與慣例
- **版權聲明**: 頂部必須包含 `# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.`
- **匯入順序**: 標準庫、第三方 (torch, numpy)、本地匯入
- **型別提示**: 複雜型別使用 `from typing import List, Optional, Dict`
- **類別命名**: PascalCase (例如 `PhysicsNeMoPINNSolver`)
- **函數/變數**: snake_case (例如 `training_step`, `boundary_data`)
- **設定檔**: 使用 Hydra/OmegaConf 進行設定管理
- **文檔字串**: 公用方法需要簡短描述
- **錯誤處理**: 失敗時使用 try/except 搭配 logger.error()
- **裝置處理**: 使用 `solver.dist.device` 進行 GPU/CPU 放置
- **分散式**: 記錄/儲存前檢查 `solver.dist.rank == 0`
- **EVM 參數**: 使用 alpha_evm 作為熵黏性正則化權重

## 專案結構
- 核心求解器: `physicsnemo_solver.py` (雙網路 PINN 求解器)
- 神經網路: `physicsnemo_net.py` (主網路 + EVM 網路)
- PDE 方程式: `physicsnemo_equations.py` (納維-史托克斯 + EVM 約束)
- 資料處理: `physicsnemo_data.py` (方腔流動與邊界條件)
- 主要訓練: `physicsnemo_train.py` (6階段漸進式訓練)
- 測試: `physicsnemo_test.py` (多雷諾數驗證)

## 檔案參考
重要： 當您遇到檔案參考 (例如 @rules/general.md)，請使用你的read工具，依需要載入。它們與當前的 SPECIFIC 任務相關。

### 說明：
- 請勿預先載入所有參考資料 - 根據實際需要使用懶惰載入。
- 載入時，將內容視為覆寫預設值的強制指示
- 需要時，以遞迴方式跟蹤參照
- 回應用戶時盡量先以計畫的方式告知用戶
- 除非用戶說「請開始實作」這種直接命令，否則不要直接執行修改

## 專案規則
- 不要自動推送到 GitHub
- README.md 應註明由 opencode + GitHub Copilot 開發
- 使用遞減 alpha_evm 值的多階段訓練
- 維持主流動 + 渦黏性的雙網路架構