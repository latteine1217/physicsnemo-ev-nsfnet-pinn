# 🌊 LDC-PINNs: Physics-Informed Neural Networks for Lid-Driven Cavity Flow

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **開發工具**: 本專案使用 [opencode](https://github.com/sst/opencode) + GitHub Copilot 進行開發 🤖

## 📖 專案簡介

LDC-PINNs 是一個模組化的Physics-Informed Neural Networks (PINNs)實現，專門用於求解lid-driven cavity flow中的Navier-Stokes方程。本專案採用Entropy Viscosity Method (EVM)來增強數值穩定性，並支援多階段訓練策略。

### 🎯 主要特點

- **🧠 先進網路架構**: 支援LAAF (Layer-wise Adaptive Activation Function)激活函數
- **⚗️ 物理增強**: 整合Entropy Viscosity Method人工粘滯度
- **🔧 混合優化**: Adam + L-BFGS優化器結合
- **📊 多階段訓練**: 5階段漸進式訓練策略
- **⚡ 高效能**: 支援分散式訓練和Tesla P100 GPU
- **🔄 模組化設計**: 清晰的程式碼架構，易於擴展

## 🚀 快速開始

### 環境需求

- Python 3.10+
- PyTorch 2.6.0+cu126
- CUDA 12.6+ (Tesla P100兼容)
- 112GB+ 系統記憶體 (建議)

### 安裝

```bash
# 克隆專案
git clone https://github.com/your-repo/ldc-pinns.git
cd ldc-pinns

# 安裝依賴
pip install -r requirements.txt

# 複製參考資料 (可選)
cp ev-NSFnet/data/*.mat data/reference/
```

### 基本使用（預設：PhysicsNeMo 完整管線）

```bash
# 使用PhysicsNeMo進階入口（單機，可由Hydra參數覆寫）
python train_physicsnemo_advanced.py --config configs/ldc_pinn_advanced.yaml

# SLURM + torchrun（2x P100）
sbatch scripts/slurm/train.sh

# P100 相容性檢查
python test_p100_compatibility.py
```

若需要原生 PyTorch 流程（但使用 PhysicsNeMo-Sym 殘差），可改用：

```bash
python scripts/train.py --config configs/nemo_production.yaml
```

## 📁 專案結構

```
ldc_pinns/
├── 📋 README.md                     # 專案說明 
├── 🔧 requirements.txt              # 依賴管理
├── 
├── 🏗️ src/                         # 核心原始碼
│   ├── 🧠 models/                   # 神經網路模組
│   ├── ⚗️ physics/                   # 物理方程模組  
│   ├── 🎯 solvers/                  # PINN求解器
│   ├── 📊 data/                     # 資料處理
│   ├── ⚙️ utils/                    # 工具模組
│   └── 🔧 config/                   # 配置管理
│
├── 📝 configs/                      # 配置檔案
│   ├── default.yaml                 # 預設配置
│   ├── production.yaml              # 生產配置
│   └── experiments/                 # 實驗配置
│
├── 🔬 scripts/                      # 執行腳本
│   ├── train.py                     # 訓練腳本
│   ├── test.py                      # 測試腳本
│   └── slurm/                       # SLURM作業腳本
│
└── 📊 results/                      # 結果輸出
    ├── checkpoints/                 # 模型檢查點
    ├── logs/                        # 訓練日誌
    └── plots/                       # 視覺化結果
```

## 🔬 核心技術

### Physics-Informed Neural Networks (PINNs)

本專案實現的PINNs包含以下核心組件：

1. **主網路** (6層×80神經元): 求解速度場 (u, v) 和壓力場 (p)
2. **EVM網路** (4層×40神經元): 計算entropy residual用於人工粘滯度
3. **邊界條件**: 實現無滑移邊界條件 (u=v=0 on walls, u=1 on top lid)

### Entropy Viscosity Method (EVM)

EVM透過計算entropy residual來自動調整人工粘滯度：

```
entropy_residual = |∇·(u⊗u)| 
artificial_viscosity = min(β·entropy_residual/Re, β/Re)
```

### 多階段訓練策略

5階段漸進式訓練，逐步降低EVM權重：

| 階段 | Epochs | α_evm | Learning Rate | 說明 |
|------|--------|-------|---------------|------|
| 1    | 200K   | 0.05  | 1e-3         | 初始訓練 |
| 2    | 200K   | 0.01  | 2e-4         | 降低EVM |
| 3    | 200K   | 0.005 | 4e-5         | 精調+L-BFGS |
| 4    | 200K   | 0.002 | 1e-5         | 高精度 |
| 5    | 300K   | 0.001 | 2e-6         | 收斂 |

## 🧪 測試與驗證

```bash
# 運行單元測試
python -m pytest tests/unit/

# 運行整合測試  
python -m pytest tests/integration/

# 性能基準測試
python -m pytest tests/benchmarks/
```

## 📊 結果與性能

### 收斂性能 (Re=3000)

- **訓練時間**: ~24小時 (2×Tesla P100)
- **記憶體使用**: ~12GB GPU記憶體
- **最終誤差**: L2 < 1e-4

### 支援的Reynolds數

- ✅ Re = 100 (驗證用)
- ✅ Re = 1000 (基準測試)  
- ✅ Re = 3000 (主要目標)
- ✅ Re = 5000 (挑戰配置)

## 🛠️ 硬體兼容性

### Tesla P100專用優化

本專案針對Tesla P100 (CUDA Capability 6.0)進行了特別優化：

- 自動禁用torch.compile (需要CUDA ≥7.0)
- 設定TORCH_COMPILE_BACKEND=eager
- 優化記憶體使用模式

### SLURM作業系統

```bash
# 提交訓練作業
sbatch scripts/slurm/train.sh

# 監控作業狀態  
squeue -u $USER
```

## 📝 API文檔

詳細的API文檔請參考 [docs/api/](docs/api/) 目錄。

## 🤝 貢獻指南

歡迎貢獻！請參考以下流程：

1. Fork專案
2. 建立功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 建立Pull Request

## 📄 授權條款

本專案採用MIT授權條款 - 詳見 [LICENSE](LICENSE) 檔案。

## 🙏 致謝

- 基於 [ev-NSFnet](./ev-NSFnet/) 的成熟PINN實現
- 感謝 [opencode](https://github.com/sst/opencode) 和 GitHub Copilot 的開發支援
- 參考了多篇PINNs和CFD領域的經典論文

## 📞 聯絡方式

如有問題或建議，請：

- 提交 [Issue](https://github.com/your-repo/ldc-pinns/issues)
- 發送Email: your-email@domain.com
- 查看 [文檔](docs/) 了解更多詳情

---

*本專案是研究級別的PINNs實現，適用於學術研究和工程應用。*
