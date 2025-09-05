# 🌊 LDC-PINNs: PhysicsNeMo Implementation

## 📋 專案概述

本專案實現了基於 **NVIDIA PhysicsNeMo 框架**的 Physics-Informed Neural Networks (PINNs) 來求解 lid-driven cavity flow 問題。這是一個從純 PyTorch 實現完全重構到 PhysicsNeMo-Sym 框架的完整實現。

### ✨ 主要特色

- ✅ **PhysicsNeMo-Sym 框架**: 使用 NVIDIA 官方物理符號計算框架
- ✅ **純物理驅動**: 無需外部數據，完全基於物理約束
- ✅ **分布式訓練**: 支援多GPU並行訓練
- ✅ **Tesla P100 相容**: 針對舊硬體優化
- ✅ **Hydra 配置**: 靈活的配置管理系統

## 🏗️ 專案架構

```
.
├── train_physicsnemo_advanced.py # 主訓練腳本 (PhysicsNeMo實現)
├── configs/
│   └── ldc_pinn_advanced.yaml   # Hydra配置文件
├── requirements.txt             # PhysicsNeMo依賴
├── outputs/                     # 訓練輸出和可視化結果
└── ev-NSFnet/                   # 原始PyTorch參考實現
```

## ⚡ PhysicsNeMo vs PyTorch 實現對比

| 特性 | PhysicsNeMo 實現 | 原始 PyTorch 實現 |
|------|-----------------|-----------------|
| **框架** | PhysicsNeMo-Sym | 純 PyTorch |
| **物理方程** | `NavierStokes` 類 | 手動微分 |
| **邊界條件** | `GeometryDatapipe` | 手動採樣 |
| **配置管理** | Hydra | YAML + 自定義 |
| **分布式** | `DistributedManager` | 手動 DDP |
| **日誌系統** | `PythonLogger` | 自定義 |
| **可擴展性** | 高 (框架抽象) | 中 (手動實現) |

## 🚀 快速開始

### 1. 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 安裝PhysicsNeMo (如果未安裝)
pip install physicsnemo
pip install "nvidia-physicsnemo.sym>=2.1.0" --no-build-isolation
```

### 2. 訓練模型

```bash
# 使用PhysicsNeMo框架訓練（進階入口）
python train_physicsnemo_advanced.py --config configs/ldc_pinn_advanced.yaml

# 指定不同的配置（Hydra覆寫）
python train_physicsnemo_advanced.py --config configs/ldc_pinn_advanced.yaml physics.Re=5000 training.max_epochs=20000
```

### 3. 結果查看

訓練結果將保存在 `./outputs/` 目錄中：
- `ldc_results_epoch_XXXXXX.png`: 流場可視化
- 訓練日誌和檢查點文件

## ⚙️ 配置參數

### 物理參數
- `physics.Re`: Reynolds數 (預設: 3000)
- `physics.nu`: 運動粘度 (1/Re)
- `physics.cavity_size`: 腔體尺寸

### 模型參數  
- `model.num_layers`: 網路層數 (6)
- `model.layer_size`: 每層神經元數 (512)
- `model.in_features`: 輸入特徵 (x, y)
- `model.out_features`: 輸出特徵 (u, v, p)

### 訓練參數
- `training.max_epochs`: 最大訓練輪數 (10000)
- `training.learning_rate`: 初始學習率 (1e-3)
- `training.scheduler.decay_rate`: 學習率衰減率

## 🔧 PhysicsNeMo 核心組件

### 1. 物理方程式定義
```python
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes

ns = NavierStokes(nu=cfg.physics.nu, rho=cfg.physics.rho, dim=2, time=False)
```

### 2. 物理約束計算
```python
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

phy_inf = PhysicsInformer(
    required_outputs=["continuity", "momentum_x", "momentum_y"],
    equations=ns,
    grad_method="autodiff"
)
```

### 3. 幾何採樣
```python
from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe
from physicsnemo.sym.geometry.primitives_2d import Rectangle

# 自動處理邊界和內部點採樣
bc_dataloader = GeometryDatapipe(
    geom_objects=[rectangle], 
    sample_type="surface"
)
```

## 📊 訓練監控

### 損失函數組成
- **連續方程殘差**: ∇·u = 0
- **動量方程殘差**: Re⁻¹∇²u - (u·∇)u - ∇p = 0  
- **邊界條件**: 無滑移邊界和移動頂壁

### 學習率調度
- 指數衰減: λ(step) = 0.9999871767586216^step
- 自動調整以確保收斂

## 🎯 結果分析

模型輸出包含：
1. **u velocity**: 水平速度分量
2. **v velocity**: 垂直速度分量  
3. **Pressure**: 壓力場
4. **Velocity Magnitude**: 速度幅度

## 🔍 與原始實現的差異

### PhysicsNeMo 優勢
- **更清晰的物理抽象**: 直接使用 NavierStokes 類
- **自動微分**: PhysicsInformer 自動處理梯度計算
- **標準化幾何**: 幾何採樣完全自動化
- **企業級日誌**: 完整的訓練監控和日誌系統

### 實現細節
- 使用 `DistributedManager` 替代手動 DDP 設置
- Hydra 配置系統提供更靈活的參數管理
- `GeometryDatapipe` 自動處理邊界條件採樣
- 標準化的 PhysicsNeMo 模型架構

## 🛠️ 開發工具

- **opencode + GitHub Copilot**: AI輔助開發
- **PhysicsNeMo**: NVIDIA科學計算AI框架
- **Hydra**: 配置管理  
- **PyTorch**: 深度學習後端

## 📈 效能比較

| 指標 | PhysicsNeMo實現 | 原始PyTorch實現 |
|------|----------------|----------------|
| **代碼量** | ~350行 | ~800行 |
| **配置靈活性** | 高 (Hydra) | 中 (硬編碼) |
| **物理正確性** | 高 (框架驗證) | 中 (手動實現) |
| **可維護性** | 高 (標準化) | 低 (自定義) |
| **擴展性** | 高 | 低 |

## 🎓 學習資源

- [PhysicsNeMo 官方文檔](https://docs.nvidia.com/deeplearning/physicsnemo/)
- [PhysicsNeMo-Sym 用戶指南](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/)
- [Lid-Driven Cavity 官方範例](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/ldc_pinns)

---

**開發者**: LDC-PINNs Team  
**框架**: NVIDIA PhysicsNeMo  
**授權**: Apache 2.0
