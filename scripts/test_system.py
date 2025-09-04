"""
PINN系統測試腳本

測試各個模組的基本功能是否正常
"""

import sys
import os
import torch
import numpy as np

# 添加src路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_config_manager():
    """測試配置管理器"""
    print("🔧 測試ConfigManager...")
    try:
        from src.config.config_manager import ConfigManager
        
        # 測試載入配置
        config = ConfigManager.load_config('configs/default.yaml')
        print(f"  ✅ 配置載入成功: Re={config.physics.reynolds_number}")
        return True
    except Exception as e:
        print(f"  ❌ ConfigManager測試失敗: {e}")
        return False

def test_networks():
    """測試神經網路"""
    print("🧠 測試神經網路...")
    try:
        from src.models.networks import FCNet
        from src.models.activations import LAAFActivation
        
        # 測試主網路
        main_net = FCNet(num_ins=2, num_outs=3, num_layers=3, hidden_size=20)
        x = torch.randn(100, 2)
        y = main_net(x)
        assert y.shape == (100, 3), f"主網路輸出形狀錯誤: {y.shape}"
        
        # 測試EVM網路
        evm_net = FCNet(num_ins=2, num_outs=1, num_layers=2, hidden_size=10,
                       activation=LAAFActivation)
        z = evm_net(x)
        assert z.shape == (100, 1), f"EVM網路輸出形狀錯誤: {z.shape}"
        
        print("  ✅ 神經網路測試通過")
        return True
    except Exception as e:
        print(f"  ❌ 神經網路測試失敗: {e}")
        return False

def test_physics_equations():
    """測試物理方程"""
    print("⚗️ 測試物理方程...")
    try:
        from src.physics.equations import PhysicsEquations
        
        physics = PhysicsEquations(reynolds_number=3000, alpha_evm=0.03)
        
        # 測試數據
        batch_size = 50
        x = torch.randn(batch_size, 1, requires_grad=True)
        y = torch.randn(batch_size, 1, requires_grad=True)
        u = torch.randn(batch_size, 1, requires_grad=True)
        v = torch.randn(batch_size, 1, requires_grad=True)
        p = torch.randn(batch_size, 1, requires_grad=True)
        e = torch.randn(batch_size, 1, requires_grad=True)
        
        # 計算物理殘差
        eq1, eq2, eq3, eq4 = physics.compute_physics_residuals(x, y, u, v, p, e)
        
        # 檢查輸出形狀
        for i, eq in enumerate([eq1, eq2, eq3, eq4]):
            assert eq.shape == (batch_size, 1), f"方程{i+1}輸出形狀錯誤: {eq.shape}"
        
        print("  ✅ 物理方程測試通過")
        return True
    except Exception as e:
        print(f"  ❌ 物理方程測試失敗: {e}")
        return False

def test_pinn_solver():
    """測試PINN求解器"""
    print("🔬 測試PINNSolver...")
    try:
        from src.solvers.pinn_solver import PINNSolver
        
        # 初始化求解器
        solver = PINNSolver('configs/default.yaml')
        
        # 生成測試數據
        batch_size = 100
        x_f = torch.randn(batch_size, 1, requires_grad=True)
        y_f = torch.randn(batch_size, 1, requires_grad=True)
        x_b = torch.randn(50, 1, requires_grad=True)
        y_b = torch.randn(50, 1, requires_grad=True)
        u_b = torch.zeros(50, 1)
        v_b = torch.zeros(50, 1)
        
        # 載入數據
        solver.load_training_data(x_f, y_f, x_b, y_b, u_b, v_b)
        
        # 設定優化器
        solver.setup_optimizer(learning_rate=1e-3)
        
        # 測試一步訓練
        loss_value, losses = solver.train_epoch()
        
        print(f"  ✅ PINN求解器測試通過: 損失={loss_value:.3e}")
        return True
    except Exception as e:
        print(f"  ❌ PINN求解器測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🚀 開始PINN系統測試...\n")
    
    tests = [
        test_config_manager,
        test_networks,
        test_physics_equations,
        test_pinn_solver
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！系統準備就緒。")
        return True
    else:
        print("⚠️ 部分測試失敗，請檢查相關模組。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)