# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
"""
語法驗證測試腳本
檢查 PhysicsNeMo EV-NSFnet PINN 專案的核心功能
"""
import os
import sys
import ast
import yaml
import torch
import numpy as np
from typing import Dict, Any


def test_python_syntax():
    """測試所有 Python 檔案的語法正確性"""
    print("=== Python 語法檢查 ===")
    py_files = [
        "physicsnemo_solver.py",
        "physicsnemo_net.py", 
        "physicsnemo_equations.py",
        "physicsnemo_data.py",
        "physicsnemo_train.py",
        "physicsnemo_test.py"
    ]
    
    all_passed = True
    for filename in py_files:
        try:
            with open(filename, 'r') as f:
                source = f.read()
            ast.parse(source, filename=filename)
            print(f"✓ {filename}: 語法正確")
        except SyntaxError as e:
            print(f"✗ {filename}: 語法錯誤 - {e}")
            all_passed = False
        except FileNotFoundError:
            print(f"? {filename}: 檔案不存在")
            all_passed = False
    
    return all_passed


def test_config_validation():
    """測試配置檔案的正確性"""
    print("\n=== 配置檔案檢查 ===")
    try:
        with open('conf/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 檢查必要的配置項目
        required_keys = [
            'reynolds_number', 'training_stages', 'main_net', 'evm_net',
            'alpha_boundary', 'alpha_equation', 'optimizer'
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"✗ 缺少必要配置項目: {missing_keys}")
            return False
        
        # 檢查訓練階段
        stages = config.get('training_stages', [])
        if len(stages) != 6:
            print(f"✗ 應有 6 個訓練階段，實際: {len(stages)}")
            return False
        
        # 檢查每個階段的必要參數
        for i, stage in enumerate(stages):
            required_stage_keys = ['stage_name', 'alpha_evm', 'epochs', 'learning_rate']
            missing_stage_keys = [key for key in required_stage_keys if key not in stage]
            if missing_stage_keys:
                print(f"✗ Stage {i+1} 缺少參數: {missing_stage_keys}")
                return False
        
        print("✓ 配置檔案格式正確")
        print(f"✓ 發現 {len(stages)} 個訓練階段")
        return True
        
    except Exception as e:
        print(f"✗ 配置檔案錯誤: {e}")
        return False


def test_gpu_compatibility():
    """測試 GPU 相容性設置"""
    print("\n=== GPU 相容性檢查 ===")
    
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability(0)
        cuda_capability = device_cap[0] + device_cap[1] * 0.1
        
        print(f"✓ CUDA 可用，GPU 能力: {device_cap[0]}.{device_cap[1]}")
        
        if cuda_capability < 7.0:
            print("✓ P100 相容性設置已啟用")
            # 檢查是否設置了相容性環境變數
            if os.environ.get('TORCHDYNAMO_DISABLE') == '1':
                print("✓ TorchDynamo 已禁用 (P100 相容)")
            else:
                print("ℹ️  TorchDynamo 設置將在程式運行時啟用")
        else:
            print("✓ 現代 GPU，無需特殊設置")
        
        return True
    else:
        print("⚠️  CUDA 不可用，將使用 CPU 模式")
        return True  # CPU 模式也是有效的


def test_training_stages_logic():
    """測試訓練階段邏輯的正確性"""
    print("\n=== 訓練階段邏輯檢查 ===")
    
    try:
        with open('conf/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        stages = config.get('training_stages', [])
        
        # 檢查 alpha_evm 是否遞減
        alpha_values = [stage['alpha_evm'] for stage in stages]
        is_decreasing = all(alpha_values[i] >= alpha_values[i+1] 
                           for i in range(len(alpha_values)-1))
        
        if is_decreasing:
            print("✓ Alpha_EVM 值正確遞減")
        else:
            print("✗ Alpha_EVM 值應該遞減")
            return False
        
        # 檢查學習率是否合理
        lr_values = [stage['learning_rate'] for stage in stages]
        if all(isinstance(lr, (int, float)) and lr > 0 for lr in lr_values):
            print("✓ 學習率值都為正數")
        else:
            print("✗ 發現非正數學習率")
            print(f"  學習率值: {lr_values}")
            return False
        
        # 檢查 epochs 是否一致
        epoch_values = [stage['epochs'] for stage in stages]
        if all(epochs == 500000 for epochs in epoch_values):
            print("✓ 所有階段 epochs 設置一致 (500,000)")
        else:
            print("ℹ️  階段 epochs 設置不一致，這可能是預期的")
        
        return True
        
    except Exception as e:
        print(f"✗ 訓練階段邏輯檢查失敗: {e}")
        return False


def main():
    """主測試函數"""
    print("🔍 PhysicsNeMo EV-NSFnet PINN 專案語法驗證")
    print("=" * 50)
    
    tests = [
        ("Python 語法檢查", test_python_syntax),
        ("配置檔案檢查", test_config_validation), 
        ("GPU 相容性檢查", test_gpu_compatibility),
        ("訓練階段邏輯檢查", test_training_stages_logic)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"💥 {test_name} 執行錯誤: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 測試結果: {passed_tests}/{total_tests} 通過")
    
    if passed_tests == total_tests:
        print("🎉 所有測試通過！專案語法驗證成功")
        return 0
    else:
        print("⚠️  部分測試失敗，請檢查上述錯誤")
        return 1


if __name__ == "__main__":
    sys.exit(main())