#!/usr/bin/env python3
"""
主訓練腳本 - 使用模組化架構

基於ev-NSFnet/train.py，採用模組化設計
支援分佈式訓練、配置管理、監督學習等功能
"""

import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加src路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.config.config_manager import ConfigManager
from src.data.cavity_data import DataLoader
from src.solvers.pinn_solver import PINNSolver
from src.utils.device_utils import setup_device, get_cuda_info
from src.utils.logger import LoggerFactory

# 啟用CUDA優化
torch.backends.cudnn.benchmark = True


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='PINN Training with Configuration Management')
    parser.add_argument('--config', type=str, default='configs/production.yaml',
                       help='配置文件路徑 (default: configs/production.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='從檢查點恢復訓練')
    parser.add_argument('--dry-run', action='store_true',
                       help='只顯示配置不執行訓練')
    return parser.parse_args()


def setup_distributed():
    """設置分佈式訓練環境"""
    # 檢查分布式環境變數
    if 'RANK' not in os.environ:
        print("💻 單GPU模式")
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        return False  # 非分布式模式
    
    # 初始化分布式進程組
    try:
        if not dist.is_initialized():
            # 只有在 rank 0 時才顯示初始化信息
            rank = int(os.environ.get('RANK', 0))
            if rank == 0:
                print("🔗 初始化分布式訓練...")
                
            dist.init_process_group(backend='nccl')
            
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ['LOCAL_RANK'])
            
            # 只有主進程顯示分布式信息
            if rank == 0:
                print(f"📡 分布式訓練設置完成: {world_size} GPUs")
                print(f"   - Backend: NCCL")
                print(f"   - 每個進程負責 GPU {local_rank}")
            
            # 使用統一設備管理函數
            device = setup_device(local_rank)
                
        return True  # 分布式模式
        
    except Exception as e:
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print(f"❌ 分布式初始化失敗: {e}")
            print("💻 退回單GPU模式")
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        return False


def cleanup_distributed():
    """清理分布式訓練環境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def display_supervision_setup(config_manager, rank=0):
    """顯示監督設置和監督點位置"""
    if rank != 0:  # 只在主進程顯示
        return
        
    supervision_config = getattr(config_manager.config, 'supervision', None)
    if not supervision_config:
        return
    
    print("🎯 監督數據配置:")
    print(f"   啟用狀態: {'✅ 啟用' if supervision_config.enabled else '❌ 關閉'}")
    print(f"   監督點數: {supervision_config.data_points}")
    print(f"   數據權重: {supervision_config.weight}")
    print(f"   隨機種子: {supervision_config.random_seed}")
    print(f"   數據路徑: {supervision_config.data_path}")
    
    if supervision_config.enabled and supervision_config.data_points > 0:
        print("\n" + "="*50)
        print("🔍 監督點位置詳細信息：")
        print("="*50)
        
        # 檢查數據文件是否存在
        if not os.path.exists(supervision_config.data_path):
            print(f"⚠️  數據文件不存在: {supervision_config.data_path}")
            return
            
        try:
            loader = DataLoader()
            loader.print_supervision_locations(
                supervision_config.data_path,
                supervision_config.data_points, 
                supervision_config.random_seed
            )
        except Exception as e:
            print(f"⚠️  載入監督點位置時出錯: {e}")


def load_training_data(config, dataloader, pinn_solver, rank=0):
    """載入訓練資料並分配給各個進程"""
    if rank == 0:
        print("📁 載入訓練資料...")
    
    # 設置邊界資料
    boundary_data = dataloader.loading_boundary_data()
    xb_cpu = torch.as_tensor(boundary_data[0], dtype=torch.float32).contiguous()
    yb_cpu = torch.as_tensor(boundary_data[1], dtype=torch.float32).contiguous()
    ub_cpu = torch.as_tensor(boundary_data[2], dtype=torch.float32).contiguous()
    vb_cpu = torch.as_tensor(boundary_data[3], dtype=torch.float32).contiguous()
    
    # 分佈式資料分配（邊界資料）
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    total_b = xb_cpu.shape[0]
    
    if total_b < world_size:
        if rank < total_b:
            b_start, b_end = rank, rank + 1
        else:
            b_start, b_end = 0, 0
    else:
        per = total_b // world_size
        b_start = rank * per
        b_end = b_start + per if rank < world_size - 1 else total_b
    
    # 分佈式同步邊界索引
    if dist.is_initialized():
        idx = [b_start, b_end]
        dist.broadcast_object_list(idx, src=0)
        b_start, b_end = idx
    
    # 移至設備
    xb = xb_cpu[b_start:b_end].to(pinn_solver.device)
    yb = yb_cpu[b_start:b_end].to(pinn_solver.device)
    ub = ub_cpu[b_start:b_end].to(pinn_solver.device)
    vb = vb_cpu[b_start:b_end].to(pinn_solver.device)
    
    pinn_solver.set_boundary_data(X=(xb, yb, ub, vb))
    
    # 設置方程式訓練資料
    eq_data = dataloader.loading_training_data()
    xf_cpu = torch.as_tensor(eq_data[0], dtype=torch.float32).contiguous()
    yf_cpu = torch.as_tensor(eq_data[1], dtype=torch.float32).contiguous()
    
    # 分佈式資料分配（方程式資料）
    total_f = xf_cpu.shape[0]
    
    if total_f < world_size:
        if rank < total_f:
            f_start, f_end = rank, rank + 1
        else:
            f_start, f_end = 0, 1
    else:
        per_f = total_f // world_size
        f_start = rank * per_f
        f_end = f_start + per_f if rank < world_size - 1 else total_f
    
    # 分佈式同步方程式索引
    if dist.is_initialized():
        idxf = [f_start, f_end]
        dist.broadcast_object_list(idxf, src=0)
        f_start, f_end = idxf
    
    # 移至設備（需要梯度）
    xf = xf_cpu[f_start:f_end].to(pinn_solver.device).contiguous().requires_grad_(True)
    yf = yf_cpu[f_start:f_end].to(pinn_solver.device).contiguous().requires_grad_(True)
    
    pinn_solver.set_eq_training_data(X=(xf, yf))
    
    return True


def load_supervision_data(config, dataloader, pinn_solver, rank=0):
    """載入監督資料"""
    supervision_config = getattr(config, 'supervision', None)
    
    if not supervision_config or not supervision_config.enabled or supervision_config.data_points <= 0:
        if rank == 0:
            print("📊 未啟用監督資料或資料點數量為0")
        return False
    
    if rank == 0:
        print(f"📊 載入監督資料: {supervision_config.data_points} 個資料點...")
    
    try:
        # 載入監督資料
        x_sup, y_sup, u_sup, v_sup, p_sup = dataloader.loading_supervision_data(
            supervision_config.data_path, 
            supervision_config.data_points,
            supervision_config.random_seed
        )
        
        # 轉換為tensor並移到GPU
        if x_sup.shape[0] > 0:  # 確保有資料點
            x_sup_tensor = torch.as_tensor(x_sup, dtype=torch.float32).to(pinn_solver.device)
            y_sup_tensor = torch.as_tensor(y_sup, dtype=torch.float32).to(pinn_solver.device)
            u_sup_tensor = torch.as_tensor(u_sup, dtype=torch.float32).to(pinn_solver.device)
            v_sup_tensor = torch.as_tensor(v_sup, dtype=torch.float32).to(pinn_solver.device)
            p_sup_tensor = torch.as_tensor(p_sup, dtype=torch.float32).to(pinn_solver.device)
            
            # 設置監督資料到PINN
            pinn_solver.set_supervision_data(
                x_sup_tensor, y_sup_tensor, 
                u_sup_tensor, v_sup_tensor, p_sup_tensor
            )
            
            if rank == 0:
                print(f"✅ 監督資料載入完成，監督點座標: ({x_sup[0,0]:.4f}, {y_sup[0,0]:.4f})")
            
            return True
        
    except Exception as e:
        if rank == 0:
            print(f"❌ 監督資料載入失敗: {e}")
    
    return False


def execute_training_stages(config, pinn_solver, is_distributed, start_epoch=0, rank=0):
    """執行分階段訓練"""
    # 使用配置中的訓練階段
    training_stages = []
    for i, stage in enumerate(config.training.training_stages):
        alpha, epochs, lr = stage[0], stage[1], stage[2]
        sched = stage[3] if len(stage) > 3 else 'Constant'
        stage_name = f"Stage {i+1}"
        training_stages.append((alpha, epochs, lr, sched, stage_name))
    
    total_epochs = sum([stage[1] for stage in training_stages])
    
    if not is_distributed or rank == 0:
        print(f"🚀 開始完整訓練：總共 {total_epochs:,} epochs，分 {len(training_stages)} 個階段")
        print("=" * 60)
    
    # 執行分階段訓練
    current_epoch = start_epoch
    
    for stage_idx, (alpha_evm, num_epochs, learning_rate, sched_name, stage_name) in enumerate(training_stages):
        # 跳過已完成的epochs
        if current_epoch >= num_epochs:
            if rank == 0:
                print(f"⏭️ 跳過 {stage_name} (已完成)")
            current_epoch -= num_epochs
            continue
        
        epochs_to_run = num_epochs - current_epoch
        
        if not is_distributed or rank == 0:
            print(f"🔄 {stage_name}: alpha_evm={alpha_evm}, epochs={epochs_to_run}/{num_epochs}, lr={learning_rate:.2e}")
        
        # 設置階段參數
        pinn_solver.current_stage = stage_name
        pinn_solver.set_alpha_evm(alpha_evm)
        
        # 重建優化器
        weight_decay = getattr(config.training, 'weight_decay', 0.0)
        if hasattr(config.training, 'weight_decay_stages') and config.training.weight_decay_stages:
            weight_decay = config.training.weight_decay_stages[stage_idx]
        
        pinn_solver.build_optimizer(learning_rate, weight_decay)
        
        if rank == 0:
            print(f"   - Optimizer: AdamW (lr={learning_rate:.2e}, wd={weight_decay})")
        
        # 執行該階段的訓練
        try:
            pinn_solver.train_stage(
                epochs=epochs_to_run,
                stage_name=stage_name,
                scheduler=sched_name
            )
        except Exception as e:
            if rank == 0:
                print(f"❌ 訓練階段 {stage_name} 發生錯誤: {e}")
            raise
        
        current_epoch = 0  # 下一階段從0開始


def main():
    """主訓練函數"""
    args = parse_args()
    
    # 設置分佈式環境
    is_distributed = setup_distributed()
    
    # 獲取當前進程的rank
    rank = int(os.environ.get('RANK', 0))
    
    try:
        # 載入配置
        if rank == 0:
            print(f"📂 載入配置文件: {args.config}")
        
        config_manager = ConfigManager.from_file(args.config)
        config = config_manager.config
        
        # 只在主進程顯示配置信息
        if rank == 0:
            warnings = config_manager.validate_config()
            if warnings:
                print("⚠️  配置警告:")
                for warning in warnings:
                    print(f"   - {warning}")
            
            config_manager.print_config()
            display_supervision_setup(config_manager, rank)
            
            if args.dry_run:
                print("🏃 Dry run模式，不執行訓練")
                return
        
        # Dry run檢查
        if args.dry_run:
            return
        
        # 啟用異常檢測（調試用）
        torch.autograd.set_detect_anomaly(False)
        
        # 創建PINN求解器
        if rank == 0:
            print("🚀 創建PINN求解器...")
        
        pinn_solver = PINNSolver(config)
        
        # 創建資料載入器
        if rank == 0:
            print("📁 創建資料載入器...")
        
        dataloader = DataLoader(
            path='./data/',
            N_f=config.training.N_f,
            N_b=1000,
            sort_by_boundary_distance=getattr(config.training, 'sort_by_boundary_distance', True)
        )
        
        # 載入訓練資料
        load_training_data(config, dataloader, pinn_solver, rank)
        
        # 載入監督資料
        load_supervision_data(config, dataloader, pinn_solver, rank)
        
        # 載入評估資料
        filename = f'./data/cavity_Re{config.physics.Re}_256_Uniform.mat'
        if os.path.exists(filename):
            eval_data = dataloader.loading_evaluate_data(filename)
            pinn_solver.set_evaluation_data(*eval_data)
            if rank == 0:
                print(f"✅ 載入評估資料: {filename}")
        elif rank == 0:
            print(f"⚠️  評估資料不存在: {filename}")
        
        # 恢復訓練狀態
        start_epoch = 0
        if args.resume:
            if rank == 0:
                print(f"🔄 正在從檢查點恢復: {args.resume}")
            start_epoch = pinn_solver.load_checkpoint(args.resume)
            if rank == 0:
                if start_epoch > 0:
                    print(f"✅ 成功恢復，將從 epoch {start_epoch} 開始")
                else:
                    print("⚠️ 無法載入檢查點，將從頭開始訓練")
        
        # 執行分階段訓練
        execute_training_stages(config, pinn_solver, is_distributed, start_epoch, rank)
        
        if rank == 0:
            print("🎉 訓練完成！")
        
    except Exception as e:
        if rank == 0:
            print(f"❌ 訓練過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        # 清理分佈式環境
        if is_distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()