"""
PINN訓練腳本

基於PhysicsNeMo框架的PINN訓練，支援：
- 多階段訓練
- 分散式訓練
- 檢查點保存/載入
- 自動混合優化（Adam + L-BFGS）
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import numpy as np
from typing import Tuple

# 添加src路徑到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.solvers.pinn_solver import PINNSolver
from src.config.config_manager import ConfigManager
from src.utils.logger import LoggerFactory


class CavityDataGenerator:
    """
    Lid-driven cavity數據生成器
    
    生成PINN訓練所需的邊界條件和方程點
    """
    
    def __init__(self, N_f: int = 20000, N_b: int = 1000):
        self.N_f = N_f  # 方程點數量
        self.N_b = N_b  # 邊界點數量
        self.x_min = -1.0
        self.x_max = 1.0
        self.y_min = -1.0
        self.y_max = 1.0
    
    def generate_boundary_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成邊界條件數據
        
        Returns:
            (x_b, y_b, u_b, v_b): 邊界點座標和邊界條件值
        """
        Nx = 513
        Ny = 513
        r_const = 10
        
        # 上邊界的滑動蓋子速度分佈
        upper_x = np.linspace(self.x_min, self.x_max, num=Nx)
        u_upper = 1 - np.cosh(r_const * (upper_x - 0.0)) / np.cosh(r_const * 1.0)
        
        # 邊界點座標: 下、上、左、右
        x_b = np.concatenate([
            np.linspace(self.x_min, self.x_max, num=Nx),  # 下邊界
            np.linspace(self.x_min, self.x_max, num=Nx),  # 上邊界
            self.x_min * np.ones([Ny]),                   # 左邊界
            self.x_max * np.ones([Ny])                    # 右邊界
        ], axis=0).reshape([-1, 1])
        
        y_b = np.concatenate([
            self.y_min * np.ones([Nx]),                   # 下邊界
            self.y_max * np.ones([Nx]),                   # 上邊界  
            np.linspace(self.y_min, self.y_max, num=Ny), # 左邊界
            np.linspace(self.y_min, self.y_max, num=Ny)  # 右邊界
        ], axis=0).reshape([-1, 1])
        
        # 邊界條件值
        u_b = np.concatenate([
            np.zeros([Nx]),     # 下邊界: u=0
            u_upper,            # 上邊界: 滑動蓋子
            np.zeros([Ny]),     # 左邊界: u=0
            np.zeros([Ny])      # 右邊界: u=0
        ], axis=0).reshape([-1, 1])
        
        v_b = np.zeros([x_b.shape[0]]).reshape([-1, 1])  # 所有邊界: v=0
        
        print(f"生成邊界點數量: {x_b.shape[0]}")
        return x_b, y_b, u_b, v_b
    
    def generate_training_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成訓練方程點（使用Latin Hypercube Sampling）
        
        Returns:
            (x_f, y_f): 方程點座標
        """
        # 簡化版LHS採樣
        np.random.seed(42)  # 確保可重現性
        
        x_f = np.random.uniform(self.x_min, self.x_max, (self.N_f, 1))
        y_f = np.random.uniform(self.y_min, self.y_max, (self.N_f, 1))
        
        print(f"生成方程點數量: {self.N_f}")
        return x_f, y_f


def setup_distributed_training():
    """設定分散式訓練環境"""
    if 'RANK' in os.environ:
        # 使用torchrun啟動的分散式訓練
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # 初始化process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        print(f"分散式訓練初始化: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        return True
    else:
        # 單GPU/CPU訓練
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        return False


def create_scheduler(optimizer, config, num_epochs: int):
    """
    創建學習率調度器
    
    Args:
        optimizer: 優化器
        config: 配置對象
        num_epochs: 總epoch數
        
    Returns:
        學習率調度器
    """
    scheduler_type = getattr(config.training, 'scheduler_type', 'constant').lower()
    
    if scheduler_type == 'cosine':
        eta_min = getattr(config.training, 'eta_min', 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=eta_min
        )
    elif scheduler_type == 'multistep':
        milestones = getattr(config.training, 'milestones', [num_epochs // 2])
        gamma = getattr(config.training, 'gamma', 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    else:
        # 常數學習率
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


def train_single_stage(solver: PINNSolver, num_epochs: int, stage_name: str,
                      save_dir: str, start_epoch: int = 0) -> float:
    """
    單階段訓練
    
    Args:
        solver: PINN求解器
        num_epochs: 訓練epoch數
        stage_name: 階段名稱
        save_dir: 保存目錄
        start_epoch: 起始epoch
        
    Returns:
        最終損失值
    """
    solver.current_stage = stage_name
    logger = solver.logger
    
    logger.info(f"=== 開始{stage_name}階段訓練 ===")
    logger.info(f"訓練epochs: {num_epochs}, 起始epoch: {start_epoch}")
    
    # 創建學習率調度器
    scheduler = create_scheduler(solver.opt, solver.config, num_epochs)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        # 訓練一個epoch
        loss_value, losses = solver.train_epoch()
        
        # 學習率調度
        scheduler.step()
        
        # 記錄最佳損失
        if loss_value < best_loss:
            best_loss = loss_value
        
        # 記錄TensorBoard
        solver.log_tensorboard(epoch, loss_value, losses)
        
        # 打印訓練日誌
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            current_lr = solver.opt.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                       f"Loss: {loss_value:.3e} | "
                       f"Best: {best_loss:.3e} | "
                       f"LR: {current_lr:.2e}")
            
            # 詳細損失信息
            if epoch % 500 == 0:
                logger.info(f"  方程損失: {losses[0]:.3e}")
                logger.info(f"  邊界損失: {losses[1]:.3e}")
                logger.info(f"  NS-X損失: {losses[3]:.3e}")
                logger.info(f"  NS-Y損失: {losses[4]:.3e}")
                logger.info(f"  連續性損失: {losses[5]:.3e}")
                logger.info(f"  Entropy損失: {losses[6]:.3e}")
        
        # 保存檢查點
        if (epoch + 1) % solver.checkpoint_freq == 0:
            checkpoint_dir = os.path.join(save_dir, stage_name)
            solver.save_checkpoint(epoch, checkpoint_dir)
    
    logger.info(f"=== {stage_name}階段完成 | 最佳損失: {best_loss:.3e} ===")
    return best_loss


def main():
    parser = argparse.ArgumentParser(description='PINN Training Script')
    parser.add_argument('--config', type=str, default='configs/production.yaml',
                       help='配置檔案路徑')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='檢查點路徑（可選）')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='結果保存目錄')
    
    args = parser.parse_args()
    
    # 設定分散式訓練
    is_distributed = setup_distributed_training()
    rank = int(os.environ.get('RANK', 0))
    
    # 初始化PINN求解器
    solver = PINNSolver(args.config)
    
    # 生成訓練數據
    data_generator = CavityDataGenerator(
        N_f=solver.config.training.N_f,
        N_b=getattr(solver.config.training, 'N_b', 1000)
    )
    
    # 邊界條件數據
    x_b, y_b, u_b, v_b = data_generator.generate_boundary_data()
    
    # 方程點數據
    x_f, y_f = data_generator.generate_training_points()
    
    # 轉換為torch張量並載入到求解器
    solver.load_training_data(
        x_f=torch.tensor(x_f, dtype=torch.float32),
        y_f=torch.tensor(y_f, dtype=torch.float32),
        x_b=torch.tensor(x_b, dtype=torch.float32),
        y_b=torch.tensor(y_b, dtype=torch.float32),
        u_b=torch.tensor(u_b, dtype=torch.float32),
        v_b=torch.tensor(v_b, dtype=torch.float32)
    )
    
    # 設定優化器
    learning_rate = getattr(solver.config.training, 'learning_rate', 1e-3)
    weight_decay = getattr(solver.config.training, 'weight_decay', 0.0)
    solver.setup_optimizer(learning_rate, weight_decay)
    
    # 創建保存目錄
    save_dir = os.path.join(args.save_dir, f"Re{solver.Re}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 訓練階段配置
    training_stages = getattr(solver.config.training, 'training_stages', None)
    
    if training_stages:
        # 多階段訓練
        for i, (alpha_evm, epochs, lr, scheduler_type) in enumerate(training_stages):
            stage_name = f"Stage{i+1}"
            
            # 更新EVM參數
            solver.update_evm_parameters(alpha_evm)
            
            # 更新學習率
            for group in solver.opt.param_groups:
                group['lr'] = lr
            
            # 執行階段訓練
            best_loss = train_single_stage(
                solver, epochs, stage_name, save_dir
            )
            
            if rank == 0:
                solver.logger.info(f"{stage_name}完成: alpha_evm={alpha_evm}, "
                                  f"lr={lr}, loss={best_loss:.3e}")
    else:
        # 單階段訓練
        num_epochs = getattr(solver.config.training, 'num_epochs', 10000)
        train_single_stage(solver, num_epochs, "Training", save_dir)
    
    # 最終保存
    if rank == 0:
        final_checkpoint_dir = os.path.join(save_dir, "final")
        solver.save_checkpoint(num_epochs - 1, final_checkpoint_dir)
        solver.logger.info("🎉 訓練完成！")
    
    # 清理分散式訓練
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()