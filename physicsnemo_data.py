# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import torch
import numpy as np
from typing import Dict, List, Union, Optional
from physicsnemo.datasets import Dataset
from physicsnemo.utils.loggers import get_logger


class PhysicsNeMoCavityDataset(Dataset):
    """
    PhysicsNeMo 標準腔體流動資料集
    
    實作用於 PINN 訓練的蓋驅動腔體流動資料集，
    包括內部配點和邊界條件資料。
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        reynolds_number: float = 5000.0,
        num_interior_points: int = 120000,
        num_boundary_points: int = 4000,
        domain_bounds: Optional[Dict[str, List[float]]] = None,
        dist=None,
        **kwargs
    ):
        """
        PhysicsNeMo 腔體資料集初始化
        
        Parameters
        ----------
        data_dir : str
            資料目錄路徑
        reynolds_number : float
            雷諾數
        num_interior_points : int
            內部配點數量 (全批次訓練)
        num_boundary_points : int
            邊界點數量
        domain_bounds : Dict[str, List[float]]
            計算域邊界 {"x": [x_min, x_max], "y": [y_min, y_max]}
        dist : DistributedManager
            分散式管理器
        """
        
        super().__init__()
        
        # 獲取日誌器
        self.logger = get_logger(__name__)
        
        # 資料集參數
        self.data_dir = data_dir
        self.reynolds_number = reynolds_number
        self.num_interior_points = num_interior_points
        self.num_boundary_points = num_boundary_points
        self.dist = dist
        
        # 設定計算域
        if domain_bounds is None:
            self.domain_bounds = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
        else:
            self.domain_bounds = domain_bounds
            
        # 生成訓練資料
        self._generate_training_data()
        
        # 載入參考資料
        self._load_reference_data()
        
        self.logger.info(
            f"PhysicsNeMo 腔體資料集初始化完成 - "
            f"Re={reynolds_number}, 內部點={num_interior_points}, 邊界點={num_boundary_points}"
        )
        
    def _generate_training_data(self):
        """生成訓練資料點"""
        
        # 生成內部配點 (使用拉丁超立方採樣)
        self.interior_data = self._generate_interior_points()
        
        # 生成邊界條件資料
        self.boundary_data = self._generate_boundary_points()
        
        # 將資料移到適當的裝置
        if self.dist is not None:
            device = self.dist.device
            for key in self.interior_data:
                self.interior_data[key] = self.interior_data[key].to(device)
            for key in self.boundary_data:
                self.boundary_data[key] = self.boundary_data[key].to(device)
                
    def _generate_interior_points(self) -> Dict[str, torch.Tensor]:
        """使用拉丁超立方採樣生成內部配點"""
        
        # 設定隨機種子 (用於可重現性)
        if self.dist is not None and self.dist.rank is not None:
            np.random.seed(42 + self.dist.rank)
        else:
            np.random.seed(42)
            
        # 拉丁超立方採樣
        n_dims = len(self.domain_bounds)
        samples = np.zeros((self.num_interior_points, n_dims))
        
        for i, (key, bounds) in enumerate(self.domain_bounds.items()):
            # LHS 分層採樣
            intervals = np.linspace(0, 1, self.num_interior_points + 1)
            samples[:, i] = np.random.uniform(intervals[:-1], intervals[1:])
            np.random.shuffle(samples[:, i])
            
            # 縮放到域邊界
            samples[:, i] = samples[:, i] * (bounds[1] - bounds[0]) + bounds[0]
        
        # 轉換為 PhysicsNeMo 字典格式
        interior_dict = {}
        for i, key in enumerate(self.domain_bounds.keys()):
            tensor = torch.tensor(samples[:, i:i+1], dtype=torch.float32)
            tensor.requires_grad_(True)  # 啟用自動微分
            interior_dict[key] = tensor
            
        return interior_dict
    
    def _generate_boundary_points(self) -> Dict[str, torch.Tensor]:
        """生成邊界條件資料點"""
        
        # 每個邊界的點數
        n_per_boundary = self.num_boundary_points // 4
        
        x_min, x_max = self.domain_bounds["x"]
        y_min, y_max = self.domain_bounds["y"]
        
        # 底部邊界 (y = 0) - 無滑移
        x_bottom = np.linspace(x_min, x_max, n_per_boundary)
        y_bottom = np.full_like(x_bottom, y_min)
        u_bottom = np.zeros_like(x_bottom)
        v_bottom = np.zeros_like(x_bottom)
        
        # 頂部邊界 (y = 1) - 移動蓋子
        x_top = np.linspace(x_min, x_max, n_per_boundary)
        y_top = np.full_like(x_top, y_max)
        # 使用原始 ev-NSFnet 的移動蓋子速度分布
        u_top = self._compute_moving_lid_velocity(x_top)
        v_top = np.zeros_like(x_top)
        
        # 左邊界 (x = 0) - 無滑移
        y_left = np.linspace(y_min, y_max, n_per_boundary)
        x_left = np.full_like(y_left, x_min)
        u_left = np.zeros_like(y_left)
        v_left = np.zeros_like(y_left)
        
        # 右邊界 (x = 1) - 無滑移
        y_right = np.linspace(y_min, y_max, n_per_boundary)
        x_right = np.full_like(y_right, x_max)
        u_right = np.zeros_like(y_right)
        v_right = np.zeros_like(y_right)
        
        # 合併所有邊界
        x_bc = np.concatenate([x_bottom, x_top, x_left, x_right])
        y_bc = np.concatenate([y_bottom, y_top, y_left, y_right])
        u_bc = np.concatenate([u_bottom, u_top, u_left, u_right])
        v_bc = np.concatenate([v_bottom, v_top, v_left, v_right])
        
        # 轉換為 PhysicsNeMo 張量格式
        boundary_dict = {
            "x": torch.tensor(x_bc.reshape(-1, 1), dtype=torch.float32),
            "y": torch.tensor(y_bc.reshape(-1, 1), dtype=torch.float32),
            "u_bc": torch.tensor(u_bc.reshape(-1, 1), dtype=torch.float32),
            "v_bc": torch.tensor(v_bc.reshape(-1, 1), dtype=torch.float32),
        }
        
        # 啟用邊界座標的自動微分
        boundary_dict["x"].requires_grad_(True)
        boundary_dict["y"].requires_grad_(True)
        
        return boundary_dict
    
    def _compute_moving_lid_velocity(self, x: np.ndarray) -> np.ndarray:
        """計算移動蓋子的速度分布 (原始 ev-NSFnet 公式)"""
        
        # u = 1 - cosh(50*(x-0.5)) / cosh(25)
        u_lid = 1.0 - np.cosh(50.0 * (x - 0.5)) / np.cosh(25.0)
        
        return u_lid
    
    def _load_reference_data(self):
        """載入參考解資料 (如果存在)"""
        
        try:
            import scipy.io
            filename = f"{self.data_dir}/cavity_Re{int(self.reynolds_number)}_256_Uniform.mat"
            
            data = scipy.io.loadmat(filename)
            
            # 提取參考資料
            x_ref = torch.tensor(data['X_ref'].flatten().reshape(-1, 1), dtype=torch.float32)
            y_ref = torch.tensor(data['Y_ref'].flatten().reshape(-1, 1), dtype=torch.float32)
            u_ref = torch.tensor(data['U_ref'].flatten().reshape(-1, 1), dtype=torch.float32)
            v_ref = torch.tensor(data['V_ref'].flatten().reshape(-1, 1), dtype=torch.float32)
            p_ref = torch.tensor(data['P_ref'].flatten().reshape(-1, 1), dtype=torch.float32)
            
            self.reference_data = {
                "x": x_ref,
                "y": y_ref,
                "u": u_ref,
                "v": v_ref,
                "p": p_ref,
            }
            
            self.logger.info(f"載入參考資料: Re={self.reynolds_number}, 點數={len(x_ref)}")
            
        except (FileNotFoundError, KeyError, ImportError) as e:
            self.logger.warning(f"無法載入參考資料 Re={self.reynolds_number}: {e}")
            self.reference_data = {}
    
    def get_interior_data(self) -> Dict[str, torch.Tensor]:
        """獲取內部配點資料 (用於 PDE 殘差計算)"""
        return self.interior_data
    
    def get_boundary_data(self) -> Dict[str, torch.Tensor]:
        """獲取邊界條件資料"""
        return self.boundary_data
    
    def get_reference_data(self) -> Dict[str, torch.Tensor]:
        """獲取參考解資料 (用於驗證)"""
        return self.reference_data
    
    def create_dataloader(
        self, 
        batch_size: Optional[int] = None, 
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """創建 PhysicsNeMo 兼容的資料載入器"""
        
        # 如果是全批次訓練 (batch_size=None)，返回完整資料
        if batch_size is None:
            return self
        
        # 否則創建標準 DataLoader
        from torch.utils.data import DataLoader
        
        if self.dist is not None and self.dist.world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(self, shuffle=shuffle)
            return DataLoader(
                self, 
                batch_size=batch_size,
                sampler=sampler,
                drop_last=drop_last
            )
        else:
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last
            )
    
    def __len__(self) -> int:
        """資料集長度"""
        return self.num_interior_points
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """獲取單個資料項 (用於 DataLoader)"""
        
        data_dict = {}
        for key, values in self.interior_data.items():
            data_dict[key] = values[idx:idx+1]
            
        return data_dict