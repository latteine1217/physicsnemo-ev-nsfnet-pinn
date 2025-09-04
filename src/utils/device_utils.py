"""
設備工具函數

處理GPU/CPU設備選擇和配置
"""

import torch
import os
from typing import Optional


def setup_device(local_rank: int = 0, logger: Optional[object] = None) -> torch.device:
    """
    設定訓練設備
    
    Args:
        local_rank: 本地GPU rank
        logger: 日誌器（可選）
        
    Returns:
        torch.device: 選定的設備
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        
        # 設定CUDA設備
        torch.cuda.set_device(local_rank)
        
        # P100相容性設定
        os.environ.setdefault('TORCH_COMPILE_BACKEND', 'eager')
        os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
        
        if logger:
            gpu_name = torch.cuda.get_device_name(local_rank)
            gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
            logger.info(f"🔥 使用GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        if logger:
            logger.info("⚠️ CUDA不可用，使用CPU")
    
    return device


def get_cuda_info() -> dict:
    """
    獲取CUDA信息
    
    Returns:
        dict: CUDA信息字典
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            "id": i,
            "name": props.name,
            "memory_total": props.total_memory / 1024**3,
            "memory_free": (torch.cuda.get_device_properties(i).total_memory - 
                           torch.cuda.memory_allocated(i)) / 1024**3,
            "compute_capability": f"{props.major}.{props.minor}"
        }
        info["devices"].append(device_info)
    
    return info