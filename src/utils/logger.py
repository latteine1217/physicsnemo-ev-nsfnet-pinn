"""
簡單的日誌器模組

提供基本的日誌功能
"""

import logging
import sys
from typing import Optional


class PINNLogger:
    """PINN專用日誌器"""
    
    def __init__(self, name: str, level: str = "INFO", rank: int = 0):
        self.rank = rank
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            # 設定日誌級別
            self.logger.setLevel(getattr(logging, level.upper()))
            
            # 創建控制台處理器
            handler = logging.StreamHandler(sys.stdout)
            
            # 設定格式
            if rank == 0:
                formatter = logging.Formatter(
                    '[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S'
                )
            else:
                formatter = logging.Formatter(
                    f'[RANK{rank}][%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S'
                )
            
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str):
        """記錄info級別訊息"""
        if self.rank == 0:  # 只有主進程輸出
            self.logger.info(message)
    
    def warning(self, message: str):
        """記錄warning級別訊息"""
        if self.rank == 0:
            self.logger.warning(message)
    
    def error(self, message: str):
        """記錄error級別訊息"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """記錄debug級別訊息"""
        if self.rank == 0:
            self.logger.debug(message)


class LoggerFactory:
    """日誌器工廠"""
    
    @staticmethod
    def get_logger(name: str, level: str = "INFO", rank: int = 0) -> PINNLogger:
        """
        獲取日誌器
        
        Args:
            name: 日誌器名稱
            level: 日誌級別
            rank: 進程rank
            
        Returns:
            PINNLogger實例
        """
        return PINNLogger(name, level, rank)