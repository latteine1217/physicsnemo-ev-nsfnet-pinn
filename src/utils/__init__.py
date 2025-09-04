"""
工具模組

提供設備管理、日誌、資料處理等工具函數
"""

from .device_utils import setup_device, get_cuda_info
from .logger import LoggerFactory, PINNLogger

try:
    from .tools import normalize_coordinates, LHSample, sort_pts, distance, minDistance
    tools_available = True
except ImportError:
    tools_available = False

if tools_available:
    __all__ = [
        'setup_device',
        'get_cuda_info',
        'LoggerFactory', 
        'PINNLogger',
        'normalize_coordinates',
        'LHSample',
        'sort_pts',
        'distance',
        'minDistance'
    ]
else:
    __all__ = [
        'setup_device',
        'get_cuda_info',
        'LoggerFactory', 
        'PINNLogger'
    ]