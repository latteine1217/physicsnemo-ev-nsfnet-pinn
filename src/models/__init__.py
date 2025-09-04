"""
神經網路模組

包含FCNet全連接網路、LAAF激活函數等神經網路組件。
"""

from .networks import FCNet, create_pinn_networks
from .activations import (
    TSAActivation, 
    LAAFActivation, 
    AdaptiveSinusoidalActivation,
    compute_activation_regularization,
    get_activation_function,
    ActivationMonitor
)

__all__ = [
    "FCNet", 
    "create_pinn_networks",
    "TSAActivation",
    "LAAFActivation", 
    "AdaptiveSinusoidalActivation",
    "compute_activation_regularization",
    "get_activation_function",
    "ActivationMonitor"
]