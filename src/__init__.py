"""
LDC-PINNs: Physics-Informed Neural Networks for Lid-Driven Cavity Flow

A modular implementation of Physics-Informed Neural Networks (PINNs) for solving
the Navier-Stokes equations in lid-driven cavity flow problems with artificial
viscosity enhancement using the Entropy Viscosity Method (EVM).

Author: Developed with opencode + GitHub Copilot
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "LDC-PINNs Team"
__description__ = "Physics-Informed Neural Networks for Lid-Driven Cavity Flow"

# 簡化導入，避免循環依賴
from . import config
from . import models
from . import physics
from . import utils

__all__ = [
    "config",
    "models",
    "physics", 
    "utils"
]