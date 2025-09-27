"""
Earth Mover Distance (EMD) implementation in PyTorch.
"""

from .emd import *

__version__ = "1.0.0"
__all__ = ["earth_mover_distance", "EMDFunction", "EMDLoss"]