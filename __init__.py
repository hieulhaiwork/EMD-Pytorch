"""
Earth Mover Distance (EMD) CUDA Extension for PyTorch

This package provides efficient computation of Earth Mover Distance 
with automatic differentiation support for deep learning applications.
"""

from .emd import *

__version__ = "1.0.0"
__all__ = ["earth_mover_distance", "EMDFunction", "EMDLoss"]