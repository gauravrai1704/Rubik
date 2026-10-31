"""External reference solvers (published solutions, third-party programs), run through our harness."""
from .dwalton76 import DWalton76
from .santa_kmcoders import KMCodersPublished

__all__ = ["DWalton76", "KMCodersPublished"]
