"""
The :mod:`frds.datasets` module includes sample datasets, useful for testing
measures without actual data.
For a complete documentation, see :doc:`/datasets/index`.
"""

from ._stocks import StockReturns

__all__ = [
    "StockReturns",
]
