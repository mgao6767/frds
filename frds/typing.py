"""Define some types used in frds"""

from typing import List, Set, Tuple, NewType
from multiprocessing.shared_memory import SharedMemory
from numpy import dtype

TableID = NewType("TableID", Tuple[str, str, str])
SharedMemoryInfo = NewType(
    "SharedMemoryInfo", Tuple[SharedMemory, tuple, dtype]
)
