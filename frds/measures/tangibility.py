from typing import Union
import pandas as pd
from frds.data.wrds.comp import funda, fundq


class Tangibility_variants:
    """Asset tangibility"""

    def __call__(self, data: Union[funda, fundq]) -> pd.Series:
        """Property, plant and equipment (net) scaled by contemporaneous total assets"""
        if isinstance(data, funda):
            return data.PPENT / data.AT
        if isinstance(data, fundq):
            return data.PPENTQ / data.ATQ


Tangibility = Tangibility_variants()
