from .base import TabBase
import frds.measures


class TabCorporateFinance(TabBase):
    def __init__(self, parent):
        super().__init__(parent, frds.measures.Category.CORPORATE_FINANCE)
