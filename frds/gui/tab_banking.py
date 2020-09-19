from .base import TabBase
import frds.measures


class TabBanking(TabBase):
    def __init__(self, parent):
        super().__init__(parent, frds.measures.Category.BANKING)
