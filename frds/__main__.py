"""Entry point of FRDS: `python -m frds`"""

import sys
from frds.gui.main import FRDSApplication

if __name__ == "__main__":
    FRDSApplication(sys.argv).run()
