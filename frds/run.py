"""Entry point of FRDS: `python -m frds`"""

import sys
from frds.gui.main import FRDSApplication


def main():
    FRDSApplication(sys.argv).run()


if __name__ == "__main__":
    main()
