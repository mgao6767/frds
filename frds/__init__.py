import sys

if sys.version_info.major < 3 and sys.version_info.minor < 8:
    print("Python3.8 and higher is required.")
    sys.exit(1)
