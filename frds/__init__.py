import sys

meta = dict(
    __version__="0.7.4",
    __description__="Financial Research Data Services",
    __author__="Mingze Gao",
    __author_email__="adrian.gao@outlook.com",
    __project_url__="https://frds.io",
    __github_url__="https://github.com/mgao6767/frds/",
)

if sys.version_info.major < 3 and sys.version_info.minor < 8:
    print("Python3.8 and higher is required.")
    sys.exit(1)

# import frds.measures
