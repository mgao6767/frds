import os

USER_DIRECTORY = os.path.expanduser("~")
DATA_DIRECTORY = os.path.join(USER_DIRECTORY, "frds")

if not os.path.exists(DATA_DIRECTORY):
    os.makedirs(DATA_DIRECTORY)

CREDENTIALS_FILE_PATH = os.path.join(DATA_DIRECTORY, "credentials.json")
