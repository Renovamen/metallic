import os
import datetime

def get_datetime() -> str:
    timestamp = datetime.datetime.now()
    return f"{timestamp:%Y.%m.%d.%H.%M.%S}." + f"{timestamp:%f}"[:3]

def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
