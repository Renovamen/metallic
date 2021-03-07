import datetime

def get_datetime():
    timestamp = datetime.datetime.now()
    return f"{timestamp:%Y.%m.%d.%H.%M.%S}." + f"{timestamp:%f}"[:3]
