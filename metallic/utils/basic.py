def get_datetime(timestamp):
    return f"{timestamp:%Y.%m.%d.%H.%M.%S}." + f"{timestamp:%f}"[:3]
