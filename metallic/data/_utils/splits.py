import os
import json

def load_splits(*args):
    basedir = os.path.join(os.path.dirname(__file__), '..')
    split_path = os.path.join(basedir, 'assets', *args)

    if not os.path.isfile(split_path):
        raise IOError('{} not found'.format(split_path))

    with open(split_path, 'r') as f:
        data = json.load(f)

    return data
