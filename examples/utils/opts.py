import argparse
import yaml

'''
convert a dict into a Class
'''
class Config:
    def __init__(self, entries: dict={}):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config(v)
            else:
                self.__dict__[k] = v


'''
Load configuration from a YAML file.

input param: 
    file_path: path to the config file (YAML)

return:
    config (dict): configuration settings
'''
def load_config(file_path):
    f = open(file_path, 'r', encoding = 'utf-8')
    config = yaml.load(f.read(), Loader = yaml.FullLoader)
    return config


def parse_opt():

    parser = argparse.ArgumentParser()
    # config file
    parser.add_argument(
        '--config',
        type = str,
        default = 'examples/configs/maml/omniglot_5way_1shot_cnn.yaml',
        help = 'Path to the configuration file (yaml).'
    )
    args = parser.parse_args()
    config_dict = load_config(args.config)
    config = Config(config_dict)

    return config