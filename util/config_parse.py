import yaml, os
import os

'''
class ConfigParser:
    def __init__(self, args):
        # load model configuration
        cfg_file = args.config+'.yaml'
        with open(cfg_file) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # load argument
        for arg in args.__dict__:
            self.config[arg] = args.__dict__[arg]

        # string None handing
        self.convert_None(self.config)

    def __getitem__(self, name):
        return self.config[name]

    def convert_None(self, d):
        for key in d:
            if d[key] == 'None':
                d[key] = None
            if isinstance(d[key], dict):
                self.convert_None(d[key])

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-d', '--device', default=None, type=str)
    args.add_argument('-r', '--resume', action='store_true')
    
    args = args.parse_args()

    args.config = "./conf/resnet_cfg.yaml"

    cp = ConfigParser(args)
'''




class ConfigParser:
    def __init__(self, args):
        cfg_file = args.config

        # If user gave no extension, add ".yaml"
        if not cfg_file.endswith(".yaml"):
            cfg_file_with_ext = cfg_file + ".yaml"
        else:
            cfg_file_with_ext = cfg_file

        # Try paths in order:
        candidate_paths = [
            cfg_file,                      # as given
            cfg_file_with_ext,             # add .yaml if missing
            os.path.join("config", cfg_file),        # search in ./config
            os.path.join("config", cfg_file_with_ext) # search in ./config with .yaml
        ]

        # Pick the first one that exists
        cfg_path = None
        for path in candidate_paths:
            if os.path.isfile(path):
                cfg_path = path
                break

        if cfg_path is None:
            raise FileNotFoundError(
                f"Config file not found. Tried: {candidate_paths}"
            )

        # Load model configuration
        with open(cfg_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Add CLI args into config
        for arg in args.__dict__:
            self.config[arg] = args.__dict__[arg]

        # String 'None' → Python None
        self.convert_None(self.config)

    def __getitem__(self, name):
        return self.config[name]

    def convert_None(self, d):
        for key in d:
            if d[key] == "None":
                d[key] = None
            if isinstance(d[key], dict):
                self.convert_None(d[key])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str)
    parser.add_argument('-d', '--device', default=None, type=str)
    parser.add_argument('-r', '--resume', action='store_true')
    args = parser.parse_args()

    # Example: both work now
    # python test.py -c SIDD ...
    # python test.py -c config/SIDD.yaml ...
    cp = ConfigParser(args)
    print("Config loaded successfully from:", args.config)
    print("Top-level keys:", list(cp.config.keys()))


