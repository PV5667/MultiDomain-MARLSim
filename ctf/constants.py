import yaml
from types import SimpleNamespace
from pathlib import Path

class Constants:
    # boilerplate code to create importable user-defined constants
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Constants, cls).__new__(cls)
            cls._instance.cfg = None
        return cls._instance

    def initialize(self, yaml_path):
        with open(yaml_path, 'r') as f:
            conf_dict = yaml.safe_load(f)
        
        def wrap(d):
            if not isinstance(d, dict):
                return d
            return SimpleNamespace(**{k: wrap(v) for k, v in d.items()})
        
        self.cfg = wrap(conf_dict)

settings = Constants()