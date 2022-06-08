import yaml
from dotmap import DotMap

def load_yaml(path):
    with open(path) as f:
        d = yaml.full_load(f)
    f.close()

    m = DotMap(d)
    return m
