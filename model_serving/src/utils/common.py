import yaml
import numpy as np


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content


def write_dict_to_yaml(dict_input: dict, yaml_file_path: str):
    try:
        current_file_data = read_yaml(yaml_file_path)
        current_file_data.update(dict_input)
        with open(yaml_file_path, "w") as f:
            yaml.dump(current_file_data, f)
    except (FileNotFoundError , AttributeError):
        with open(yaml_file_path, "w") as f:
            yaml.dump(dict_input, f)


def softmax_func(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
