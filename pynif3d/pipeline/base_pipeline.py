import os

import gdown
import torch
import yaml

from pynif3d.common.verification import check_in_options, check_path_exists
from pynif3d.log.log_funcs import func_logger


class BasePipeline(torch.nn.Module):
    @func_logger
    def __init__(self):
        super().__init__()

    def load_pretrained_model(self, yaml_file, model_name, cache_directory="."):
        check_path_exists(yaml_file)

        with open(yaml_file) as stream:
            data = yaml.safe_load(stream)

        pretrained_models = list(data.keys())
        check_in_options(model_name, pretrained_models, "model_name")

        model_path = os.path.join(cache_directory, "model.pt")
        url = data[model_name]["url"]
        md5 = data[model_name]["md5"]
        gdown.cached_download(url, model_path, md5)

        check_path_exists(model_path)
        state_dict = torch.load(model_path)["model_state"]
        self.load_state_dict(state_dict)
