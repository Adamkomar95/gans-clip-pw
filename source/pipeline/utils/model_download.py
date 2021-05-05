import os
import wget
import hydra
from omegaconf import DictConfig, OmegaConf
import pathlib
import requests


@hydra.main(config_path="./", config_name="model_download")
def download_vqgan(cfg: DictConfig):

    #creates path for vqgan model to be saved
    save_path = os.path.join(pathlib.Path(__file__).parents[3], cfg.models.vqgan.model_save_path)

    #sets vqgan model download
    vqgan_model = requests.get(cfg.models.vqgan.model_download_path)

    open(save_path, 'wb').write(vqgan_model.content)

if __name__ == "__main__":
    download_vqgan()