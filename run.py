import hydra
from omegaconf import DictConfig, OmegaConf

from source.configs.config import Config
from source.pipeline.biggan_clip import BigGanDataFlow
from source.pipeline.siren_clip import SirenDataFlow
from source.pipeline.vqgan_clip import VQGanDataFlow

@hydra.main(config_path=".", config_name="run_config")
def runner(cfg: DictConfig):
    print("Initializing training for configuration:")
    print(cfg.pretty())

    # print('BigGanXClip starting.')
    # biggan_pipeline = BigGanDataFlow(text=cfg.train.text,
    #                                  epochs=cfg.train.epochs,
    #                                  iterations=cfg.train.iterations)
    # biggan_pipeline.run()
    # print('BigGanXClip finished.')
    # print('SirenXClip starting.')
    # siren_pipeline = SirenDataFlow(text=cfg.train.text,
    #                                 epochs=cfg.train.epochs,
    #                                 iterations=cfg.train.iterations)
    # siren_pipeline.run()
    # print('SirenXClip finished.')
    print('VQGanXClip starting.')
    vqgan_pipeline = VQGanDataFlow(text=cfg.train.text,
                                    epochs=cfg.train.epochs,
                                    iterations=cfg.train.iterations)
    vqgan_pipeline.run()
    print('VQGanXClip finished.')

if __name__ == '__main__':
    runner()