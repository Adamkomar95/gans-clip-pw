"""
Adam:
- CLIP musimy koniecznie inicjować jednorazowo i podawać go do obu modeli w konstruktorach.
- wiele metod z klas ...Flow dot. encodingów CLIPa można wydzielić to jednej klasy - problem jest taki, że
opierają się na parametrach tych klas, więc trzeba by to wtedy podawać w parametrach i zrobi się pasta-code.
Do przemyślenia.

Adagrad:
- Zgadzam się.

Adadelta:
- Spoko.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from source.configs.config import Config
from source.pipeline.biggan_clip import BigGanDataFlow
from source.pipeline.siren_clip import SirenDataFlow
# from source.pipeline.vqgan_clip import VQGanDataFlow

@hydra.main(config_name="config")
def runner(cfg: Config):
    cfg = Config()

    print('BigGanXClip starting.')
    biggan_pipeline = BigGanDataFlow(text=cfg.train.TEXT,
                                     epochs=Scfg.train.EPOCHS,
                                     iterations=cfg.train.ITERATIONS)
    biggan_pipeline.run()
    print('BigGanXClip finished.')
    print('SirenXClip starting.')
    siren_pipeline = SirenDataFlow(text=cfg.train.TEXT,
                                   epochs=cfg.train.EPOCHS,
                                   iterations=cfg.train.ITERATIONS)
    siren_pipeline.run()
    print('SirenXClip finished.')
    print('SirenXClip starting.')
    vqgan_pipeline = VQGanDataFlow(text=cfg.train.TEXT,
                                   epochs=cfg.train.EPOCHS,
                                   iterations=cfg.train.ITERATIONS)
    vqgan_pipeline.run()
    print('SirenXClip finished.')

if __name__ == '__main__':
    runner()