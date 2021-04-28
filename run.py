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

from source.pipeline.biggan_clip import BigGanDataFlow
from source.pipeline.siren_clip import SirenDataFlow
from source.pipeline.vqgan_clip import VQGanDataFlow

@hydra.main(config_path="./configs", config_name="run_config")
def runner(cfg: DictConfig):
    print("Initializing training for configuration:")
    print(cfg.pretty())

    print('BigGanXClip starting.')
    biggan_pipeline = BigGanDataFlow(text=cfg.train.text,
                                     epochs=cfg.train.epochs,
                                     iterations=cfg.train.iterations)
    biggan_pipeline.run()
    print('BigGanXClip finished.')



    print('SirenXClip starting.')
    siren_pipeline = SirenDataFlow(text=cfg.train.text,
                                    epochs=cfg.train.epochs,
                                    iterations=cfg.train.iterations)
    siren_pipeline.run()
    print('SirenXClip finished.')

    print('VQGanXClip starting.')
    vqgan_pipeline = VQGanDataFlow(text=cfg.train.text,
                                    epochs=cfg.train.epochs,
                                    iterations=cfg.train.iterations,
                                    lr = cfg.models.vqgan.lr,
                                    model_path=cfg.models.vqgan.model_config,
                                    ckpt_path=cfg.models.vqgan.model_ckpt,
                                    model_name=cfg.models.vqgan.model_name,
                                    model_size=cfg.models.vqgan.model_size,
                                    sideX=cfg.models.vqgan.sideX,
                                    sideY=cfg.models.vqgan.sideY,
                                    samples=cfg.models.vqgan.samples,
                                    save_freq=cfg.models.vqgan.save_freq,
                                    sync=cfg.models.vqgan.sync,
                                    overscan=cfg.models.vqgan.overscan,
                                    upload_image=cfg.models.vqgan.upload_image,
                                    )

    vqgan_pipeline.run()
    print('VQGanXClip finished.')


if __name__ == '__main__':
    runner()