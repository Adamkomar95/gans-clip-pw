"""
Adam:
- CLIP musimy koniecznie inicjować jednorazowo i podawać go do obu modeli w konstruktorach.
- wiele metod z klas ...Flow dot. encodingów CLIPa można wydzielić to jednej klasy - problem jest taki, że
opierają się na parametrach tych klas, więc trzeba by to wtedy podawać w parametrach i zrobi się pasta-code.
Do przemyślenia.
"""


from source.configs.config import Config
from source.pipeline.biggan_clip import BigGanDataFlow
from source.pipeline.siren_clip import SirenDataFlow


if __name__ == '__main__':
    cfg = Config()

    print('BigGanXClip starting.')
    biggan_pipeline = BigGanDataFlow(text=cfg.train.TEXT,
                                     epochs=cfg.train.EPOCHS,
                                     iterations=cfg.train.ITERATIONS)
    biggan_pipeline.run()
    print('BigGanXClip finished.')
    print('SirenXClip starting.')
    siren_pipeline = SirenDataFlow(text=cfg.train.TEXT,
                                   epochs=cfg.train.EPOCHS,
                                   iterations=cfg.train.ITERATIONS)
    siren_pipeline.run()
    print('SirenXClip finished.')
