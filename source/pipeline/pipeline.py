from source.configs.config import Config
from source.pipeline.gan_clip import DataFlow


if __name__ == '__main__':
    cfg = Config()
    pipeline = DataFlow(text=cfg.train.text, epochs=1, iterations=1000)
    pipeline.run()
