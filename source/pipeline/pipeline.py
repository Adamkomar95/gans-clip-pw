from source.configs.config import Config
from source.pipeline.biggan_clip import BigGanDataFlow


if __name__ == '__main__':
    cfg = Config()
    biggan_pipeline = BigGanDataFlow(text=cfg.train.text, epochs=1, iterations=1000)
    biggan_pipeline.run()
