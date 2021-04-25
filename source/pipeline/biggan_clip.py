"""
Latents + BigGAN => Model + EMA -> BigSleep (CLIP w lossie)


Odpuszczam na razie signal (biblioteka do procesowania asynchronicznego i "bezpiecznego zatrzymywania procesu"


"""

from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image

from source.models.gans.BigGAN.BigGAN import BigGAN
from source.models.clip.clip import load, tokenize
from source.pipeline.utils.utils import exists, create_text_path, open_folder
from source.pipeline.utils.torch_utils import differentiable_topk, create_clip_img_transform, rand_cutout
from source.pipeline.utils.ema import EMA


class Latents(torch.nn.Module):
    def __init__(
            self,
            num_latents=15,
            num_classes=1000,
            z_dim=128,
            max_classes=None,
            class_temperature=2.
    ):
        super().__init__()
        self.normu = torch.nn.Parameter(torch.zeros(num_latents, z_dim).normal_(std=1))
        self.cls = torch.nn.Parameter(torch.zeros(num_latents, num_classes).normal_(mean=-3.9, std=.3))
        self.register_buffer('thresh_lat', torch.tensor(1))

        assert not exists(
            max_classes) or 0 < max_classes <= num_classes, f'max_classes must be between 0 and {num_classes}'
        self.max_classes = max_classes
        self.class_temperature = class_temperature

    def forward(self):
        if exists(self.max_classes):
            classes = differentiable_topk(self.cls, self.max_classes, temperature=self.class_temperature)
        else:
            classes = torch.sigmoid(self.cls)

        return self.normu, classes


class Model(nn.Module):
    """
    Loaded BigGan + Latents
    """

    def __init__(
            self,
            image_size,
            max_classes=None,
            class_temperature=2.,
            ema_decay=0.99
    ):
        super().__init__()
        assert image_size in (128, 256, 512), 'image size must be one of 128, 256, or 512'
        self.biggan = BigGAN.from_pretrained(f'biggan-deep-{image_size}')  # TU POJAWIA SIE GAN
        self.max_classes = max_classes
        self.class_temperature = class_temperature
        self.ema_decay \
            = ema_decay
        self.latents = None
        self.init_latents()

    def init_latents(self):
        latents = Latents(
            num_latents=len(self.biggan.config.layers) + 1,
            num_classes=self.biggan.config.num_classes,
            z_dim=self.biggan.config.z_dim,
            max_classes=self.max_classes,
            class_temperature=self.class_temperature
        )
        self.latents = EMA(latents, self.ema_decay)

    def forward(self):
        self.biggan.eval()
        out = self.biggan(*self.latents(), 1)
        return (out + 1) / 2


class BigSleep(nn.Module):
    """
    BigSleep torch module.
    """

    def __init__(
            self,
            perceptor,  # from CLIP
            normalize_image,  # from CLIP
            num_cutouts=128,
            loss_coef=100,
            image_size=512,
            bilinear=False,
            max_classes=None,
            class_temperature=2.,
            # experimental_resample=False,  # not using resample currently
            ema_decay=0.99,
            center_bias=False
    ):
        super().__init__()
        self.normalize_image = normalize_image
        self.perceptor = perceptor
        self.loss_coef = loss_coef
        self.image_size = image_size
        self.num_cutouts = num_cutouts
        # self.experimental_resample = experimental_resample, # not using resample currently
        self.center_bias = center_bias

        self.interpolation_settings = {'mode': 'bilinear', 'align_corners': False} if bilinear else {'mode': 'nearest'}

        self.model = Model(
            image_size=image_size,
            max_classes=max_classes,
            class_temperature=class_temperature,
            ema_decay=ema_decay
        )

    def reset(self):
        self.model.init_latents()

    def sim_txt_to_img(self, text_embed, img_embed, text_type="max"):
        """
        CLIP + GAN "LOSS"
        """
        sign = -1
        if text_type == "min":
            sign = 1
        return sign * self.loss_coef * torch.cosine_similarity(text_embed, img_embed, dim=-1).mean()

    def forward(self, text_embeds, text_min_embeds=[], return_loss=True):
        width, num_cutouts = self.image_size, self.num_cutouts

        out = self.model()

        if not return_loss:
            return out

        pieces = []
        for ch in range(num_cutouts):
            # sample cutout size
            size = int(width * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
            # get cutout
            apper = rand_cutout(out, size, center_bias=self.center_bias)
            apper = F.interpolate(apper, (224, 224), **self.interpolation_settings)
            pieces.append(apper)

        into = torch.cat(pieces)
        into = self.normalize_image(into)

        image_embed = self.perceptor.encode_image(into)

        latents, soft_one_hot_classes = self.model.latents()
        num_latents = latents.shape[0]
        latent_thres = self.model.latents.model.thresh_lat

        lat_loss = torch.abs(1 - torch.std(latents, dim=1)).mean() + \
                   torch.abs(torch.mean(latents, dim=1)).mean() + \
                   4 * torch.max(torch.square(latents).mean(), latent_thres)

        for array in latents:
            mean = torch.mean(array)
            diffs = array - mean
            var = torch.mean(torch.pow(diffs, 2.0))
            std = torch.pow(var, 0.5)
            zscores = diffs / std
            skews = torch.mean(torch.pow(zscores, 3.0))
            kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0

            lat_loss = lat_loss + torch.abs(kurtoses) / num_latents + torch.abs(skews) / num_latents

        cls_loss = ((50 * torch.topk(soft_one_hot_classes, largest=False, dim=1, k=999)[0]) ** 2).mean()

        results = []
        for txt_embed in text_embeds:
            results.append(self.sim_txt_to_img(txt_embed, image_embed))
        for txt_min_embed in text_min_embeds:
            results.append(self.sim_txt_to_img(txt_min_embed, image_embed, "min"))
        sim_loss = sum(results).mean()
        return out, (lat_loss, cls_loss, sim_loss)


class BigGanDataFlow:

    def __init__(
            self,
            *,
            text=None,
            img=None,
            encoding=None,
            text_min="",
            lr=.07,
            image_size=512,
            gradient_accumulate_every=1,
            save_every=50,
            epochs=20,
            iterations=1050,
            save_progress=False,
            bilinear=False,
            open_folder=True,
            seed=None,
            append_seed=False,
            torch_deterministic=False,
            max_classes=None,
            class_temperature=2.,
            save_date_time=False,
            save_best=False,
            # experimental_resample=False,
            ema_decay=0.99,
            num_cutouts=128,
            center_bias=False,
    ):
        self.image_size = image_size
        self.bilinear = bilinear
        self.torch_deterministic = torch_deterministic
        self.max_classes = max_classes
        self.class_temperature = class_temperature
        # self.experimental_resample = experimental_resample
        self.ema_decay = ema_decay
        self.num_cutouts = num_cutouts
        self.center_bias = center_bias
        self.seed = seed
        self.append_seed = append_seed
        self.epochs = epochs
        self.iterations = iterations
        self.lr = lr
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.save_progress = save_progress
        self.save_date_time = save_date_time

        self.save_best = save_best
        self.current_best_score = 0

        self.open_folder = open_folder
        self.total_image_updates = (self.epochs * self.iterations) / self.save_every
        self.encoded_texts = {
            "max": [],  # te słowa chcemy na obrazie
            "min": []  # tych nie chcemy
        }
        self.model = None
        self.optimizer = None
        self.text = text
        self.img = img
        self.encoding = encoding
        self.text_min = text_min

    @property
    def seed_suffix(self):
        return f'.{self.seed}' if self.append_seed and exists(self.seed) else ''

    def create_clip_encoding(self, text=None, img=None, encoding=None):
        self.text = text
        self.img = img
        if encoding is not None:
            encoding = encoding.cuda()
        elif text is not None and img is not None:
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        elif text is not None:
            encoding = self.create_text_encoding(text)
        elif img is not None:
            encoding = self.create_img_encoding(img)
        return encoding

    def create_text_encoding(self, text):
        tokenized_text = tokenize(text).cuda()
        with torch.no_grad():
            text_encoding = self.perceptor.encode_text(tokenized_text).detach()
        return text_encoding

    def create_img_encoding(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        normed_img = self.clip_transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            img_encoding = self.perceptor.encode_image(normed_img).detach()
        return img_encoding

    def encode_multiple_phrases(self, text, img=None, encoding=None, text_type="max"):
        if text is not None and "|" in text:
            self.encoded_texts[text_type] = [self.create_clip_encoding(text=prompt_min, img=img, encoding=encoding) for
                                             prompt_min in text.split("|")]
        else:
            self.encoded_texts[text_type] = [self.create_clip_encoding(text=text, img=img, encoding=encoding)]

    def encode_max_and_min(self, text, img=None, encoding=None, text_min=""):
        self.encode_multiple_phrases(text, img=img, encoding=encoding)
        if text_min is not None and text_min != "":
            self.encode_multiple_phrases(text_min, img=img, encoding=encoding, text_type="min")

    def set_clip_encoding(self, text=None, img=None, encoding=None, text_min=""):
        self.current_best_score = 0
        self.text = text
        self.text_min = text_min

        if len(text_min) > 0:
            text = text + "_wout_" + text_min[:255] if text is not None else "wout_" + text_min[:255]
        text_path = create_text_path(text=text, img=img, encoding=encoding)
        if self.save_date_time:
            text_path = datetime.now().strftime("%y%m%d-%H%M%S-") + text_path

        self.text_path = text_path
        self.filename = Path(f'./biggan_{text_path}{self.seed_suffix}.png')
        self.encode_max_and_min(text, img=img, encoding=encoding, text_min=text_min)  # Tokenize and encode each prompt

    def reset(self):
        self.model.reset()
        self.model = self.model.cuda()
        self.optimizer = Adam(self.model.model.latents.parameters(), self.lr)

    def train_step(self, epoch, i, pbar=None):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            out, losses = self.model(self.encoded_texts["max"], self.encoded_texts["min"])
            loss = sum(losses) / self.gradient_accumulate_every
            total_loss += loss
            loss.backward()

        self.optimizer.step()
        self.model.model.latents.update()
        self.optimizer.zero_grad()

        if (i + 1) % self.save_every == 0:
            with torch.no_grad():
                self.model.model.latents.eval()
                out, losses = self.model(self.encoded_texts["max"], self.encoded_texts["min"])
                top_score, best = torch.topk(losses[2], k=1, largest=False)
                image = self.model.model()[best].cpu()
                self.model.model.latents.train()

                save_image(image, str(self.filename))
                if pbar is not None:
                    pbar.update(1)
                else:
                    print(f'image updated at "./{str(self.filename)}"')

                if self.save_progress:
                    total_iterations = epoch * self.iterations + i
                    num = total_iterations // self.save_every
                    save_image(image, Path(f'./{self.text_path}.{num}{self.seed_suffix}.png'))

                if self.save_best and top_score.item() < self.current_best_score:
                    self.current_best_score = top_score.item()
                    save_image(image, Path(f'./{self.text_path}{self.seed_suffix}.best.png'))

        return out, total_loss

    def run(self):

        # CLIP
        self.perceptor, self.normalize_image = load('ViT-B/32', jit=False)
        self.clip_transform = create_clip_img_transform(224)
        self.set_clip_encoding(text=self.text, img=self.img, encoding=self.encoding, text_min=self.text_min)

        print('CLIP loaded, CLIP text encoding finished.')

        # BigGAN
        self.model = BigSleep(
            perceptor=self.perceptor,
            normalize_image=self.normalize_image,
            image_size=self.image_size,
            bilinear=self.bilinear,
            max_classes=self.max_classes,
            class_temperature=self.class_temperature,
            # experimental_resample=self.experimental_resample,
            ema_decay=self.ema_decay,
            num_cutouts=self.num_cutouts,
            center_bias=self.center_bias,
        ).cuda()

        print('BigGAN loaded.')

        self.optimizer = Adam(self.model.model.latents.model.parameters(), self.lr)

        penalizing = ""
        if len(self.text_min) > 0:
            penalizing = f'penalizing "{self.text_min}"'
        print(f'Imagining "{self.text_path}" {penalizing}...')

        with torch.no_grad():
            self.model(self.encoded_texts["max"][0])  # one warmup step due to issue with CLIP and CUDA

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        image_pbar = tqdm(total=self.total_image_updates, desc='image update', position=2, leave=True)
        for epoch in trange(self.epochs, desc='      epochs', position=0, leave=True):
            pbar = trange(self.iterations, desc='   iteration', position=1, leave=True)
            image_pbar.update(0)
            for i in pbar:
                out, loss = self.train_step(epoch, i, image_pbar)
                pbar.set_description(f'loss: {loss.item():04.2f}')
