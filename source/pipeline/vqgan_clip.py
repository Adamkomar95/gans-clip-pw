import os
import random
from datetime import datetime
from pathlib import Path
import requests

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wget
from omegaconf import OmegaConf
from PIL import Image
from siren_pytorch import SirenNet, SirenWrapper
from source.models.clip.clip import load, tokenize
from source.pipeline.utils.torch_utils import (create_clip_img_transform,
                                               interpolate, rand_cutout)
from source.pipeline.utils.utils import (checkout, create_text_path, default,
                                         exists, load_vqgan, open_folder,
                                         slice_imgs, vqgan_image)
from taming.models.vqgan import VQModel
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch_optimizer import AdamP, DiffGrad
from tqdm import tqdm, trange


class VQGan(nn.Module):
    def __init__(
            self,
            clip_perceptor,
            clip_norm,
            input_res,
            total_batches,
            batch_size,
            num_layers=8,
            image_width=512,
            loss_coef=100,
            theta_initial=None,
            theta_hidden=None,
    ):
        super().__init__()
        # load clip
        self.perceptor = clip_perceptor
        self.input_resolution = input_res
        self.normalize_image = clip_norm
        
        self.loss_coef = loss_coef
        self.image_width = image_width

        self.batch_size = batch_size
        self.total_batches = total_batches
        self.num_batches_processed = 0

        w0 = default(theta_hidden, 30.)
        w0_initial = default(theta_initial, 30.)

class VQGanDataFlow(nn.Module):
    def __init__(
            self,
            *,
            text=None,
            img=None,
            clip_encoding=None,
            lr=1e-5,
            batch_size=4,
            gradient_accumulate_every=4,
            save_every=100,
            image_width=512,
            num_layers=16,
            epochs=20,
            iterations=1050,
            save_progress=True,
            seed=None,
            open_folder=True,
            save_date_time=False,
            # start_image_path=None,
            # start_image_train_iters=10,
            # start_image_lr=3e-4,
            theta_initial=None,
            theta_hidden=None,
            model_name="ViT-B/32", # można w BigSleep tak samo podawać model w parametrze
            optimizer="AdamP",
            jit=False,
            hidden_size=256,
            model_path="./",
            model_size=1024,
            ckpt_path="./",
            sideX=100,
            sideY=100,
            sync = 0.,
            samples = 1.,
            save_freq = 1,
            overscan = False,
            upload_image = False,
            root_dir='',
            model_download_path='',
    ):

        super().__init__()

        self.epochs = epochs
        self.model_name = model_name
        self.jit = jit

        if exists(seed):
            tqdm.write(f'setting seed: {seed}')
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            
        # # jit models only compatible with version 1.7.1
        # if "1.7.1" not in torch.__version__:
        #     if self.jit:
        #         print("Setting jit to False because torch version is not 1.7.1.")
        #     self.jit = False

        self.iterations = iterations
        self.image_width = image_width
        self.total_batches = self.epochs * self.iterations * batch_size * gradient_accumulate_every
        self.batch_size = batch_size
        self.image_width = image_width
        self.num_layers = num_layers
        self.theta_initial = theta_initial
        self.theta_hidden = theta_hidden
        self.lr = lr
        self.optimizer = optimizer
        self.model = None
        self.scaler = GradScaler()
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every
        self.save_date_time = save_date_time
        self.open_folder = open_folder
        self.save_progress = save_progress
        self.text = text
        self.img = img
        self.clip_encoding = clip_encoding

        #VQGan addition
        self.model_path = model_path
        self.model_download_path = model_download_path
        self.ckpt_path = ckpt_path
        self.sideX = sideX #@param {type:"integer"}
        self.sideY = sideY #@param {type:"integer"}
        self.model_name = model_name #@param ['ViT-B/32', 'RN101', 'RN50x4', 'RN50']
        self.samples = samples
        self.upload_image = upload_image
        self.model_size = model_size
        self.overscan = overscan
        self.sync = sync
        self.save_freq = save_freq

        #path managment
        self.root_dir = root_dir


    def create_clip_encoding(self, text=None, img=None, encoding=None):
        self.text = text
        self.img = img
        if encoding is not None:
            encoding = encoding.to(self.device)
        elif text is not None and img is not None:
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        elif text is not None:
            encoding = self.create_text_encoding(text)
        elif img is not None:
            encoding = self.create_img_encoding(img)
        return encoding

    def create_text_encoding(self, text):
        tokenized_text = tokenize(text).to(self.device)
        with torch.no_grad():
            text_encoding = self.perceptor.encode_text(tokenized_text).detach()
        return text_encoding
    
    def create_img_encoding(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        normed_img = self.clip_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_encoding = self.perceptor.encode_image(normed_img).detach()
        return img_encoding
    
    def set_clip_encoding(self, text=None, img=None, encoding=None):
        encoding = self.create_clip_encoding(text=text, img=img, encoding=encoding)
        self.clip_encoding = encoding.to(self.device)

    def image_output_path(self, sequence_number=None):
        """
        Returns underscore separated Path.
        A current timestamp is prepended if `self.save_date_time` is set.
        Sequence number left padded with 6 zeroes is appended if `save_every` is set.
        :rtype: Path
        """
        output_path = self.textpath
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            output_path = f"{output_path}.{sequence_number_left_padded}"
        if self.save_date_time:
            current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
            output_path = f"{current_time}_{output_path}"
        return Path(f"{output_path}.jpg")

    def train_step(self, epoch, iteration):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            with autocast(enabled=True):
                out, loss = self.model(self.clip_encoding)
            loss = loss / self.gradient_accumulate_every
            total_loss += loss
            self.scaler.scale(loss).backward()    
        out = out.cpu().float().clamp(0., 1.)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if (iteration % self.save_every == 0) and self.save_progress:
            self.save_image(epoch, iteration, img=out)

        return out, total_loss
    
    def get_img_sequence_number(self, epoch, iteration):
        current_total_iterations = epoch * self.iterations + iteration
        sequence_number = current_total_iterations // self.save_every
        return sequence_number

    @torch.no_grad()
    def save_image(self, epoch, iteration, img=None):
        sequence_number = self.get_img_sequence_number(epoch, iteration)

        if img is None:
            img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0., 1.)
        self.filename = self.image_output_path(sequence_number=sequence_number)
        
        pil_img = T.ToPILImage()(img.squeeze())
        pil_img.save(self.filename, quality=95, subsampling=0)
        pil_img.save(f"siren_{self.textpath}.jpg", quality=95, subsampling=0)

        tqdm.write(f'image updated at "./{str(self.filename)}"')

    def model_download(self):

        """
        Downloading VQGan model.
        """

        #creates path for vqgan model to be saved
        save_path = os.path.join(self.root_dir, self.ckpt_path)
        #sets vqgan model download
        if not Path(save_path).is_file():
            print("Downloading VQGAN model")
            vqgan_model = requests.get(self.model_download_path)
            open(save_path, 'wb').write(vqgan_model.content)
        else:
            print('VQGAN model already downloaded')

        return

    def run(self):

        workdir = '_out'
        tempdir = os.path.join(workdir, 'out')
        os.makedirs(tempdir, exist_ok=True)

        #Download VQGan if necessary
        self.model_download()

        # Load CLIP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_perceptor, norm_in = load(self.model_name, jit=self.jit, device=self.device)
        self.perceptor = clip_perceptor.eval()
        for param in self.perceptor.parameters():
            param.requires_grad = False
        if self.jit == False:
            input_res = clip_perceptor.visual.input_resolution
        else:
            input_res = clip_perceptor.input_resolution.item()
        self.clip_transform = create_clip_img_transform(input_res)

        print('CLIP Loaded, text, image encoded')

        modsize = 288 if self.model_name == 'RN50x4' else 224
        xmem = {'RN50':0.5, 'RN50x4':0.16, 'RN101':0.33}
        if 'RN' in self.model_name:
            self.samples = int(self.samples * xmem[self.model_name])

        if len(self.text) > 0:
            print(' text:', self.text)
            txt_enc = self.create_text_encoding(self.text)

        print(self.root_dir)

        abs_model_path = os.path.join(self.root_dir, self.model_path)
        abs_ckpt_path = os.path.join(self.root_dir, self.ckpt_path)

        config_vqgan = OmegaConf.load(abs_model_path)
        model_vqgan = load_vqgan(config_vqgan, ckpt_path=abs_ckpt_path).to(self.device)

        class latents(nn.Module):
            def __init__(self, shape):
                super(latents, self).__init__()
                init_rnd = torch.zeros(shape).normal_(0.,4.)
                self.lats = torch.nn.Parameter(init_rnd.cuda())
            def forward(self):
                return self.lats # [1,256, h//16, w//16]

        shape = [1, 256, self.sideY//16, self.sideX//16]
        lats = latents(shape).cuda()
        optimizer = torch.optim.Adam(lats.parameters(), self.lr)

        def save_img(img, fname=None):
            img = np.array(img)[:,:,:]
            img = np.transpose(img, (1,2,0))
            img = np.clip(img*255, 0, 255).astype(np.uint8)
            if fname is not None:
                imageio.imsave(fname, np.array(img))
                imageio.imsave('result.jpg', np.array(img))

        def checkout(num):
            with torch.no_grad():
                img = vqgan_image(model_vqgan, lats()).cpu().numpy()[0]
            save_img(img, os.path.join(tempdir, '%04d.jpg' % num))

        prev_enc = 0
        def train(i):
            loss = 0
            img_out = vqgan_image(model_vqgan, lats())

            imgs_sliced = slice_imgs([img_out], self.samples, modsize, norm_in, overscan=self.overscan)
            out_enc = clip_perceptor.encode_image(imgs_sliced[-1])

            if self.upload_image:
                loss += -1. * 0.5 * torch.cosine_similarity(img_enc, out_enc, dim=-1).mean()
            if len(self.text) > 0:
                loss += -1 * torch.cosine_similarity(txt_enc, out_enc, dim=-1).mean()

            if self.sync > 0 and self.upload_image:
                loss -= self.sync * ssim_loss(F.interpolate(img_out, ssim_size).float(), img_in)

            del img_out, imgs_sliced, out_enc; torch.cuda.empty_cache()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if i % self.save_freq == 0:
                checkout(i // self.save_freq)

        for i in range(self.iterations):
            train(i)
            print(f'Step {i+1}/{self.iterations}...')
