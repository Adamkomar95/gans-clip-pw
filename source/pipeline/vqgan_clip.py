import random
from datetime import datetime
from pathlib import Path
import torch
from siren_pytorch import SirenNet, SirenWrapper
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch_optimizer import DiffGrad, AdamP
from PIL import Image
import torchvision.transforms as T
from tqdm import trange, tqdm


from source.models.clip.clip import load, tokenize
from source.pipeline.utils.utils import exists, default, open_folder, create_text_path, load_config, load_vqgan, vqgan_image
from source.pipeline.utils.torch_utils import rand_cutout, create_clip_img_transform, interpolate


##### OLD

import os
import io
import time
from math import exp
import random
import imageio
import numpy as np
import PIL
from base64 import b64encode
import moviepy, moviepy.editor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from IPython.display import HTML, Image, display, clear_output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import ipywidgets as ipy

import warnings
warnings.filterwarnings("ignore")

import clip
import pytorch_ssim as ssim

#to refactor
from clip_fft import slice_imgs, checkout
from utils import pad_up_to, basename, img_list, img_read, plot_text
import pytorch_lightning as pl

import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
#####


def norm_siren_output(img):
    return ((img + 1) * 0.5).clamp(0.0, 1.0)


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
            lower_bound_cutout=0.1, # should be smaller than 0.8
            upper_bound_cutout=1.0,
            saturate_bound=False,
            gauss_sampling=False,
            gauss_mean=0.6,
            gauss_std=0.2,
            do_cutout=True,
            center_bias=False,
            center_focus=2,
            hidden_size=256,
            averaging_weight=0.3,
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

        siren = SirenNet(
            dim_in=2,
            dim_hidden=hidden_size,
            num_layers=num_layers,
            dim_out=3,
            use_bias=True,
            w0=w0,
            w0_initial=w0_initial
        )

        self.model = SirenWrapper(
            siren,
            image_width=image_width,
            image_height=image_width
        )

        self.saturate_bound = saturate_bound
        self.saturate_limit = 0.75  # cutouts above this value lead to destabilization
        self.lower_bound_cutout = lower_bound_cutout
        self.upper_bound_cutout = upper_bound_cutout
        self.gauss_sampling = gauss_sampling
        self.gauss_mean = gauss_mean
        self.gauss_std = gauss_std
        self.do_cutout = do_cutout
        self.center_bias = center_bias
        self.center_focus = center_focus
        self.averaging_weight = averaging_weight
        
    def sample_sizes(self, lower, upper, width, gauss_mean):
        if self.gauss_sampling:
            gauss_samples = torch.zeros(self.batch_size).normal_(mean=gauss_mean, std=self.gauss_std)
            outside_bounds_mask = (gauss_samples > upper) | (gauss_samples < upper)
            gauss_samples[outside_bounds_mask] = torch.zeros((len(gauss_samples[outside_bounds_mask]),)).uniform_(lower, upper)
            sizes = (gauss_samples * width).int()
        else:
            lower *= width
            upper *= width
            sizes = torch.randint(int(lower), int(upper), (self.batch_size,))
        return sizes

    def forward(self, text_embed, return_loss=True, dry_run=False):
        out = self.model()
        out = norm_siren_output(out)

        if not return_loss:
            return out
                
        # determine upper and lower sampling bound
        width = out.shape[-1]
        lower_bound = self.lower_bound_cutout
        if self.saturate_bound:
            progress_fraction = self.num_batches_processed / self.total_batches
            lower_bound += (self.saturate_limit - self.lower_bound_cutout) * progress_fraction

        # sample cutout sizes between lower and upper bound
        sizes = self.sample_sizes(lower_bound, self.upper_bound_cutout, width, self.gauss_mean)

        # create normalized random cutouts
        if self.do_cutout:   
            image_pieces = [rand_cutout(out, size, center_bias=self.center_bias, center_focus=self.center_focus) for size in sizes]
            image_pieces = [interpolate(piece, self.input_resolution) for piece in image_pieces]
        else:
            image_pieces = [interpolate(out.clone(), self.input_resolution) for _ in sizes]

        # normalize
        image_pieces = torch.cat([self.normalize_image(piece) for piece in image_pieces])
        
        # calc image embedding
        with autocast(enabled=False):
            image_embed = self.perceptor.encode_image(image_pieces)
            
        # calc loss
        # loss over averaged features of cutouts
        avg_image_embed = image_embed.mean(dim=0).unsqueeze(0)
        averaged_loss = -self.loss_coef * torch.cosine_similarity(text_embed, avg_image_embed, dim=-1).mean()
        # loss over all cutouts
        general_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
        # merge losses
        loss = averaged_loss * (self.averaging_weight) + general_loss * (1 - self.averaging_weight)

        # count batches
        if not dry_run:
            self.num_batches_processed += self.batch_size
        
        return out, loss


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
            lower_bound_cutout=0.1, # should be smaller than 0.8
            upper_bound_cutout=1.0,
            saturate_bound=False,
            averaging_weight=0.3,
            gauss_sampling=False,
            gauss_mean=0.6,
            gauss_std=0.2,
            do_cutout=True,
            center_bias=False,
            center_focus=2,
            optimizer="AdamP",
            jit=True,
            hidden_size=256,
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
            
        # jit models only compatible with version 1.7.1
        if "1.7.1" not in torch.__version__:
            if self.jit:
                print("Setting jit to False because torch version is not 1.7.1.")
            self.jit = False

        self.iterations = iterations
        self.image_width = image_width
        self.total_batches = self.epochs * self.iterations * batch_size * gradient_accumulate_every
        self.batch_size = batch_size
        self.image_width = image_width
        self.num_layers = num_layers
        self.theta_initial = theta_initial
        self.theta_hidden = theta_hidden
        self.lower_bound_cutout = lower_bound_cutout
        self.upper_bound_cutout = upper_bound_cutout
        self.saturate_bound = saturate_bound
        self.gauss_sampling = gauss_sampling
        self.gauss_mean = gauss_mean
        self.gauss_std = gauss_std
        self.do_cutout = do_cutout
        self.center_bias = center_bias
        self.center_focus = center_focus
        self.hidden_size = hidden_size
        self.averaging_weight = averaging_weight
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

    def run(self):

        workdir = '_out'
        tempdir = os.path.join(workdir, 'ttt')

        clear_output()

        text = "sun in the ocean" #@param {type:"string"}
        # subtract = "" #@param {type:"string"}
        # translate = False #@param {type:"boolean"}
        # invert = False #@param {type:"boolean"}
        upload_image = False #@param {type:"boolean"}    - WILL BE IMPORTANT FOR MODELS COMPARISON
        os.makedirs(tempdir, exist_ok=True)
        sideX = 400 #@param {type:"integer"}
        sideY = 400 #@param {type:"integer"}
        model = 'ViT-B/32' #@param ['ViT-B/32', 'RN101', 'RN50x4', 'RN50']
        VQGAN_size = 1024 #@param [1024, 16384]
        overscan = False #@param {type:"boolean"}
        sync =  0. #@param {type:"number"} - WILL BE IMPORTANT FOR MODELS COMPARISON
        steps = 300 #@param {type:"integer"}
        samples = 1 #@param {type:"integer"}
        learning_rate = 0.1 #@param {type:"number"}
        save_freq = 1 #@param {type:"integer"}

        model_clip, _ = clip.load(model)
        modsize = 288 if model == 'RN50x4' else 224
        xmem = {'RN50':0.5, 'RN50x4':0.16, 'RN101':0.33}
        if 'RN' in model:
            samples = int(samples * xmem[model])

        def enc_text(txt):
            emb = model_clip.encode_text(clip.tokenize(txt).cuda())
            return emb.detach().clone()
                
        norm_in = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        sign = -1.

        if len(text) > 0:
            print(' text:', text)
            txt_enc = enc_text(text)

        config_vqgan = load_config("./content/models_TT/model-%d.yaml" % int(VQGAN_size), display=False)
        model_vqgan = load_vqgan(config_vqgan, ckpt_path="./content/models_TT/last-%d.ckpt" % int(VQGAN_size)).cuda()

        shape = [1, 256, sideY//16, sideX//16]
        lats = latents(shape).cuda()
        optimizer = torch.optim.Adam(lats.parameters(), learning_rate)

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

            imgs_sliced = slice_imgs([img_out], samples, modsize, norm_in, overscan=overscan)
            out_enc = model_clip.encode_image(imgs_sliced[-1])

            if upload_image:
                loss += sign * 0.5 * torch.cosine_similarity(img_enc, out_enc, dim=-1).mean()
            if len(text) > 0:
                loss += sign * torch.cosine_similarity(txt_enc, out_enc, dim=-1).mean()

            if sync > 0 and upload_image:
                loss -= sync * ssim_loss(F.interpolate(img_out, ssim_size).float(), img_in)

            del img_out, imgs_sliced, out_enc; torch.cuda.empty_cache()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if i % save_freq == 0:
                checkout(i // save_freq)

        for i in range(steps):
            train(i)
            print(f'Step {i+1}/{steps}...')

        HTML(makevid(tempdir))
        torch.save(lats.lats, tempdir + '.pt')
