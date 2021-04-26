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


#to remove
from IPython.display import HTML, Image, display, clear_output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import ipywidgets as ipy

import warnings
warnings.filterwarnings("ignore")

# !pip install git+https://github.com/openai/CLIP.git
import clip
import pytorch_ssim as ssim


#to refactor
from clip_fft import slice_imgs, checkout
from utils import pad_up_to, basename, img_list, img_read, plot_text
import pytorch_lightning as pl


#to include as single files (???)
# !git clone https://github.com/CompVis/taming-transformers
# !mv taming-transformers/* ./

import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def vqgan_image(model, z):
    x = model.decode(z)
    x = (x+1.)/2.
    return x

class latents(torch.nn.Module):
    def __init__(self, shape):
        super(latents, self).__init__()
        init_rnd = torch.zeros(shape).normal_(0.,4.)
        self.lats = torch.nn.Parameter(init_rnd.cuda())
    def forward(self):
        return self.lats # [1,256, h//16, w//16]

def makevid(seq_dir, size=None):
  out_video = seq_dir + '.mp4'
  moviepy.editor.ImageSequenceClip(img_list(seq_dir), fps=25).write_videofile(out_video, verbose=False)
  data_url = "data:video/mp4;base64," + b64encode(open(out_video,'rb').read()).decode()
  wh = '' if size is None else 'width=%d height=%d' % (size, size)
  return """<video %s controls><source src="%s" type="video/mp4"></video>""" % (wh, data_url)

print('\nDone with function imports!')

####### INPUT 
######## SETTINGS AND GENERATING

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
  if len(text) > 0: # input text
      loss += sign * torch.cosine_similarity(txt_enc, out_enc, dim=-1).mean()

  if sync > 0 and upload_image:
      loss -= sync * ssim_loss(F.interpolate(img_out, ssim_size).float(), img_in)

  del img_out, imgs_sliced, out_enc; torch.cuda.empty_cache()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if i % save_freq == 0:
    checkout(i // save_freq)

if __name__ == "__main__":
  # outpic = ipy.Output()

  # pbar = ProgressBar(steps)
  for i in range(steps):
    train(i)
    print(f'Step {i+1}/{steps}...')
    # _ = pbar.upd()

  HTML(makevid(tempdir))
  torch.save(lats.lats, tempdir + '.pt')


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
        # Generowanie na podstawie podanego obrazka - na razie ignoruję

        # self.start_image = None
        # self.start_image_train_iters = start_image_train_iters
        # self.start_image_lr = start_image_lr
        # if exists(start_image_path):
        #     file = Path(start_image_path)
        #     assert file.exists(), f'file does not exist at given starting image path {self.start_image_path}'
        #     image = Image.open(str(file))
        #     start_img_transform = T.Compose([T.Resize(image_width),
        #                                      T.CenterCrop((image_width, image_width)),
        #                                      T.ToTensor()])
        #     image_tensor = start_img_transform(image).unsqueeze(0).to(self.device)
        #     self.start_image = image_tensor

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

        # Load CLIP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_perceptor, norm = load(self.model_name, jit=self.jit, device=self.device)
        self.perceptor = clip_perceptor.eval()
        for param in self.perceptor.parameters():
            param.requires_grad = False
        if self.jit == False:
            input_res = clip_perceptor.visual.input_resolution
        else:
            input_res = clip_perceptor.input_resolution.item()
        self.clip_transform = create_clip_img_transform(input_res)

        print('CLIP Loaded, text,image encoded')

        # Siren
        self.model = DeepDaze(
                self.perceptor,
                norm,
                input_res,
                self.total_batches,
                batch_size=self.batch_size,
                image_width=self.image_width,
                num_layers=self.num_layers,
                theta_initial=self.theta_initial,
                theta_hidden=self.theta_hidden,
                lower_bound_cutout=self.lower_bound_cutout,
                upper_bound_cutout=self.upper_bound_cutout,
                saturate_bound=self.saturate_bound,
                gauss_sampling=self.gauss_sampling,
                gauss_mean=self.gauss_mean,
                gauss_std=self.gauss_std,
                do_cutout=self.do_cutout,
                center_bias=self.center_bias,
                center_focus=self.center_focus,
                hidden_size=self.hidden_size,
                averaging_weight=self.averaging_weight,
            ).to(self.device)

        # optimizer
        siren_params = self.model.model.parameters()
        if self.optimizer == "AdamP":
            self.optimizer = AdamP(siren_params, self.lr)
        elif self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(siren_params, self.lr)
        elif self.optimizer == "DiffGrad":
            self.optimizer = DiffGrad(siren_params, self.lr)

        print('Siren and optimizer initialized.')

        # paths
        self.textpath = create_text_path(text=self.text, img=self.img,
                                         encoding=self.clip_encoding)
        self.filename = self.image_output_path()

        # create coding to optimize for
        self.clip_encoding = self.create_clip_encoding(text=self.text, img=self.img, encoding=self.clip_encoding)

        tqdm.write(f'Imagining "{self.textpath}" from the depths of my weights...')

        with torch.no_grad():
            self.model(self.clip_encoding, dry_run=True) # do one warmup step due to potential issue with CLIP and CUDA

        if self.open_folder:
            open_folder('/')
            self.open_folder = False

        try:
            for epoch in trange(self.epochs, desc='epochs'):
                pbar = trange(self.iterations, desc='iteration')
                for i in pbar:
                    _, loss = self.train_step(epoch, i)
                    pbar.set_description(f'loss: {loss.item():.2f}')

        except KeyboardInterrupt:
            print('interrupted by keyboard, gracefully exiting')
            return

        self.save_image(epoch, i)  # one final save at end
