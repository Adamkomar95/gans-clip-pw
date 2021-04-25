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

# if upload_image:
#   in_img = list(uploaded.values())[0]
#   print(' image:', list(uploaded)[0])
#   img_in = torch.from_numpy(imageio.imread(in_img).astype(np.float32)/255.).unsqueeze(0).permute(0,3,1,2).cuda()[:,:3,:,:]
#   in_sliced = slice_imgs([img_in], samples, modsize, transform=norm_in)[0]
#   img_enc = model_clip.encode_image(in_sliced).detach().clone()
#   if sync > 0:
#     overscan = True
#     ssim_loss = ssim.SSIM(window_size = 11)
#     ssim_size = [sideY//4, sideX//4]
#     img_in = F.interpolate(img_in, ssim_size).float()
#     # img_in = F.interpolate(img_in, (sideY, sideX)).float()
#   else:
#     del img_in
#   del in_sliced; torch.cuda.empty_cache()

if len(text) > 0:
  print(' text:', text)
  # if translate:
  #   translator = Translator()
  #   text = translator.translate(text, dest='en').text
  #   print(' translated to:', text) 
  txt_enc = enc_text(text)
#  if no_text > 0:
#      txt_plot = torch.from_numpy(plot_text(text, modsize)/255.).unsqueeze(0).permute(0,3,1,2).cuda()
#      txt_plot_enc = model_clip.encode_image(txt_plot).detach().clone()

# if len(subtract) > 0:
#   print(' without:', subtract)
#   if translate:
#       translator = Translator()
#       subtract = translator.translate(subtract, dest='en').text
#       print(' translated to:', subtract) 
#   txt_enc0 = enc_text(subtract)

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
  # outpic.clear_output()
  # with outpic:
  #   display(Image('result.jpg'))

prev_enc = 0
def train(i):
  loss = 0
  img_out = vqgan_image(model_vqgan, lats())

  imgs_sliced = slice_imgs([img_out], samples, modsize, norm_in, overscan=overscan)
  out_enc = model_clip.encode_image(imgs_sliced[-1])
#  if diverse != 0:
#    imgs_sliced = slice_imgs([vqgan_image(model_vqgan, lats())], samples, modsize, norm_in, overscan=overscan)
#    out_enc2 = model_clip.encode_image(imgs_sliced[-1])
#    loss += diverse * torch.cosine_similarity(out_enc, out_enc2, dim=-1).mean()
#    del out_enc2; torch.cuda.empty_cache()
  if upload_image:
      loss += sign * 0.5 * torch.cosine_similarity(img_enc, out_enc, dim=-1).mean()
  if len(text) > 0: # input text
      loss += sign * torch.cosine_similarity(txt_enc, out_enc, dim=-1).mean()
#      if no_text > 0:
#          loss -= sign * no_text * torch.cosine_similarity(txt_plot_enc, out_enc, dim=-1).mean()
  # if len(subtract) > 0: # subtract text
  #     loss += -sign * 0.5 * torch.cosine_similarity(txt_enc0, out_enc, dim=-1).mean()
  if sync > 0 and upload_image: # image composition sync
      loss -= sync * ssim_loss(F.interpolate(img_out, ssim_size).float(), img_in)
#  if expand > 0:
#    global prev_enc
#    if i > 0:
#      loss += expand * torch.cosine_similarity(out_enc, prev_enc, dim=-1).mean()
#    prev_enc = out_enc.detach()
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