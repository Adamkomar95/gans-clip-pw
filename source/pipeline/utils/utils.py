import os
import sys
import subprocess
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
import torch
import torch.nn.functional as F

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/','\\')]
    if cmd_list == None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


def create_text_path(text=None, img=None, encoding=None):
    """
    Trzeba uważać na długość haseł (rózne modele, róznie to znoszą)
    """
    input_name = ""
    if text is not None:
        input_name += text
    if img is not None:
        if isinstance(img, str):
            img_name = "".join(img.split(".")[:-1]) # replace spaces by underscores, remove img extension
            img_name = img_name.split("/")[-1]  # only take img name, not path
        else:
            img_name = "PIL_img"
        input_name += "_" + img_name
    if encoding is not None:
        input_name = "your_encoding"
    return input_name.replace("-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:255]


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

#FOR VQGAN
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def checkout(img, fname=None, verbose=False):
    img = np.transpose(np.array(img)[:,:,:], (1,2,0))
    if fname is not None:
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        imsave(fname, img)

def slice_imgs(imgs, count, size=224, transform=None, overscan=False, micro=None, uniform=False):
    def map(x, a, b):
        return x * (b-a) + a

    rnd_size = torch.rand(count)
    if uniform is True or overscan is True:
        rnd_offx = torch.rand(count)
        rnd_offy = torch.rand(count)
    else: # normal around center
        rnd_offx = torch.clip(torch.randn(count) * 0.2 + 0.5, 0., 1.)
        rnd_offy = torch.clip(torch.randn(count) * 0.2 + 0.5, 0., 1.)
    
    sz = [img.shape[2:] for img in imgs]
    sz_min = [torch.min(torch.tensor(s)) for s in sz]
    if overscan is True:
        sz = [[2*s[0], 2*s[1]] for s in list(sz)]
        imgs = [pad_up_to(imgs[i], sz[i], type='centr') for i in range(len(imgs))]

    sliced = []
    for i, img in enumerate(imgs):
        cuts = []
        for c in range(count):
            if micro is True: # both scales, micro mode
                csize = map(rnd_size[c], size//4, max(size, 0.25*sz_min[i])).int()
            elif micro is False: # both scales, macro mode
                csize = map(rnd_size[c], 0.5*sz_min[i], sz_min[i]).int()
            else: # single scale
                csize = map(rnd_size[c], size, sz_min[i]).int()
            offsetx = map(rnd_offx[c], 0, sz[i][1] - csize).int()
            offsety = map(rnd_offy[c], 0, sz[i][0] - csize).int()
            cut = img[:, :, offsety:offsety + csize, offsetx:offsetx + csize]
            cut = F.interpolate(cut, (size,size), mode='bicubic', align_corners=False) # bilinear
            if transform is not None: 
                cut = transform(cut)
            cuts.append(cut)
        sliced.append(torch.cat(cuts, 0))
    return sliced