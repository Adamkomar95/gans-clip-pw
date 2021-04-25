import os
import sys
import subprocess


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