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
from source.pipeline.utils.utils import exists, default, open_folder, create_text_path
from source.pipeline.utils.torch_utils import rand_cutout, create_clip_img_transform, interpolate


def norm_siren_output(img):
    return ((img + 1) * 0.5).clamp(0.0, 1.0)


class DeepDaze(nn.Module):
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


class SirenDataFlow(nn.Module):
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
        self.image = img
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
        elif self.create_story:
            encoding = self.update_story_encoding(epoch=0, iteration=1)
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
        self.textpath = create_text_path(self.perceptor.context_length, text=self.text, img=self.img,
                                         encoding=self.clip_encoding)
        self.filename = self.image_output_path()

        # create coding to optimize for
        self.clip_encoding = self.create_clip_encoding(text=self.text, img=self.img, encoding=self.clip_encoding)

        # PRZETWARZANIE ZDJECIA POCZATKOWEGO - IGNORUJE

        # if exists(self.start_image):
        #     tqdm.write('Preparing with initial image...')
        #     optim = DiffGrad(self.model.model.parameters(), lr = self.start_image_lr)
        #     pbar = trange(self.start_image_train_iters, desc='iteration')
        #     try:
        #         for _ in pbar:
        #             loss = self.model.model(self.start_image)
        #             loss.backward()
        #             pbar.set_description(f'loss: {loss.item():.2f}')
        #
        #             optim.step()
        #             optim.zero_grad()
        #     except KeyboardInterrupt:
        #         print('interrupted by keyboard, gracefully exiting')
        #         return exit()
        #
        #     del self.start_image
        #     del optim

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
