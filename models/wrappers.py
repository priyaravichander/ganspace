# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import torch
import numpy as np
import re
import os
import random
from pathlib import Path
from types import SimpleNamespace
from utils import download_ckpt
from config import Config
from netdissect import proggan, zdataset
from . import biggan
from . import stylegan
from . import stylegan2
from abc import abstractmethod, ABC as AbstractBaseClass
from functools import singledispatch

class BaseModel(AbstractBaseClass, torch.nn.Module):

    # Set parameters for identifying model from instance
    def __init__(self, model_name, class_name):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.outclass = class_name

    # Stop model evaluation as soon as possible after
    # given layer has been executed, used to speed up
    # netdissect.InstrumentedModel::retain_layer().
    # Validate with tests/partial_forward_test.py
    # Can use forward() as fallback at the cost of performance.
    @abstractmethod
    def partial_forward(self, x, layer_name):
        pass

    # Generate batch of latent vectors
    @abstractmethod
    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        pass

    # Maximum number of latents that can be provided
    # Typically one for each layer
    def get_max_latents(self):
        return 1

    # Name of primary latent space
    # E.g. StyleGAN can alternatively use W
    def latent_space_name(self):
        return 'Z'

    def get_latent_shape(self):
        return tuple(self.sample_latent(1).shape)

    def get_latent_dims(self):
        return np.prod(self.get_latent_shape())

    def set_output_class(self, new_class):
        self.outclass = new_class

    # Map from typical range [-1, 1] to [0, 1]
    def forward(self, x):
        out = self.model.forward(x)
        return 0.5*(out+1)

    # Generate images and convert to numpy
    def sample_np(self, z=None, n_samples=1, seed=None):
        if z is None:
            z = self.sample_latent(n_samples, seed=seed)
        elif isinstance(z, list):
            z = [torch.tensor(l).to(self.device) if not torch.is_tensor(l) else l for l in z]
        elif not torch.is_tensor(z):
            z = torch.tensor(z).to(self.device)
        img = self.forward(z)
        img_np = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return np.clip(img_np, 0.0, 1.0).squeeze()

    # For models that use part of latent as conditioning
    def get_conditional_state(self, z):
        return None

    # For models that use part of latent as conditioning
    def set_conditional_state(self, z, c):
        return z

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)

    

# PyTorch port of StyleGAN 2
class StyleGAN2(BaseModel):
    def __init__(self, device, class_name, truncation=1.0, use_w=False):
        super(StyleGAN2, self).__init__('StyleGAN2', class_name or 'ffhq')
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = use_w # use W as primary latent space?

        # Image widths
        configs = {
            # Converted NVIDIA official
            'ffhq': 1024,
            'car': 512,
            'cat': 256,
            'church': 256,
            'horse': 256,
            # Tuomas
            'bedrooms': 256,
            'kitchen': 256,
            'places': 256,
        }

        assert self.outclass in configs, \
            f'Invalid StyleGAN2 class {self.outclass}, should be one of [{", ".join(configs.keys())}]'

        self.resolution = configs[self.outclass]
        self.name = f'StyleGAN2-{self.outclass}'
        self.has_latent_residual = True
        self.load_model()
        self.set_noise_seed(0)

    def latent_space_name(self):
        return 'W' if self.w_primary else 'Z'

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    # URLs created with https://sites.google.com/site/gdocs2direct/
    def download_checkpoint(self, outfile):
        checkpoints = {
            'horse': 'https://drive.google.com/uc?export=download&id=18SkqWAkgt0fIwDEf2pqeaenNi4OoCo-0',
            'ffhq': 'https://drive.google.com/uc?export=download&id=1FJRwzAkV-XWbxgTwxEmEACvuqF5DsBiV',
            'church': 'https://drive.google.com/uc?export=download&id=1HFM694112b_im01JT7wop0faftw9ty5g',
            'car': 'https://drive.google.com/uc?export=download&id=1iRoWclWVbDBAy5iXYZrQnKYSbZUqXI6y',
            'cat': 'https://drive.google.com/uc?export=download&id=15vJP8GDr0FlRYpE8gD7CdeEz2mXrQMgN',
            'places': 'https://drive.google.com/uc?export=download&id=1X8-wIH3aYKjgDZt4KMOtQzN1m4AlCVhm',
            'bedrooms': 'https://drive.google.com/uc?export=download&id=1nZTW7mjazs-qPhkmbsOLLA_6qws-eNQu',
            'kitchen': 'https://drive.google.com/uc?export=download&id=15dCpnZ1YLAnETAPB0FGmXwdBclbwMEkZ'
        }

        url = checkpoints[self.outclass]
        download_ckpt(url, outfile)

    def load_model(self):
        checkpoint_root = os.environ.get('GANCONTROL_CHECKPOINT_DIR', Path(__file__).parent / 'checkpoints')
        checkpoint = Path(checkpoint_root) / f'stylegan2/stylegan2_{self.outclass}_{self.resolution}.pt'
        
        self.model = stylegan2.Generator(self.resolution, 512, 8).to(self.device)

        if not checkpoint.is_file():
            os.makedirs(checkpoint.parent, exist_ok=True)
            self.download_checkpoint(checkpoint)
        
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['g_ema'], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max) # use (reproducible) global rand state

        rng = np.random.RandomState(seed)
        z = torch.from_numpy(
                rng.standard_normal(512 * n_samples)
                .reshape(n_samples, 512)).float().to(self.device) #[N, 512]
        
        if self.w_primary:
            z = self.model.style(z)

        return z

    def get_max_latents(self):
        return self.model.n_latent

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError('StyleGAN2: cannot change output class without reloading')
    
    def forward(self, x):
        x = x if isinstance(x, list) else [x]
        out, _ = self.model(x, noise=self.noise,
            truncation=self.truncation, truncation_latent=self.latent_avg, input_is_w=self.w_primary)
        return 0.5*(out+1)

    def partial_forward(self, x, layer_name):
        styles = x if isinstance(x, list) else [x]
        inject_index = None
        noise = self.noise

        if not self.w_primary:
            styles = [self.model.style(s) for s in styles]

        if len(styles) == 1:
            # One global latent
            inject_index = self.model.n_latent
            latent = self.model.strided_style(styles[0].unsqueeze(1).repeat(1, inject_index, 1)) # [N, 18, 512]
        elif len(styles) == 2:
            # Latent mixing with two latents
            if inject_index is None:
                inject_index = random.randint(1, self.model.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.model.n_latent - inject_index, 1)

            latent = self.model.strided_style(torch.cat([latent, latent2], 1))
        else:
            # One latent per layer
            assert len(styles) == self.model.n_latent, f'Expected {self.model.n_latents} latents, got {len(styles)}'
            styles = torch.stack(styles, dim=1) # [N, 18, 512]
            latent = self.model.strided_style(styles)

        if 'style' in layer_name:
            return

        out = self.model.input(latent)
        if 'input' == layer_name:
            return

        out = self.model.conv1(out, latent[:, 0], noise=noise[0])
        if 'conv1' in layer_name:
            return

        skip = self.model.to_rgb1(out, latent[:, 1])
        if 'to_rgb1' in layer_name:
            return

        i = 1
        noise_i = 1

        for conv1, conv2, to_rgb in zip(
            self.model.convs[::2], self.model.convs[1::2], self.model.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise[noise_i])
            if f'convs.{i-1}' in layer_name:
                return

            out = conv2(out, latent[:, i + 1], noise=noise[noise_i + 1])
            if f'convs.{i}' in layer_name:
                return
            
            skip = to_rgb(out, latent[:, i + 2], skip)
            if f'to_rgbs.{i//2}' in layer_name:
                return

            i += 2
            noise_i += 2

        image = skip

        raise RuntimeError(f'Layer {layer_name} not encountered in partial_forward')

    def set_noise_seed(self, seed):
        torch.manual_seed(seed)
        self.noise = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=self.device)]

        for i in range(3, self.model.log_size + 1):
            for _ in range(2):
                self.noise.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=self.device))

# PyTorch port of StyleGAN 1
class StyleGAN(BaseModel):
    def __init__(self, device, class_name, truncation=1.0, use_w=False):
        super(StyleGAN, self).__init__('StyleGAN', class_name or 'ffhq')
        self.device = device
        self.w_primary = use_w # is W primary latent space?

        configs = {
            # Official
            'ffhq': 1024,
            'celebahq': 1024,
            'bedrooms': 256,
            'cars': 512,
            'cats': 256,
            
            # From https://github.com/justinpinkney/awesome-pretrained-stylegan
            'vases': 1024,
            'wikiart': 512,
            'fireworks': 512,
            'abstract': 512,
            'anime': 512,
            'ukiyo-e': 512,
        }

        assert self.outclass in configs, \
            f'Invalid StyleGAN class {self.outclass}, should be one of [{", ".join(configs.keys())}]'

        self.resolution = configs[self.outclass]
        self.name = f'StyleGAN-{self.outclass}'
        self.has_latent_residual = True
        self.load_model()
        self.set_noise_seed(0)

    def latent_space_name(self):
        return 'W' if self.w_primary else 'Z'

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    def load_model(self):
        checkpoint_root = os.environ.get('GANCONTROL_CHECKPOINT_DIR', Path(__file__).parent / 'checkpoints')
        checkpoint = Path(checkpoint_root) / f'stylegan/stylegan_{self.outclass}_{self.resolution}.pt'
        
        self.model = stylegan.StyleGAN_G(self.resolution).to(self.device)

        urls_tf = {
            'vases': 'https://thisvesseldoesnotexist.s3-us-west-2.amazonaws.com/public/network-snapshot-008980.pkl',
            'fireworks': 'https://mega.nz/#!7uBHnACY!quIW-pjdDa7NqnZOYh1z5UemWwPOW6HkYSoJ4usCg9U',
            'abstract': 'https://mega.nz/#!vCQyHQZT!zdeOg3VvT4922Z2UfxO51xgAfJD-NAK2nW7H_jMlilU',
            'anime': 'https://mega.nz/#!vawjXISI!F7s13yRicxDA3QYqYDL2kjnc2K7Zk3DwCIYETREmBP4',
            'ukiyo-e': 'https://drive.google.com/uc?id=1CHbJlci9NhVFifNQb3vCGu6zw4eqzvTd',
        }

        urls_torch = {
            'celebahq': 'https://drive.google.com/uc?export=download&id=1lGcRwNoXy_uwXkD6sy43aAa-rMHRR7Ad',
            'bedrooms': 'https://drive.google.com/uc?export=download&id=1r0_s83-XK2dKlyY3WjNYsfZ5-fnH8QgI',
            'ffhq': 'https://drive.google.com/uc?export=download&id=1GcxTcLDPYxQqcQjeHpLUutGzwOlXXcks',
            'cars': 'https://drive.google.com/uc?export=download&id=1aaUXHRHjQ9ww91x4mtPZD0w50fsIkXWt',
            'cats': 'https://drive.google.com/uc?export=download&id=1JzA5iiS3qPrztVofQAjbb0N4xKdjOOyV',
            'wikiart': 'https://drive.google.com/uc?export=download&id=1fN3noa7Rsl9slrDXsgZVDsYFxV0O08Vx',
        }

        if not checkpoint.is_file():
            os.makedirs(checkpoint.parent, exist_ok=True)
            if self.outclass in urls_torch:
                download_ckpt(urls_torch[self.outclass], checkpoint)
            else:
                checkpoint_tf = checkpoint.with_suffix('.pkl')
                if not checkpoint_tf.is_file():
                    download_ckpt(urls_tf[self.outclass], checkpoint_tf)
                print('Converting TensorFlow checkpoint to PyTorch')
                self.model.export_from_tf(checkpoint_tf)
        
        self.model.load_weights(checkpoint)

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max) # use (reproducible) global rand state

        rng = np.random.RandomState(seed)
        noise = torch.from_numpy(
                rng.standard_normal(512 * n_samples)
                .reshape(n_samples, 512)).float().to(self.device) #[N, 512]
        
        if self.w_primary:
            noise = self.model._modules['g_mapping'].forward(noise)
        
        return noise

    def get_max_latents(self):
        return 18

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError('StyleGAN: cannot change output class without reloading')

    def forward(self, x):
        out = self.model.forward(x, latent_is_w=self.w_primary)
        return 0.5*(out+1)

    # Run model only until given layer
    def partial_forward(self, x, layer_name):
        mapping = self.model._modules['g_mapping']
        G = self.model._modules['g_synthesis']
        trunc = self.model._modules.get('truncation', lambda x : x)

        if not self.w_primary:
            x = mapping.forward(x) # handles list inputs

        if isinstance(x, list):
            x = torch.stack(x, dim=1)
        else:
            x = x.unsqueeze(1).expand(-1, 18, -1)

        # Whole mapping
        if 'g_mapping' in layer_name:
            return

        x = trunc(x)
        if layer_name == 'truncation':
            return

        # Get names of children
        def iterate(m, name, seen):
            children = getattr(m, '_modules', [])
            if len(children) > 0:
                for child_name, module in children.items():
                    seen += iterate(module, f'{name}.{child_name}', seen)
                return seen
            else:
                return [name]

        # Generator
        batch_size = x.size(0)
        for i, (n, m) in enumerate(G.blocks.items()): # InputBlock or GSynthesisBlock
            if i == 0:
                r = m(x[:, 2*i:2*i+2])
            else:
                r = m(r, x[:, 2*i:2*i+2])

            children = iterate(m, f'g_synthesis.blocks.{n}', [])
            for c in children:
                if layer_name in c: # substring
                    return

        raise RuntimeError(f'Layer {layer_name} not encountered in partial_forward')


    def set_noise_seed(self, seed):
        G = self.model._modules['g_synthesis']

        def for_each_child(this, name, func):
            children = getattr(this, '_modules', [])
            for child_name, module in children.items():
                for_each_child(module, f'{name}.{child_name}', func)            
            func(this, name)

        def modify(m, name):
            if isinstance(m, stylegan.NoiseLayer):
                H, W = [int(s) for s in name.split('.')[2].split('x')]
                torch.random.manual_seed(seed)
                m.noise = torch.randn(1, 1, H, W, device=self.device, dtype=torch.float32)
                #m.noise = 1.0 # should be [N, 1, H, W], but this also works

        for_each_child(G, 'g_synthesis', modify)

class GANZooModel(BaseModel):
    def __init__(self, device, model_name):
        super(GANZooModel, self).__init__(model_name, 'default')
        self.device = device
        self.base_model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
            model_name, pretrained=True, useGPU=(device.type == 'cuda'))
        self.model = self.base_model.netG.to(self.device)
        self.name = model_name
        self.has_latent_residual = False

    def sample_latent(self, n_samples=1, seed=0, truncation=None):
        # Uses torch.randn
        noise, _ = self.base_model.buildNoiseData(n_samples)
        return noise

    # Don't bother for now
    def partial_forward(self, x, layer_name):
        return self.forward(x)

    def get_conditional_state(self, z):
        return z[:, -20:] # last 20 = conditioning

    def set_conditional_state(self, z, c):
        z[:, -20:] = c
        return z
    
    def forward(self, x):
        out = self.base_model.test(x)
        return 0.5*(out+1)


class ProGAN(BaseModel):
    def __init__(self, device, lsun_class=None):
        super(ProGAN, self).__init__('ProGAN', lsun_class)
        self.device = device

        # These are downloaded by GANDissect
        valid_classes = [ 'bedroom', 'churchoutdoor', 'conferenceroom', 'diningroom', 'kitchen', 'livingroom', 'restaurant' ]
        assert self.outclass in valid_classes, \
            f'Invalid LSUN class {self.outclass}, should be one of {valid_classes}'

        self.load_model()
        self.name = f'ProGAN-{self.outclass}'
        self.has_latent_residual = False

    def load_model(self):
        checkpoint_root = os.environ.get('GANCONTROL_CHECKPOINT_DIR', Path(__file__).parent / 'checkpoints')
        checkpoint = Path(checkpoint_root) / f'progan/{self.outclass}_lsun.pth'
        
        if not checkpoint.is_file():
            os.makedirs(checkpoint.parent, exist_ok=True)
            url = f'http://netdissect.csail.mit.edu/data/ganmodel/karras/{self.outclass}_lsun.pth'
            download_ckpt(url, checkpoint)

        self.model = proggan.from_pth_file(str(checkpoint.resolve())).to(self.device)

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max) # use (reproducible) global rand state
        noise = zdataset.z_sample_for_model(self.model, n_samples, seed=seed)[...]
        return noise.to(self.device)

    def forward(self, x):
        if isinstance(x, list):
            assert len(x) == 1, "ProGAN only supports a single global latent"
            x = x[0]
        
        out = self.model.forward(x)
        return 0.5*(out+1)

    # Run model only until given layer
    def partial_forward(self, x, layer_name):
        assert isinstance(self.model, torch.nn.Sequential), 'Expected sequential model'

        if isinstance(x, list):
            assert len(x) == 1, "ProGAN only supports a single global latent"
            x = x[0]

        x = x.view(x.shape[0], x.shape[1], 1, 1)
        for name, module in self.model._modules.items(): # ordered dict
            x = module(x)
            if name == layer_name:
                return

        raise RuntimeError(f'Layer {layer_name} not encountered in partial_forward')


class BigGAN(BaseModel):
    def __init__(self, device, resolution, class_name, truncation=1.0):
        super(BigGAN, self).__init__(f'BigGAN-{resolution}', class_name)
        self.device = device
        self.truncation = truncation
        self.load_model(f'biggan-deep-{resolution}')
        self.set_output_class(class_name or 'husky')
        self.name = f'BigGAN-{resolution}-{self.outclass}-t{self.truncation}'
        self.has_latent_residual = True

    # Default implementaiton fails without an internet
    # connection, even if the model has been cached
    def load_model(self, name):        
        if name not in biggan.model.PRETRAINED_MODEL_ARCHIVE_MAP:
            raise RuntimeError('Unknown BigGAN model name', name)
        
        checkpoint_root = os.environ.get('GANCONTROL_CHECKPOINT_DIR', Path(__file__).parent / 'checkpoints')
        model_path = Path(checkpoint_root) / name

        os.makedirs(model_path, exist_ok=True)
        
        model_file = model_path / biggan.model.WEIGHTS_NAME
        config_file = model_path / biggan.model.CONFIG_NAME
        model_url = biggan.model.PRETRAINED_MODEL_ARCHIVE_MAP[name]
        config_url = biggan.model.PRETRAINED_CONFIG_ARCHIVE_MAP[name]

        for filename, url in ((model_file, model_url), (config_file, config_url)):
            if not filename.is_file():
                print('Downloading', url)
                with open(filename, 'wb') as f:
                    if url.startswith("s3://"):
                        biggan.s3_get(url, f)
                    else:
                        biggan.http_get(url, f)

        self.model = biggan.BigGAN.from_pretrained(model_path).to(self.device)

    def sample_latent(self, n_samples=1, truncation=None, seed=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max) # use (reproducible) global rand state
        
        noise_vector = biggan.truncated_noise_sample(truncation=truncation or self.truncation, batch_size=n_samples, seed=seed)
        noise = torch.from_numpy(noise_vector) #[N, 128] 
        
        return noise.to(self.device)

    # One extra for gen_z
    def get_max_latents(self):
        return len(self.model.config.layers) + 1

    def get_conditional_state(self, z):
        return self.v_class

    def set_conditional_state(self, z, c):
        self.v_class = c
    
    def is_valid_class(self, class_id):
        if isinstance(class_id, int):
            return class_id < 1000
        elif isinstance(class_id, str):
            return biggan.one_hot_from_names([class_id.replace(' ', '_')]) is not None
        else:
            raise RuntimeError(f'Unknown class identifier {class_id}')

    def set_output_class(self, class_id):
        if isinstance(class_id, int):
            self.v_class = torch.from_numpy(biggan.one_hot_from_int([class_id])).to(self.device)
            self.outclass = f'class{class_id}'
        elif isinstance(class_id, str):
            self.outclass = class_id.replace(' ', '_')
            self.v_class = torch.from_numpy(biggan.one_hot_from_names([class_id])).to(self.device)
        else:
            raise RuntimeError(f'Unknown class identifier {class_id}')
    
    def forward(self, x):        
        # Duplicate along batch dimension
        if isinstance(x, list):
            c = self.v_class.repeat(x[0].shape[0], 1)
            class_vector = len(x)*[c]
        else:
            class_vector = self.v_class.repeat(x.shape[0], 1)
        out = self.model.forward(x, class_vector, self.truncation)  # [N, 3, 128, 128], in [-1, 1]
        return 0.5*(out+1)

    # Run model only until given layer
    # Used to speed up PCA sample collection
    def partial_forward(self, x, layer_name):
        if layer_name in ['embeddings', 'generator.gen_z']:
            n_layers = 0
        elif 'generator.layers' in layer_name:
            layer_base = re.match('^generator\.layers\.[0-9]+', layer_name)[0]
            n_layers = int(layer_base.split('.')[-1]) + 1
        else:
            n_layers = len(self.model.config.layers)

        if not isinstance(x, list):
            x = self.model.n_latents*[x]

        if isinstance(self.v_class, list):
            labels = [c.repeat(x[0].shape[0], 1) for c in class_label]
            embed = [self.model.embeddings(l) for l in labels]
        else:
            class_label = self.v_class.repeat(x[0].shape[0], 1)
            embed = len(x)*[self.model.embeddings(class_label)]
        
        assert len(x) == self.model.n_latents, f'Expected {self.model.n_latents} latents, got {len(x)}'
        assert len(embed) == self.model.n_latents, f'Expected {self.model.n_latents} class vectors, got {len(class_label)}'

        cond_vectors = [torch.cat((z, e), dim=1) for (z, e) in zip(x, embed)]

        # Generator forward
        z = self.model.generator.gen_z(cond_vectors[0])
        z = z.view(-1, 4, 4, 16 * self.model.generator.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()

        cond_idx = 1
        for i, layer in enumerate(self.model.generator.layers[:n_layers]):
            if isinstance(layer, biggan.GenBlock):
                z = layer(z, cond_vectors[cond_idx], self.truncation)
                cond_idx += 1
            else:
                z = layer(z)

        return None

    
    
#Cycle GAN implementation

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGAN(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

# Version 1: separate parameters
@singledispatch
def get_model(name, output_class, device, **kwargs):
    # Check if optionally provided existing model can be reused
    inst = kwargs.get('inst', None)
    model = kwargs.get('model', None)
    
    if inst or model:
        cached = model or inst.model
        
        network_same = (cached.model_name == name)
        outclass_same = (cached.outclass == output_class)
        can_change_class = ('BigGAN' in name)
        
        if network_same and (outclass_same or can_change_class):
            cached.set_output_class(output_class)
            return cached
    
    if name == 'DCGAN':
        import warnings
        warnings.filterwarnings("ignore", message="nn.functional.tanh is deprecated")
        model = GANZooModel(device, 'DCGAN')
    elif name == 'ProGAN':
        model = ProGAN(device, output_class)
    elif 'BigGAN' in name:
        assert '-' in name, 'Please specify BigGAN resolution, e.g. BigGAN-512'
        model = BigGAN(device, name.split('-')[-1], class_name=output_class)
    elif name == 'StyleGAN':
        model = StyleGAN(device, class_name=output_class)
    elif name == 'StyleGAN2':
        model = StyleGAN2(device, class_name=output_class)
    elif name == 'CycleGAN':
        model = CycleGAN(device, class_name=output_class)
    else:
        raise RuntimeError(f'Unknown model {name}')

    return model

# Version 2: Config object
@get_model.register(Config)
def _(cfg, device, **kwargs):
    kwargs['use_w'] = kwargs.get('use_w', cfg.use_w) # explicit arg can override cfg
    return get_model(cfg.model, cfg.output_class, device, **kwargs)

# Version 1: separate parameters
@singledispatch
def get_instrumented_model(name, output_class, layers, device, **kwargs):
    model = get_model(name, output_class, device, **kwargs)
    model.eval()

    inst = kwargs.get('inst', None)
    if inst:
        inst.close()

    if not isinstance(layers, list):
        layers = [layers]

    # Verify given layer names
    module_names = [name for (name, _) in model.named_modules()]
    for layer_name in layers:
        if not layer_name in module_names:
            print(f"Layer '{layer_name}' not found in model!")
            print("Available layers:", '\n'.join(module_names))
            raise RuntimeError(f"Unknown layer '{layer_name}''")
    
    # Reset StyleGANs to z mode for shape annotation
    if hasattr(model, 'use_z'):
        model.use_z()

    from netdissect.modelconfig import create_instrumented_model
    inst = create_instrumented_model(SimpleNamespace(
        model = model,
        layers = layers,
        cuda = device.type == 'cuda',
        gen = True,
        latent_shape = model.get_latent_shape()
    ))

    if kwargs.get('use_w', False):
        model.use_w()

    return inst

# Version 2: Config object
@get_instrumented_model.register(Config)
def _(cfg, device, **kwargs):
    kwargs['use_w'] = kwargs.get('use_w', cfg.use_w) # explicit arg can override cfg
    return get_instrumented_model(cfg.model, cfg.output_class, cfg.layer, device, **kwargs)
