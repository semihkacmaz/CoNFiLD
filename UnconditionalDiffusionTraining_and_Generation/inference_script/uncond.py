## Imports
import torch
import numpy as np
from src.script_util import create_model, create_gaussian_diffusion

## Setup
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
  
device = torch.device(dev)  

torch.manual_seed(42)
np.random.seed(42)

## Hyperparams
test_batch_size = 1
time_length = 32
latent_length = 256

image_size= 256
num_channels= 128
num_res_blocks= 2
num_heads=4
num_head_channels=64
attention_resolutions="32,16,8"

steps=1000
noise_schedule="cosine"

## Create model and diffusion
unet_model = create_model(image_size=image_size,
                          num_channels= num_channels,
                          num_res_blocks= num_res_blocks,
                          num_heads=num_heads,
                          num_head_channels=num_head_channels,
                          attention_resolutions=attention_resolutions
                        )

unet_model.load_state_dict(torch.load('/add/path/here'))
unet_model.to(device);

diff_model = create_gaussian_diffusion(steps=steps,
                                       noise_schedule=noise_schedule
                                    )

## Unconditional sample
sample_fn = diff_model.p_sample_loop
samples = sample_fn(unet_model, (test_batch_size, 1, time_length, latent_length))

np.save('/add/path/here',(samples[:, 0]).detach().cpu().numpy())