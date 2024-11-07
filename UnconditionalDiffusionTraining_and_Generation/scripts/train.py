#Imports
import torch
import numpy as np
from src.script_util import create_model, create_gaussian_diffusion
from src.train_util import TrainLoop
from torch.utils.data import DataLoader, TensorDataset
from src.dist_util import setup_dist, dev
from src.logger import configure, log

## Setup
torch.manual_seed(42)
np.random.seed(42)
setup_dist()
configure(dir="/add/path/here",format_strs=["stdout","log","tensorboard_new"])

## HyperParams (Change according to the case)
batch_size = 16
test_batch_size = 16

image_size= 128
num_channels= 128
num_res_blocks= 2
num_heads=4
num_head_channels=64
attention_resolutions="32,16,8"
channel_mult = None

steps=1000
noise_schedule="cosine"

microbatch= -1
lr = 5e-5
ema_rate="0.9999"
log_interval=1000
save_interval=10000
lr_anneal_steps=0

## Data Preprocessing
train_data = np.load('/add/path/here')
valid_data = np.load('/add/path/here')

max_val, min_val = np.max(train_data, keepdims=True), np.min(train_data, keepdims=True)
norm_train_data = -1 + (train_data - min_val)*2. / (max_val - min_val)
norm_valid_data = -1 + (valid_data - min_val)*2. / (max_val - min_val)

norm_train_data = torch.from_numpy(norm_train_data[:, None, ...])
norm_valid_data = torch.from_numpy(norm_valid_data[:, None, ...])

log("creating data loader...")

dl_train = DataLoader(TensorDataset(norm_train_data), batch_size=batch_size, shuffle=True)
dl_valid = DataLoader(TensorDataset(norm_valid_data), batch_size=test_batch_size, shuffle=True)

def dl_iter(dl):
    while True:
        yield from dl 

## Unet Model
log("creating model and diffusion...")

unet_model = create_model(image_size=image_size,
                          num_channels= num_channels,
                          num_res_blocks= num_res_blocks,
                          num_heads=num_heads,
                          num_head_channels=num_head_channels,
                          attention_resolutions=attention_resolutions,
                          channel_mult=channel_mult
                        )

unet_model.to(dev())

## Gaussian Diffusion
diff_model = create_gaussian_diffusion(steps=steps,
                                       noise_schedule=noise_schedule
                                    )

## Training Loop
log("training...")

train_uncond_model = TrainLoop(
                                model=unet_model,
                                diffusion=diff_model,
                                train_data = dl_iter(dl_train),
                                valid_data=dl_iter(dl_valid),
                                batch_size= batch_size,
                                microbatch= microbatch,
                                lr = lr,
                                ema_rate=ema_rate,
                                log_interval=log_interval,
                                save_interval=save_interval,
                                lr_anneal_steps=lr_anneal_steps,
                                resume_checkpoint="")

train_uncond_model.run_loop()