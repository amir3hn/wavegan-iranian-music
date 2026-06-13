import torch
import random
import numpy as np
import logging
import os

target_signals_dir ='./data'

model_prefix = 'exp1' 
n_iterations = 100000
use_batchnorm = False
lr_g = 1e-5
lr_d = 2e-5 
beta1 = 0.5
beta2 = 0.9
decay_lr = True 
generator_batch_size_factor = 1 
n_critic = 1 
validate = False
p_coeff = 10
batch_size = 32
noise_latent_dim = 100  
model_capacity_size = 32    
store_cost_every = 300
progress_bar_step_iter_size = 400

take_backup = True
backup_every_n_iters = 1000
save_samples_every = 1000
output_dir = './output'
if not(os.path.isdir(output_dir)):
    os.makedirs(output_dir)

window_length = 262144   #[16384, 32768, 65536, 131072, 262144] 
sampling_rate = 16000
normalize_audio = True 
num_channels = 1


LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)

cuda = torch.cuda.is_available()
if cuda:
    # Use all available GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)  # Optional: Ensure the default GPU is set
else:
    device = torch.device("cpu")
# update the seed
manual_seed = 2019
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
if cuda:
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.empty_cache()
