import torch, os, yaml, random, numpy as np

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_ckpt(path, model_state, optim_state, epoch, best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model': model_state, 'optim': optim_state, 'epoch': epoch, 'best': best}, path)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
