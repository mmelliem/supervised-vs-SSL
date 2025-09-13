import torch
import numpy as np
import os
import glob
import random
import sys
sys.path.append('/home/melan/supervised-vs-SSL/supervised_vs_ssl/models/byol-a/v2')
from byol_a2.models import AudioNTT2022Encoder

print(torch.__version__)
print(torch.version.cuda)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioNTT2022Encoder(n_mels=64)
model.to(device)
model.eval()

# List of layer names to extract (from model.named_modules())
layer_names = [
    'features.0',  # Conv2d
    'features.1',  # BatchNorm2d
    'features.2',  # ReLU
    'features.3',  # MaxPool2d
    'features.4',  # Conv2d
    'features.5',  # BatchNorm2d
    'features.6',  # ReLU
    'features.7',  # MaxPool2d
    'fc.0',        # Linear
    'fc.1',        # ReLU
    'fc.2',        # Dropout
    'fc.3',        # Linear
    'fc.4',        # ReLU
]

results_dir = '/home/melan/supervised-vs-SSL/data/results'
os.makedirs(results_dir, exist_ok=True)
for lname in layer_names:
    os.makedirs(os.path.join(results_dir, lname.replace('.', '_')), exist_ok=True)

def extract_and_save_features(spec_path, idx):
    spec = np.load(spec_path)
    x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # shape: (1, 1, n_mels, n_frames)
    feats = {}
    hooks = []
    def save_hook(name):
        def hook(module, input, output):
            feats[name] = output.detach().cpu().numpy()
        return hook
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(save_hook(name)))
    with torch.no_grad():
        _ = model(x)
    for h in hooks:
        h.remove()
    # Save each layer's features
    for lname in layer_names:
        out_dir = os.path.join(results_dir, lname.replace('.', '_'))
        np.save(os.path.join(out_dir, f'{idx:04d}.npy'), feats[lname])

spec_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/500_for_byola/'
spec_files = glob.glob(os.path.join(spec_dir, '*.npy'))
random.shuffle(spec_files)
sampled_files = spec_files[:500]  # Select 500 random files

for idx, spec_path in enumerate(sampled_files):
    extract_and_save_features(spec_path, idx)