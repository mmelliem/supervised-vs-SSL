import os
import glob
import torch
import numpy as np
from hear21passt.base import get_basic_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_basic_model(mode="embedding")
model.eval()
model = model.to(device)

spec_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/your_spectrogram_folder'  # <-- change this!
results_dir = '/home/melan/supervised-vs-SSL/data/results/passt/intermediate_features'
os.makedirs(results_dir, exist_ok=True)

spec_files = glob.glob(os.path.join(spec_dir, '**', '*.npy'), recursive=True)

# Example: Extract intermediate features using hooks
layer_names = ['net.transformer.layers.0', 'net.transformer.layers.1', 'net.transformer.layers.2']  # adjust as needed
for lname in layer_names:
    os.makedirs(os.path.join(results_dir, lname.replace('.', '_')), exist_ok=True)

def extract_and_save_intermediate(spec_path, idx):
    spec = np.load(spec_path)
    x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(device)
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
    for lname in layer_names:
        out_dir = os.path.join(results_dir, lname.replace('.', '_'))
        np.save(os.path.join(out_dir, f'{idx:06d}.npy'), feats[lname])

for idx, spec_path in enumerate(spec_files):
    extract_and_save_intermediate(spec_path, idx)
    if idx < 5 or idx % 100 == 0:
        print(f"Processed {idx+1} files...", flush=True)