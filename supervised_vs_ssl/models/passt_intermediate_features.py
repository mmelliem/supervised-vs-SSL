import os
import glob
import torch
import numpy as np
import random
from hear21passt.base import get_basic_model
import soundfile as sf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_basic_model(mode="embed_only")
model.eval()
model = model.to(device)

# --- Paths ---
speech_wav_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/10s_speech'
music_wav_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/10s_music'
results_dir = '/home/melan/supervised-vs-SSL/data/results/passt/intermediate_embeddings'
os.makedirs(results_dir, exist_ok=True)

import random

# --- Collect and randomize all .wav files ---
speech_files = glob.glob(os.path.join(speech_wav_dir, '**', '*.wav'), recursive=True)
music_files = glob.glob(os.path.join(music_wav_dir, '**', '*.wav'), recursive=True)

# Randomly sample 250 from each (if there are enough files)
speech_sample = random.sample(speech_files, min(250, len(speech_files)))
music_sample = random.sample(music_files, min(250, len(music_files)))

all_files = speech_sample + music_sample
random.shuffle(all_files)
# ...existing code...

for name, module in model.named_modules():
    print(f"Layer: {name}, Type: {type(module)}")

# Use correct layer names
layer_names = [f'net.blocks.{i}' for i in range(12)]

# Create output folders for each layer
for lname in layer_names:
    os.makedirs(os.path.join(results_dir, lname.replace('.', '_')), exist_ok=True)

def extract_and_save_intermediate(wav_path):
    waveform, sr = sf.read(wav_path)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
    feats = {}
    hooks = []
    def save_hook(name):
        def hook(module, input, output):
            print(f"Hook fired for {name}, output shape: {output.shape}")
            feats[name] = output.detach().cpu().numpy()
        return hook
    for name, module in model.named_modules():
        if name in [f'net.blocks.{i}' for i in range(12)]:
            hooks.append(module.register_forward_hook(save_hook(name)))
    with torch.no_grad():
        out = model(x)
        print(f"Model output shape: {out.shape}")
        print(f"Features captured for {wav_path}: {list(feats.keys())}")
    for h in hooks:
        h.remove()
    orig_name = os.path.splitext(os.path.basename(wav_path))[0]
    for lname in layer_names:
        if lname in feats:
            out_dir = os.path.join(results_dir, lname.replace('.', '_'))
            layer_feat = feats[lname].flatten()[:500]
            np.save(os.path.join(out_dir, f'{orig_name}.npy'), layer_feat)

# ...existing code...
for idx, wav_path in enumerate(all_files):
    extract_and_save_intermediate(wav_path)
    if idx < 5 or idx % 100 == 0:
        print(f"Processed {idx+1} files...", flush=True)