import torch
import numpy as np
import os
import glob
import random
import sys
import json
import pandas as pd

sys.path.append('/home/melan/supervised-vs-SSL/supervised_vs_ssl/models/byol-a/v2')
from byol_a2.models import AudioNTT2022Encoder

print(torch.__version__)
print(torch.version.cuda)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioNTT2022Encoder(n_mels=64)
model.to(device)
model.eval()

speech_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/noisy_byola_speech'
music_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/noisy_byola_music'
spec_files = glob.glob(os.path.join(speech_dir, '**', '*.npy'), recursive=True) + \
             glob.glob(os.path.join(music_dir, '**', '*.npy'), recursive=True)

random.shuffle(spec_files)

results_dir = '/home/melan/supervised-vs-SSL/data/results/byola/final_embeddings'
os.makedirs(results_dir, exist_ok=True)

# --- Load music genre metadata ---
# Adjust path and column names as needed
music_metadata_path = '/home/melan/supervised-vs-SSL/data/fma_data/fma_metadata/tracks.csv'
music_metadata = pd.read_csv(music_metadata_path, index_col=0, header=[0, 1])
genre_map = pd.read_csv('/home/melan/supervised-vs-SSL/data/fma_data/fma_metadata/raw_genres.csv', index_col='genre_id')['genre_title'].to_dict()


def get_label(spec_path):
    fname = os.path.basename(spec_path)
    if spec_path.startswith(speech_dir):
        label = fname.split('_')[0]
    elif spec_path.startswith(music_dir):
        track_id = int(fname[3:5])
        genres_str = music_metadata.loc[track_id, ('track', 'genres')]
        genres = eval(genres_str) if pd.notnull(genres_str) else []
        label = genres
    else:
        label = 'unknown'
    return label

def extract_final_embedding(spec_path, idx):
    spec = np.load(spec_path)
    x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(x)
    np.save(os.path.join(results_dir, f'{idx:06d}.npy'), feats.cpu().numpy())

mapping = []
labels = []

for idx, spec_path in enumerate(spec_files):
    fname = os.path.basename(spec_path)
    if spec_path.startswith(speech_dir):
        speech_label = fname.split('_')[0]
        label = speech_label
        print(f"[SPEECH] File {idx}: {spec_path}", flush=True)
        print(f"  Speech label (before _): {speech_label}", flush=True)
    elif spec_path.startswith(music_dir):
        track_id_str = fname[3:6]  # characters 3-5
        try:
            track_id = int(track_id_str)
        except ValueError:
            track_id = None
        print(f"[MUSIC] File {idx}: {spec_path}", flush=True)
        print(f"  Track ID from filename (chars 3-5): {track_id_str}", flush=True)
        print(f"  Track ID used for metadata lookup: {track_id}", flush=True)
        if track_id is not None and track_id in music_metadata.index:
            genres_str = music_metadata.loc[track_id, ('track', 'genres')]
            genre_ids = eval(genres_str) if pd.notnull(genres_str) else []
            genres = [genre_map.get(gid, 'unknown') for gid in genre_ids]
        else:
            genres = []
        label = genres
        print(f"  Genres from metadata: {genres}", flush=True)
    else:
        label = 'unknown'
        print(f"[UNKNOWN] File {idx}: {spec_path}", flush=True)
    mapping.append(spec_path)
    labels.append(label)
    print(f"  Embedding file name: {idx:06d}.npy", flush=True)
    print(f"  Label to be saved: {label}", flush=True)
    extract_final_embedding(spec_path, idx)
    if idx < 5 or idx % 100 == 0:
        print("-" * 40, flush=True)
    if idx % 100 == 0:
        print(f"Processed {idx+1} files so far...", flush=True)

with open(os.path.join(results_dir, 'filename_mapping.json'), 'w') as f:
    json.dump(mapping, f)
with open(os.path.join(results_dir, 'labels.json'), 'w') as f:
    json.dump(labels, f)
print("Saved labels.json and filename_mapping.json to", results_dir, flush=True)