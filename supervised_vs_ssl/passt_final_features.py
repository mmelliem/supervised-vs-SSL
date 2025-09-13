import torch
import numpy as np
import os
import glob
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
results_dir = '/home/melan/supervised-vs-SSL/data/results/passt/final_embeddings'
os.makedirs(results_dir, exist_ok=True)

# --- Collect and randomize all .wav files ---
speech_files = glob.glob(os.path.join(speech_wav_dir, '**', '*.wav'), recursive=True)
music_files = glob.glob(os.path.join(music_wav_dir, '**', '*.wav'), recursive=True)
all_files = speech_files + music_files
random.shuffle(all_files)

# --- Extract features and save with original filename ---
for idx, wav_path in enumerate(all_files):
    try:
        waveform, sr = sf.read(wav_path)
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")

    if len(waveform.shape) > 1:  # stereo to mono
        waveform = np.mean(waveform, axis=1)
    x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(x)
    orig_name = os.path.splitext(os.path.basename(wav_path))[0]
    np.save(os.path.join(results_dir, f'{orig_name}.npy'), embedding.cpu().numpy())
    if idx < 5 or idx % 100 == 0:
        print(f"Processed {idx+1} files...", flush=True)   
    print(f"Processing: {wav_path} -> {orig_name}.npy")
    

    import os

    embeddings_dir = '/home/melan/supervised-vs-SSL/data/results/passt/final_embeddings'
    embedding_files = [os.path.splitext(f)[0] for f in os.listdir(embeddings_dir) if f.endswith('.npy')]
    print("First 10 embedding files:", embedding_files[:10])
    print("Total embeddings:", len(embedding_files))

    import pandas as pd
    df = pd.read_csv('/home/melan/supervised-vs-SSL/data/passt_labels_mapping.csv')
    csv_basenames = [os.path.splitext(os.path.basename(f))[0] for f in df['file_path'].astype(str)]
    print("First 10 CSV basenames:", csv_basenames[:10])
    print("Total CSV entries:", len(csv_basenames))
    
    import pandas as pd
    import os

    embeddings_dir = '/home/melan/supervised-vs-SSL/data/results/passt/final_embeddings'
    label_csv_path = '/home/melan/supervised-vs-SSL/data/passt_labels_mapping.csv'

    embedding_files = [os.path.splitext(f)[0] for f in os.listdir(embeddings_dir) if f.endswith('.npy')]
    df = pd.read_csv(label_csv_path)
    csv_basenames = [os.path.splitext(os.path.basename(f))[0] for f in df['file_path'].astype(str)]

    print("First 5 embedding files:", embedding_files[:5])
    print("First 5 CSV basenames:", csv_basenames[:5])

    print(sorted([os.path.basename(f) for f in speech_files[:10]]))
print(sorted([os.path.basename(f) for f in music_files[:10]]))
    