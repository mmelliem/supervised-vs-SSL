import os
import pandas as pd
import glob
import csv

# --- Paths ---
speech_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/noisy_byola_speech'
music_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/noisy_byola_music'
music_metadata_path = '/home/melan/supervised-vs-SSL/data/fma_data/fma_metadata/tracks.csv'
genre_map_path = '/home/melan/supervised-vs-SSL/data/fma_data/fma_metadata/raw_genres.csv'
output_csv_path = '/home/melan/supervised-vs-SSL/data/labels_mapping.csv'  # <-- Change this to wherever you want

# --- Load metadata ---
music_metadata = pd.read_csv(music_metadata_path, index_col=0, header=[0, 1])
genre_map = pd.read_csv(genre_map_path, index_col='genre_id')['genre_title'].to_dict()

# --- Collect all .npy files ---
speech_files = glob.glob(os.path.join(speech_dir, '**', '*.npy'), recursive=True)
music_files = glob.glob(os.path.join(music_dir, '**', '*.npy'), recursive=True)
all_files = speech_files + music_files

# --- Write CSV ---
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file_path', 'label'])  # header

    for idx, spec_path in enumerate(all_files):
        fname = os.path.basename(spec_path)
        if spec_path.startswith(speech_dir):
            label = fname.split('_')[0]
        elif spec_path.startswith(music_dir):
            track_id_str = fname[3:6]
            try:
                track_id = int(track_id_str)
            except ValueError:
                track_id = None
            if track_id is not None and track_id in music_metadata.index:
                genres_str = music_metadata.loc[track_id, ('track', 'genres')]
                genre_ids = eval(genres_str) if pd.notnull(genres_str) else []
                genres = [genre_map.get(gid, 'unknown') for gid in genre_ids]
            else:
                genres = []
            label = '|'.join(genres)  # join multiple genres as a string
        else:
            label = 'unknown'
        writer.writerow([spec_path, label])
        if idx < 5 or idx % 100 == 0:
            print(f"Processed {idx+1} files...", flush=True)

print(f"Labels CSV saved to: {output_csv_path}")