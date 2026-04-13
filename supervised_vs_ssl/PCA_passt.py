import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import seaborn as sns
import glob
import os

OUTPUT_DIR = '/home/mellie/supervised-vs-SSL/supervised_vs_ssl/PCA_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting PaSST Analysis...")

PASST_FEATURES = '/home/mellie/supervised-vs-SSL/data/passt_features'

passt_labels_df = pd.read_csv('/home/mellie/supervised-vs-SSL/supervised_vs_ssl/passt_predictions.csv')
print(f"Loaded PaSST labels with {len(passt_labels_df)} entries")

music_genres = ['Hip-Hop|Nerdcore|Alternative Hip-Hop', 'Reggae - Dancehall', 'Latin America', 'Hip-Hop', 'New Age|Instrumental', 'Folk', 'Hip-Hop|Alternative Hip-Hop|Rap', 'Ambient', 'Pop', 'Rock|Noise-Rock']
speech_commands = ['right', 'no', 'off', 'up', 'yes', 'down', 'cat', 'stop', 'dog', 'go', 'marvin', 'sheila', 'nine', 'on', 'eight', 'bird', 'four', 'two', 'seven', 'one', 'five', 'three', 'tree', 'happy', 'wow', 'six', 'house', 'left', 'zero', 'bed']

passt_task_labels = np.array([(1 if label in speech_commands else 0) for label in passt_labels_df['actual_label']])

print(f"Number of speech samples: {np.sum(passt_task_labels == 1)}")
print(f"Number of music samples: {np.sum(passt_task_labels == 0)}")

def process_passt_features():
    """
    Process PaSST features, run PCA (2 components), and save a PNG and CSV per layer.
    CSV contains: feature_file, task, pc1, pc2
    """
    passt_output_dir = os.path.join(OUTPUT_DIR, 'passt')
    os.makedirs(passt_output_dir, exist_ok=True)

    layer_dirs = sorted([d for d in os.listdir(PASST_FEATURES) if os.path.isdir(os.path.join(PASST_FEATURES, d))])

    layer_pca_results = {}
    layer_info = []

    for layer_name in layer_dirs:
        feature_dir = os.path.join(PASST_FEATURES, layer_name)
        print(f"Processing layer: {layer_name}")

        feature_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
        all_features = []
        filenames = []

        for file_path in tqdm(feature_files, desc=f"Loading {layer_name}"):
            features = np.load(file_path)
            if features.ndim > 2:
                features = features.reshape(features.shape[0], -1)
            elif features.ndim == 1:
                features = features.reshape(1, -1)
            all_features.append(features)
            n_rows = features.shape[0]
            filenames.extend([os.path.basename(file_path)] * n_rows)

        if not all_features:
            print(f"No features in {layer_name}, skipping")
            continue

        features = np.vstack(all_features)
        print(f"Loaded {features.shape[0]} samples x {features.shape[1]} features")

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)

        n_samples = features_pca.shape[0]
        labels_trim = passt_task_labels[:n_samples] if len(passt_task_labels) >= n_samples else np.pad(passt_task_labels, (0, n_samples - len(passt_task_labels)), 'constant')
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = sns.color_palette('coolwarm', as_cmap=True)
        sc = ax.scatter(features_pca[:, 0], features_pca[:, 1], c=labels_trim, cmap=cmap, alpha=0.7, s=30)
        ax.set_title(f'Layer: {layer_name} (n={n_samples})')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(sc, ax=ax, label='Task (0=Music, 1=Speech)')
        out_png = os.path.join(passt_output_dir, f"{layer_name}_pca.png")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

        if len(filenames) >= n_samples:
            fn_trim = filenames[:n_samples]
        else:
            fn_trim = filenames + ["unknown"] * (n_samples - len(filenames))
        df_layer = pd.DataFrame({
            'feature_file': fn_trim,
            'task': labels_trim,
            'pc1': features_pca[:, 0],
            'pc2': features_pca[:, 1],
        })
        out_csv = os.path.join(passt_output_dir, f"{layer_name}_pca.csv")
        df_layer.to_csv(out_csv, index=False)

        layer_pca_results[layer_name] = features_pca
        ev = pca.explained_variance_ratio_
        layer_info.append({
            'layer': layer_name,
            'n_samples': n_samples,
            'n_features': features.shape[1],
            'explained_variance_pc1': float(ev[0]) if len(ev) > 0 else None,
            'explained_variance_pc2': float(ev[1]) if len(ev) > 1 else None,
            'total_explained_variance': float(ev.sum()) if ev.size else None,
            'png': out_png,
            'csv': out_csv,
        })

    pd.DataFrame(layer_info).to_csv(os.path.join(OUTPUT_DIR, 'layer_info.csv'), index=False)
    return layer_pca_results


def plot_pca_results(layer_pca_results):
    """No-op: per-layer PNGs are already saved by `process_passt_features`."""
    print("Per-layer PCA PNGs saved under:", os.path.join(OUTPUT_DIR, 'passt'))

print("Processing features and applying PCA...")
layer_pca_results = process_passt_features()

print("Creating PCA visualizations...")
plot_pca_results(layer_pca_results)