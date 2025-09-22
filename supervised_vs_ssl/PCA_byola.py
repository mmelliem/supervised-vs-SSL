import numpy as np
import pandas as pd
from pathlib import Path
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'byola_features'
OUT = ROOT / 'PCA_results_byola'
OUT.mkdir(parents=True, exist_ok=True)

mapping_csv = DATA / 'mapping_features0_binary_labels.csv'
if not mapping_csv.exists():
    mapping_csv = DATA / 'mapping_features0_full_labels.csv'

map_df = pd.read_csv(mapping_csv)
map_df['label'] = map_df['label'].fillna('unknown')

layers = sorted(
    [
        p
        for p in DATA.iterdir()
        if p.is_dir()
        and (
            p.name.startswith('features_')
            or p.name.startswith('fc_')
            or p.name == 'z_final_embeddings'
        )
    ]
)

summary = []
for layer in layers:
    print('Processing', layer.name)
    files = sorted(list(layer.glob('*.npy')))
    rows = []
    labels = []
    fnames = []
    for _, row in map_df.iterrows():
        fname = row['feature_file']
        if (layer / fname).exists():
            arr = np.load(layer / fname)
            rows.append(arr.reshape(-1).astype(np.float32))
            labels.append(row['label'])
            fnames.append(fname)
    if not rows:
        print('  no files for', layer.name)
        continue
    X = np.vstack(rows)
    print('  loaded X', X.shape)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)

    unique = list(dict.fromkeys(labels))
    if COLORS is None:
        cmap = plt.get_cmap('tab10')
        color_map = {lab: cmap(i % 10) for i, lab in enumerate(unique)}
    else:
        color_map = {lab: COLORS.get(lab, '#7f7f7f') for lab in unique}

    fig, ax = plt.subplots(figsize=(6, 5))
    for lab in unique:
        idx = [i for i, l in enumerate(labels) if l == lab]
        pts = Z[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=[color_map[lab]],
            label=f"{lab} ({len(idx)})",
            s=18,
            alpha=0.8,
        )
    ax.set_title(f'BYOL-A PCA: {layer.name} (n={Z.shape[0]})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    out_png = OUT / f'pca_{layer.name}.png'
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    df_layer = pd.DataFrame({
        'feature_file': fnames,
        'label': labels,
        'pc1': Z[:, 0],
        'pc2': Z[:, 1],
    })
    out_csv = OUT / f'pca_{layer.name}.csv'
    df_layer.to_csv(out_csv, index=False)

    summary.append(
        (
            layer.name,
            Z.shape[0],
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
            str(out_png),
            str(out_csv),
        )
    )

summary_df = pd.DataFrame(summary, columns=['layer', 'n_samples', 'pc1_var', 'pc2_var', 'png', 'csv'])
summary_df.to_csv(OUT / 'pca_per_layer_summary.csv', index=False)

print('Done. Outputs in', OUT)
