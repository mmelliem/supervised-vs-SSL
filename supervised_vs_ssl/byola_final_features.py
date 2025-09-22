import argparse
import numpy as np
import torch
from pathlib import Path
from byol_a2.models import AudioNTT2022Encoder
import sys
sys.path.append(
    '/home/mellie/supervised-vs-SSL/supervised_vs_ssl/models/byol-a/v2'
)

def main(input_dir: Path, output_dir: Path, device: str = None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = AudioNTT2022Encoder(n_mels=64)
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob('*.npy'))
    if not files:
        print('No .npy files found under', input_dir)
        return

    for i, p in enumerate(files):
        spec = np.load(p)
        x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model(x)
        out_name = p.with_suffix('.npy').name
        out_path = output_dir / out_name
        np.save(out_path, feats.cpu().numpy())
        if i < 5 or (i + 1) % 100 == 0:
            print(f'[{i+1}/{len(files)}] -> saved {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, default=Path('/home/mellie/supervised-vs-SSL/data/2.2_processed_byola'))
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'byola_features' / 'z_final_embeddings')
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()
    main(args.input_dir, args.output_dir, args.device)