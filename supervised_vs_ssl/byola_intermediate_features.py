import argparse
import torch
import os
import glob
import sys
import numpy as np
from byol_a2.models import AudioNTT2022Encoder

sys.path.append('/home/mellie/supervised-vs-SSL/supervised_vs_ssl/models/byol-a/v2')
spec_dir = '/home/mellie/supervised-vs-SSL/data/2.2_processed_byola'

def main():
    parser = argparse.ArgumentParser(description='Extract BYOL-A intermediate layer features (CPU-friendly)')
    parser.add_argument('--spec-dir', default='/home/mellie/supervised-vs-SSL/data/2.2_processed_byola', help='Directory with preprocessed spectrogram .npy files')
    parser.add_argument('--out-dir', default='/home/mellie/supervised-vs-SSL/data/byola_features', help='Where to write per-layer features')
    parser.add_argument('--device', default='cpu', help="Device to run on (default 'cpu')")
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of files to process')
    args = parser.parse_args()

    print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())

    device = torch.device(args.device if args.device else 'cpu')

    model = AudioNTT2022Encoder(n_mels=64)
    model.to(device)
    model.eval()

    layer_names = [
        'features.0', 'features.1', 'features.2', 'features.3',
        'features.4', 'features.5', 'features.6', 'features.7',
        'fc.0', 'fc.1', 'fc.2', 'fc.3', 'fc.4',
    ]

    results_dir = args.out_dir
    os.makedirs(results_dir, exist_ok=True)
    for lname in layer_names:
        os.makedirs(os.path.join(results_dir, lname.replace('.', '_')), exist_ok=True)

    def extract_and_save_features(spec_path):
        spec = np.load(spec_path)
        x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        feats = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                feats[name] = output.detach().cpu().numpy()
            return hook

        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            _ = model(x)

        for h in hooks:
            h.remove()

        base = os.path.basename(spec_path)
        if base.endswith('.npy'):
            base = base[:-4]

        for lname in layer_names:
            out_dir = os.path.join(results_dir, lname.replace('.', '_'))
            arr = feats.get(lname)
            if arr is None:
                print(' Warning: no features for', lname, 'from', spec_path)
                continue
            out_path = os.path.join(out_dir, f'{base}.npy')
            np.save(out_path, arr)

    spec_dir = args.spec_dir
    files = sorted(glob.glob(os.path.join(spec_dir, '*.npy')))
    if args.limit:
        files = files[:args.limit]

    print('Found', len(files), 'spec files in', spec_dir)
    for i, spec_path in enumerate(files, 1):
        print(f'[{i}/{len(files)}] processing', spec_path)
        try:
            extract_and_save_features(spec_path)
        except Exception as e:
            print(' Error processing', spec_path, e)

if __name__ == '__main__':
    main()