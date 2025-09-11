import os
import numpy as np
import librosa
import glob
import soundfile as sf
import matplotlib.pyplot as plt
import random

noise_files = sorted(glob.glob('/home/melan/supervised-vs-SSL/data/noise_data/scenes_stereo/*.wav', recursive=True))
speech_files = glob.glob('/home/melan/supervised-vs-SSL/data/speech_data/speechcommands/**/*.wav', recursive=True)
random.shuffle(speech_files)
num_speech_files = 65000
speech_files = speech_files[:num_speech_files]
music_files = sorted(glob.glob('/home/melan/supervised-vs-SSL/data/fma_data/1900MB_subset/*.mp3', recursive=True))

speech_output_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/noisy_passt_speech/'
music_output_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/noisy_passt_music/'
os.makedirs(speech_output_dir, exist_ok=True)
os.makedirs(music_output_dir, exist_ok=True)


# --- PARAMETERS ---
target_sr = 32000
duration_sec = 0.95
n_fft = 1024
win_length = 1024
hop_length = 320
n_mels = 128
fmin = 0
fmax = 16000
SNR = 12
target_len = int(duration_sec * target_sr)


def overlay_audio_numpy(y_main, y_noise, SNR=SNR, target_rms=0.1):
    # Normalize both to target RMS
    y_main = normalize_rms(y_main, target_rms)
    y_noise = normalize_rms(y_noise, target_rms)
    # Adjust noise for SNR
    snr_linear = 10 ** (SNR / 20)
    y_noise_scaled = y_noise / snr_linear
    # Overlay
    y_overlay = y_main + y_noise_scaled
    # Clip to [-1, 1] to avoid overflow
    y_overlay = np.clip(y_overlay, -1.0, 1.0)
    return y_overlay


def extract_audio_excerpts_numpy(audio_path, clip_length_sec=duration_sec, target_sr=target_sr):
    try:
        y, sr = librosa.load(audio_path, sr=target_sr)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}", flush=True)
        return []
    clip_len = int(clip_length_sec * target_sr)
    total_len = len(y)
    clips = []
    for start in range(0, total_len - clip_len + 1, clip_len):
        end = start + clip_len
        clips.append(y[start:end])
    return clips


def normalize_rms(y, target_rms=0.1):
    rms = np.sqrt(np.mean(y**2))
    if rms == 0:
        return y
    return y * (target_rms / rms)


def compute_byola_log_mel(
    y, target_sr=target_sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
    n_mels=n_mels, fmin=fmin, fmax=fmax, visualize=False, save_vis_path=None
):
    import nnAudio.features
    import torch
    # Ensure input is mono
    if y.ndim > 1:
        y = y[0]
    # Convert to torch tensor
    y_tensor = torch.tensor(y, dtype=torch.float32)
    # Compute mel spectrogram
    to_melspec = nnAudio.features.MelSpectrogram(
        sr=target_sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, center=True, power=2, verbose=False
    )
    mel = to_melspec(y_tensor)
    log_mel = (mel + torch.finfo(torch.float32).eps).log()
    # Normalize using provided stats
    stats = [-9.660292, 4.7219563]
    mean, std = stats
    log_mel = (log_mel - mean) / std

    # Convert to numpy 2D array
    log_mel = log_mel.cpu().numpy()
    if log_mel.ndim == 3:
        log_mel = log_mel.squeeze()  # Remove batch/channel if present

    # Optional visualization
    if visualize:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=fmax)
        plt.title('Normalized Log Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        if save_vis_path:
            plt.savefig(save_vis_path)
            print(f"    Saved visualization: {save_vis_path}")
        plt.close()

    return log_mel


def compute_passt_log_mel(y, target_sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                           n_mels=n_mels, fmin=fmin, fmax=fmax, visualize=False, save_vis_path=None):
    # Compute log mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    log_S = librosa.power_to_db(S, ref=np.max)

    # Normalize
    mean = np.mean(log_S)
    std = np.std(log_S)
    log_S_norm = (log_S - mean) / std

    # Optional visualization
    if visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_S_norm, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=fmax)
        plt.title('Normalized Log Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        if save_vis_path:
            plt.savefig(save_vis_path)
            print(f"    Saved visualization: {save_vis_path}")
        plt.close()

    return log_S_norm

noise_idx = 0


for main_file in music_files:
    main_clips = extract_audio_excerpts_numpy(main_file, clip_length_sec=duration_sec, target_sr=target_sr)

    for clip_num, y_main in enumerate(main_clips):
        # Normalize main clip
        y_main_norm = normalize_rms(y_main, target_rms=0.1)

        # Pick noise file (loop if needed)
        noise_file = noise_files[noise_idx % len(noise_files)]
        noise_idx += 1

        # Load and fix length of noise
        y_noise, sr_noise = librosa.load(noise_file, sr=target_sr)
        y_noise_fixed = y_noise[:target_len] if len(y_noise) >= target_len else np.pad(y_noise, (0, target_len - len(y_noise)), mode='constant')

        # Normalize noise
        y_noise_norm = normalize_rms(y_noise_fixed, target_rms=0.1)

        # Overlay and save
        y_overlay = overlay_audio_numpy(y_main_norm, y_noise_norm, SNR=SNR, target_rms=0.1)
        out_name = f"{os.path.splitext(os.path.basename(main_file))[0]}_{clip_num:04d}_{os.path.splitext(os.path.basename(noise_file))[0]}_{SNR}.wav"
        out_path = os.path.join(music_output_dir, out_name)

        export_wav = False  # Set to False to skip saving .wav files
        export_png = False # Set to False to skip saving .png visualizations

        if export_wav:
            sf.write(out_path, y_overlay, target_sr)

        # Compute and optionally save mel spectrogram
        mel_spec = compute_passt_log_mel( # change depending on the model
            y_overlay, target_sr,
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax, visualize=export_png,
            save_vis_path=os.path.join(music_output_dir, out_name + ".png") if export_png else None
        )
        npy_path = os.path.join(music_output_dir, out_name + ".npy")
        np.save(npy_path, mel_spec)
        print(f"    Saved mel spectrogram: {npy_path}")
        print(f"    Mel spectrogram shape: {mel_spec.shape}")
        print(f"    Mel spectrogram shape: {mel_spec.shape}")

        npy_files = [f for f in os.listdir(music_output_dir) if f.endswith('.npy')]
        print(f"Speech .npy files: {len(npy_files)}")

 


