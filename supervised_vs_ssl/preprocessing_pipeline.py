import os
import numpy as np
import librosa
import glob
import soundfile as sf
import matplotlib.pyplot as plt

# '/home/melan/supervised-vs-SSL/data/fma_data/fma_small/**/*.mp3' '/home/melan/supervised-vs-SSL/data/speech_data/*.mp3'
main_files = sorted(glob.glob('/home/melan/supervised-vs-SSL/data/fma_data/fma_small/**/*.mp3', recursive=True))
noise_files = sorted(glob.glob('/home/melan/supervised-vs-SSL/data/noise_data/scenes_stereo/*.wav', recursive=True))
output_dir = '/home/melan/supervised-vs-SSL/data/preprocessed/noisy_music/'
os.makedirs(output_dir, exist_ok=True)

# --- PARAMETERS ---
target_sr = 16000
duration_sec = 6
n_fft = 1024
win_length = 1024
hop_length = 320
n_mels = 128
f_min = 0
f_max = 8000
SNR = 12
target_len = duration_sec * target_sr


def overlay_audio_numpy(y_main, y_noise, SNR=12, target_rms=0.1):
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


def extract_audio_excerpts_numpy(audio_path, clip_length_sec=6, target_sr=16000):
    y, sr = librosa.load(audio_path, sr=target_sr)
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


def compute_log_mel_spectrogram(audio_path, noise_path=None, visualize=False):
    # Load and resample audio
    y, sr = librosa.load(audio_path, sr=target_sr)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, max(0, target_len - len(y))), mode='constant')

    # Overlay environmental noise if provided
    if noise_path:
        y_noisy = overlay_audio_numpy(y, noise_path, SNR=SNR)
    else:
        y_noisy = y

    # Compute log mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y_noisy,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max
    )
    log_S = librosa.power_to_db(S, ref=np.max)

    # Normalize
    mean = np.mean(log_S)
    std = np.std(log_S)
    log_S_norm = (log_S - mean) / std

    # Optional visualization
    if visualize:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_S_norm, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=fmax)
        plt.title('Normalized Log Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

    return log_S_norm

print(f"Found {len(main_files)} main files")
print(f"Found {len(noise_files)} noise files")
if len(main_files) == 0:
    print("No main files found! Check your glob pattern.")
if len(noise_files) == 0:
    print("No noise files found! Check your glob pattern.")

noise_idx = 0

for main_file in main_files:
    print(f"\nProcessing main file: {main_file}")
    main_clips = extract_audio_excerpts_numpy(main_file, clip_length_sec=6, target_sr=target_sr)
    print(f"Extracted {len(main_clips)} clips from {main_file}")

    for clip_num, y_main in enumerate(main_clips):
        print(f"  Clip {clip_num+1}/{len(main_clips)}: shape {y_main.shape}")

        # Normalize main clip
        y_main_norm = normalize_rms(y_main, target_rms=0.1)
        print(f"    Normalized main clip RMS: {np.sqrt(np.mean(y_main_norm**2)):.4f}")

        # Pick noise file (loop if needed)
        noise_file = noise_files[noise_idx % len(noise_files)]
        print(f"    Using noise file: {noise_file}")
        noise_idx += 1

        # Load and fix length of noise
        y_noise, sr_noise = librosa.load(noise_file, sr=target_sr)
        y_noise_fixed = y_noise[:target_len] if len(y_noise) >= target_len else np.pad(y_noise, (0, target_len - len(y_noise)), mode='constant')
        print(f"    Loaded noise, shape: {y_noise_fixed.shape}, sr: {sr_noise}")

        # Normalize noise
        y_noise_norm = normalize_rms(y_noise_fixed, target_rms=0.1)
        print(f"    Normalized noise RMS: {np.sqrt(np.mean(y_noise_norm**2)):.4f}")

        # Overlay and save
        y_overlay = overlay_audio_numpy(y_main_norm, y_noise_norm, SNR=SNR, target_rms=0.1)
        out_name = f"{os.path.splitext(os.path.basename(main_file))[0]}_{clip_num:04d}_{os.path.splitext(os.path.basename(noise_file))[0]}_{SNR}.wav"
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, y_overlay, target_sr)
        print(f"    Saved: {out_path}")