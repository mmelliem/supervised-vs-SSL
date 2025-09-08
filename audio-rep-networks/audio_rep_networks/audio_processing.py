import os
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf

from pydub import AudioSegment, silence
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

def load_wav(input_file, return_file=True):
    """
    Converts an audio file to a .wav file, if needed. Stores the output wav
    file with the same name and directory as the input file.

    Parameters
    ----------
    input_file : str
        Path to the input audio file.
    output_file : str
        Path to store the output .wav file.
    """
    # Store current dir and change working directory to the data directory
    old_dir = os.getcwd()
    os.chdir(os.environ['DATA_DIR'])

    # Get absolute path of the input file and change working directory back
    input_file = os.path.abspath(input_file)
    os.chdir(old_dir)

    # Get the file extension of the input file
    file_path, file_type = input_file.split('.')[0], \
        input_file.split('.')[-1]
    # Check if input file already exists as a .wav file
    if file_type == 'wav':
        print(f"{input_file} is already in .wav format.")
        return AudioSegment.from_file(input_file, format="wav")
    elif os.path.exists(f"{file_path}.wav"):
        print(f"{file_path}.wav already exists.")
        return AudioSegment.from_file(f"{file_path}.wav", format="wav")
    else:
        # Load the audio file
        file_nonwav = AudioSegment.from_file(input_file, format=file_type)
        # Convert the input file to a .wav file
        file_wav = file_nonwav.export(
            Path(f"{file_path}.wav"),
            format="wav"
        )

    if return_file:
        return file_wav


def overlay_audio(audio1, audio2, SNR=None, overlay_dir=None):
    """
    Overlay audio2 over audio1 at a specified SNR,
    that is dB, either same, quieter or louder.
    Both audio files must be in the same format.

    Parameters
    ----------
    audio1 : str
        Path to the first audio file.
    audio2 : str
        Path to the second audio file. The file that will be overlayed.
    SNR : int
        Signal-to-Noise Ratio in dB. Default is None.
        Positive integers will make audio2 louder, negative integers will
        make it quieter than audio1.

    Returns
    -------
    file_handle : AudioSegment
        AudioSegment object of the overlayed audio.
    """

    # Load the audio files
    sound1 = load_wav(audio1, return_file=True)
    sound2 = load_wav(audio2, return_file=True)

    # Check if SNR is specified
    if SNR is not None:

        # Check if SNR is an integer
        if not isinstance(SNR, int):
            raise ValueError("SNR must be an integer.")
        else:
            # Make audio2 louder or quieter
            sound2 = sound2 + 6
            # set SNR
            SNR_overlay = SNR
    else:
        # Default SNR is 0
        SNR_overlay = 0

    # Overlay sound2 over sound1 at the specified position
    overlay = sound1.overlay(sound2, position=0)

    # Save the overlayed audio
    if overlay_dir is not None:

        # Set the names of the audio file
        audio1_name = audio1.split('/')[-1].split('.')[0]
        audio2_name = audio2.split('/')[-1].split('.')[0]
        overlay_name = (f"{audio1_name}_{audio2_name}_"
                        f"{SNR_overlay}.wav")

        # Export the overlayed audio
        overlay.export(overlay_dir + overlay_name, format="wav")

    return overlay


def extract_audio_excerpts(audio_file, clip_length_sec, num_clips,
                           min_audio_percent=90, seed=42, output_path=None,
                           file_extension=None):
    """
    Extract a specified number of random clips from an audio file,
    ensuring that each clip contains at least min_audio_percent
    non-silent audio.

    Parameters:
    - audio_file: str, path to the audio file
    - clip_length_sec: float, length of each clip in seconds
    - num_clips: int, number of clips to extract
    - min_audio_percent: float, minimum percentage of non-silent audio
      in a clip
    - seed: int, optional random seed for reproducibility
    - output_path: str, optional directory to save the audio excerpts
    - file_extension: str, optional file extension for the output files

    Returns:
    - list of file paths of the extracted clips
    """

    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Load the audio file
    sound = AudioSegment.from_file(audio_file)

    # Calculate the duration of the audio file and the clip length
    duration_ms = len(sound)
    clip_length_ms = int(clip_length_sec * 1000)

    # Check if the requested number of clips is possible
    if clip_length_ms > duration_ms:
        raise ValueError("Clip length is longer than the audio duration.")

    # Prepare output directory
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "audio_excerpts")
    else:
        output_path = os.path.join(output_path, "audio_excerpts")
    os.makedirs(output_path, exist_ok=True)

    # Prepare output file names
    base_name, input_extension = os.path.splitext(os.path.basename(audio_file))
    if file_extension is None:
        file_extension = input_extension.lstrip('.')

    # Extract the clips randomly
    clips = set()
    attempts = 0
    max_attempts = num_clips * 10  # Avoid infinite loops

    # Extract clips until the requested number is reached
    while len(clips) < num_clips and attempts < max_attempts:

        # Randomly select the start and end times of the clip
        start_ms = random.randint(0, duration_ms - clip_length_ms)
        end_ms = start_ms + clip_length_ms

        # Check if the clip has not been extracted before
        if (start_ms, end_ms) not in clips:
            clip = sound[start_ms:end_ms]

            # Adjust min_silence_len to be at least 10% of the clip length
            min_silence_len = max(
                int(0.1 * clip_length_ms),
                int((100 - min_audio_percent) / 100 * clip_length_ms)
            )

            # Adjust silence_thresh to be 50% quieter than the average loudness
            silence_thresh = clip.dBFS - 6

            # Detect silence in the clip
            silent_ranges = silence.detect_silence(
                clip, min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )

            # Calculate the percentage of silent audio in the clip
            silent_duration = sum(end - start for start, end in silent_ranges)
            silent_percent = (silent_duration / clip_length_ms) * 100

            # Check if the clip contains enough non-silent audio
            if (100 - silent_percent) >= min_audio_percent:
                # Generate file name with start and stop times
                file_name = f"{base_name}_{start_ms}_{end_ms}.{file_extension}"
                output_file = os.path.join(output_path, file_name)

                # Export the clip
                clip.export(output_file, format=file_extension)
                clips.add((start_ms, end_ms))

        attempts += 1

    # Return the file paths of the extracted clips
    return [
        os.path.join(
            output_path, f"{base_name}_{start}_{stop}.{file_extension}"
        )
        for start, stop in clips
    ]


def generate_spectrum_matched_noise(input_file, seed=42, output_path=None):
    """
    Generate music-shaped noise by matching the noise spectrum to
    the average spectrum of the music.

    Parameters:
    - input_file: str, path to the audio file
    - seed: int, optional random seed for reproducibility
    - output_path: str, optional directory to save the noise file

    Returns:
    - str, path to the generated noise file
    """

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Load the input sound
    y, sr = librosa.load(input_file, sr=None, mono=True)

    # Compute the FFT of the input sound
    spectrum = np.fft.rfft(y)

    # Extract magnitude spectrum
    magnitude = np.abs(spectrum)

    # Generate random phase
    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(len(magnitude)))

    # Construct noise spectrum with the magnitude of the input and random phase
    noise_spectrum = magnitude * random_phase

    # Inverse FFT to get the time-domain noise
    noise = np.fft.irfft(noise_spectrum)

    # Normalize to avoid clipping
    noise = noise / np.max(np.abs(noise))

    # Prepare output directory
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "input_shaped_noise")
    os.makedirs(output_path, exist_ok=True)

    # Generate file name and save the noise file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    noise_file = os.path.join(output_path,
                              f"{base_name}_music_shaped_noise.wav")
    sf.write(noise_file, noise, sr)

    return noise_file
