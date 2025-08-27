import pytest
from pydub import AudioSegment
from audio_rep_networks.audio_processing import (
    overlay_audio,
    extract_audio_excerpts,
    generate_spectrum_matched_noise
)
import os
import requests
from tqdm import tqdm
import numpy as np
from scipy.signal import welch
import librosa
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def audio_files(tmp_path_factory):
    """
    Fixture to download and provide the audio files.
    """
    url_wav = (
        "https://github.com/mcdermottLab/kelletal2018/raw/master/demo_stim/"
        "example_1.wav"
    )
    url_overlay = (
        "https://github.com/mcdermottLab/kelletal2018/raw/master/demo_stim/"
        "example_2.wav"
    )

    # Get the base directory & set file paths
    base_dir = tmp_path_factory.getbasetemp()
    # Set DATA_DIR in environment to the base directory
    os.environ["DATA_DIR"] = str(base_dir)
    # Set the audio file paths
    file1 = base_dir / "audio1.wav"
    file2 = base_dir / "audio2.wav"

    for url, file in [(url_wav, file1), (url_overlay, file2)]:
        if not file.exists():
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(file, 'wb') as f, tqdm(
                desc=f"Downloading {file.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

    return file1, file2


# Test overlay_audio without SNR and overlay directory
def test_overlay_audio_without_snr_and_overlay_dir(tmp_path, audio_files):
    """
    Test overlay_audio without specifying SNR and overlay directory.
    """
    file1, file2 = audio_files

    # Call the function without SNR and overlay_dir
    result = overlay_audio(str(file1), str(file2))

    # Check that the result is an AudioSegment
    assert isinstance(result, AudioSegment)
    assert len(result) > 0


# Test overlay_audio with SNR and overlay directory
@pytest.mark.parametrize("snr", [6, -2, 2, 10])
def test_overlay_audio_with_snr_and_overlay_dir(tmp_path, audio_files, snr):
    """
    Test overlay_audio with specified SNR and overlay directory using pytest
    parametrize.
    """

    # Get the audio files
    file1, file2 = audio_files

    # Create directory for overlays
    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()

    # Call the function with each SNR and overlay_dir
    result = overlay_audio(
        str(file1), str(file2), SNR=snr, overlay_dir=str(overlay_dir) + "/"
    )

    # Check that the result is an AudioSegment
    assert isinstance(result, AudioSegment)
    assert len(result) > 0

    # Check that the file was created
    expected_file = overlay_dir / f"audio1_audio2_{snr}.wav"
    assert expected_file.exists()


@pytest.fixture
def silent_audio_file(tmp_path):
    """Generate a silent audio file for testing."""
    silent_audio = AudioSegment.silent(duration=5000)  # 5 seconds of silence
    silent_file = tmp_path / "silent_audio.wav"
    silent_audio.export(silent_file, format="wav")
    return str(silent_file)


@pytest.mark.parametrize(
    "clip_length_sec, num_clips, seed, output_path, file_extension, "
    "min_audio_percent, audio_source",
    [
        (0.2, 4, 42, None, None, 50, "regular"),
        (0.8, 10, 42, None, None, 60, "regular"),
        (0.2, 4, 42, "custom_directory_4", "mp3", 70, "regular"),
        (0.2, 10, 42, "custom_directory_10", "mp3", 80, "regular"),
        (1.0, 3, 42, None, "wav", 50, "silent")  # Test case for silent audio
    ]
)
# Test extract_audio_excerpts
def test_extract_audio_excerpts(
    clip_length_sec, num_clips, seed, output_path, file_extension,
    min_audio_percent, audio_source, audio_files, silent_audio_file, tmp_path
):
    """
    Test the extract_audio_excerpts function with various
    inputs, including silent audio.
    """

    # Select the appropriate audio file (either normal or silent)
    if audio_source == "silent":
        audio_file = silent_audio_file
    else:
        audio_file, _ = audio_files

    # Set output directory
    output_dir = tmp_path / "audio_excerpts"
    output_dir.mkdir()

    # Run the function
    extracted_clips = extract_audio_excerpts(
        audio_file, clip_length_sec, num_clips, min_audio_percent, seed,
        output_path=str(output_dir),
        file_extension=file_extension
    )

    # Check the input
    if audio_source == "silent":
        # If the audio is silent, no valid clips should be extracted
        assert len(extracted_clips) == 0
    else:
        # Regular checks for normal audio
        assert len(extracted_clips) == num_clips
        assert all(isinstance(clip, str) for clip in extracted_clips)

        # Check file names
        expected_extension = file_extension if file_extension else "wav"
        for clip in extracted_clips:
            assert clip.endswith(expected_extension)

        # Check if files exist
        for clip in extracted_clips:
            assert os.path.exists(clip)

        # Ensure randomness does not produce overlapping clips
        starts = [
            int(os.path.basename(c).split('_')[-2]) for c in extracted_clips
        ]
        assert len(starts) == len(set(starts))


# Test invalid clip length
def test_extract_audio_excerpts_invalid_clip_length(audio_files):
    """
    Test that the function raises a ValueError when the clip
    length is longer than the audio duration.
    """

    # Get the audio files
    audio_file1, _ = audio_files

    # Call the function with an invalid clip length
    with pytest.raises(
        ValueError, match="Clip length is longer than the audio duration"
    ):
        extract_audio_excerpts(audio_file1, 10000, 1)


# Test generate_spectrum_matched_noise
def test_music_shaped_noise_spectrum(audio_files):
    """
    Test that the generated noise has a similar average
    spectrum to the original audio.
    """

    # Get the audio files
    audio_file1, _ = audio_files
    noise_file = generate_spectrum_matched_noise(audio_file1)

    # Load original and noise audio
    original_samples, original_sr = librosa.load(audio_file1, sr=None,
                                                 mono=True)
    noise_samples, noise_sr = librosa.load(noise_file, sr=None,
                                           mono=True)

    # Ensure sample rate consistency
    assert original_sr == noise_sr

    # Compute PSDs
    freqs_orig, psd_orig = welch(original_samples, fs=original_sr)
    freqs_noise, psd_noise = welch(noise_samples, fs=noise_sr)

    # Compare PSDs using correlation
    correlation = np.corrcoef(psd_orig, psd_noise)[0, 1]

    # Assert that the correlation is high (e.g., > 0.9)
    assert correlation > 0.9
