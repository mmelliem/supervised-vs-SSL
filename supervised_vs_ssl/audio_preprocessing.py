# is there anything outside of audio-rep-network or conap that i'd need?
# this is all from the conap project, me and lynette wrote these, delete the ones that aren't relevant

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import pycochleagram.cochleagram as cgram
import pycochleagram.erbfilter as erb


def plot_waveform(
        input,
        export_path=None,
        export_format=None,
        sr=None,
        trim_silence=True
):
    """
    Plot the waveform of a WAV audio file.
    Parameters:
        input (str): Path to the input audio file
        export_path (str): Path to save the exported image file
        export_format (str): Format to export the image file
            ('png', 'jpg', 'jpeg', 'pdf', 'svg')
        sr (int): Sample rate, defaults to the sample rate of the audio file
        trim_silence (bool): Whether to trim silence from the audio
    Returns:
            None
    """
    # Ensure sr is defined
    sr = None

    # Check if input is a path or a loaded audio array
    if isinstance(input, str):  # If the input is a file path
        audio, sr = librosa.load(input, sr=sr)
    else:  # If the input is an already loaded audio array
        audio, sr = input, sr

    # Trim silence from the beginning and end of the audio
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=30)

    duration = len(audio) / sr

    # Plot waveform
    plt.plot(np.linspace(0, duration, len(audio)), audio)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (Hz)')
    plt.xlim(0, duration)
    plt.show()

    # If export_path is provided, save the audio file as wav or mp3
    if export_path and export_format:
        if export_format.lower() not in ['wav', 'mp3']:
            raise ValueError(
                f"Unsupported export format: '{export_format}'. "
                "Only 'wav' and 'mp3' are supported."
            )
        # Export the audio file to export_path
        sf.write(export_path, audio, sr, format=export_format.lower())
    elif export_path or export_format:
        raise ValueError(
            "Both export_path and export_format must be provided for export."
        )

    # After plt.show() (or instead of it if you don't want to show)
    if export_path and export_format:
        if export_format.lower() not in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            raise ValueError(
                f"Unsupported export format: '{export_format}'. "
                "Supported formats: png, jpg, jpeg, pdf, svg."
            )
        # Export the plot to the specified path
        plt.savefig(export_path, format=export_format.lower())
    elif export_path or export_format:
        raise ValueError(
            "Both export_path and export_format must be provided for export."
        )


def compute_spectrogram(
        input,
        sr=None,
        n_fft=2048,
        hop_length=512,
        window='hann',
        max_Hz=None,
        trim_silence=False
):
    """
    Compute a spectrogram from a WAV audio file.
    Parameters:
        input (str): Path to the input audio file
        sr (int): Sample rate, defaults to the sample rate of the audio file
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        window (str): Window function type
        max_Hz (int): Maximum frequency to consider in the spectrogram
        trim_silence (bool): Whether to trim silence from the audio
    Returns:
        tuple: (audio, frequencies, times)
            - spectrogram (ndarray): Spectrogram of the audio
            - sr (int): Sample rate of the audio
            - freqs (ndarray): Frequencies corresponding to the spectrogram
            - times (ndarray): Time frames corresponding to the spectrogram
    """
    # Check if input is a path or a loaded audio array
    if isinstance(input, str):  # If the input is a file path
        audio, sr = librosa.load(input, sr=sr)
    else:  # If the input is an already loaded audio array
        audio, sr = input, sr

    # Trim silence from the beginning and end of the audio
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=30)

    # Compute the spectrogram
    spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft,
                                      hop_length=hop_length,
                                      window=window))

    # Get frequency and times
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.times_like(spectrogram, sr=sr, hop_length=hop_length)

    # Limit the frequency range if specified
    if max_Hz is not None:
        freq_mask = freqs <= max_Hz
        freqs = freqs[freq_mask]
        spectrogram = spectrogram[freq_mask, :]

    return spectrogram, sr, freqs, times


def plot_spectrogram(
        input,
        export_path=None,
        export_format=None,
        sr=None,
        max_Hz=None,
        trim_silence=False,
        log_scale=False
):
    """
    Plot the spectrogram of a WAV audio file.
    Parameters:
        input (str): Path to the input audio file
        export_path (str): Path to save the exported image file
        export_format (str): Format to export the image file
            ('png', 'jpg', 'jpeg', 'pdf', 'svg')
        max_Hz (int): Maximum frequency to consider in the spectrogram
        trim_silence (bool): Whether to trim silence from the audio
        log_scale (bool): Whether to use a logarithmic scale for the y-axis
    Returns:
        None
    """
    # Check if input is a path or a loaded audio array
    if isinstance(input, str):  # If the input is a file path
        audio, sr = librosa.load(input, sr=sr)
    # If the input is an already loaded audio array
    elif isinstance(input, np.ndarray):
        audio, sr = input, sr
    else:
        raise ValueError(
            "Input must be a file path or a loaded 2D audio array"
        )

    spectrogram, sr, freqs, times = compute_spectrogram(
        audio,
        max_Hz=max_Hz,
        trim_silence=trim_silence
    )

    # Convert to dB scale (more common for spectrograms)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Apply logarithmic scale if specified
    y_axis = 'log' if log_scale else 'hz'

    # Plotting the spectrogram
    librosa.display.specshow(
        spectrogram_db,
        x_axis='time',
        y_axis=y_axis,
        sr=sr
    )
    plt.title('Spectrogram')
    plt.xlim(0, times.max())
    if not log_scale:
        plt.ylim(0, freqs.max())
    plt.colorbar(label='dB')
    plt.show()

    # After plt.show() (or instead of it if you don't want to show)
    if export_path and export_format:
        if export_format.lower() not in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            raise ValueError(
                f"Unsupported export format: '{export_format}'. "
                "Supported formats: png, jpg, jpeg, pdf, svg."
            )
        # Export the plot to the specified path
        plt.savefig(export_path, format=export_format.lower())
    elif export_path or export_format:
        raise ValueError(
            "Both export_path and export_format must be provided for export."
        )


def compute_mel_spectrogram(
        input,
        sr=None,
        n_fft=2048,
        hop_length=512,
        window='hann',
        max_Hz=None,
        trim_silence=False,
        n_mels=128
):
    """
    Compute a mel spectrogram from a WAV audio file.
    Parameters:
        input (str): Path to the input audio file
        sr (int): Sample rate, defaults to the sample rate of the audio file
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        window (str): Window function type
        max_Hz (int): Maximum frequency to consider in the spectrogram
        trim_silence (bool): Whether to trim silence from the audio
        n_mels (int): Number of mel bands to generate
    Returns:
        tuple: (mel_spectrogram, sr, freqs, times)
            - mel_spectrogram (ndarray): Mel spectrogram of the audio
            - sr (int): Sample rate of the audio
            - freqs (ndarray): Frequencies corresponding to the mel spectrogram
            - times (ndarray): Time frames corresponding to the mel spectrogram
    """
    # Check if input is a path or a loaded audio array
    if isinstance(input, str):  # If the input is a file path
        audio, sr = librosa.load(input, sr=sr)
    else:  # If the input is an already loaded audio array
        audio, sr = input, sr

    # Compute the spectrogram
    spectrogram, sr, freqs, times = compute_spectrogram(
        audio, sr, n_fft, hop_length, window, max_Hz, trim_silence
    )

    # Compute the mel filter
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    # Apply mel filter to spectrogram array using dot product
    mel_spectrogram = np.dot(mel_filter, spectrogram)

    return mel_spectrogram, sr, freqs, times


def plot_mel_spectrogram(
        input,
        export_path=None,
        export_format=None,
        sr=None,
        max_Hz=None,
        trim_silence=False,
        log_scale=False,
        n_mels=128
):
    """
    Plot the mel spectrogram of a WAV audio file.
    Parameters:
        input (str): Path to the input audio file
        export_path (str): Path to save the exported image file
        export_format (str): Format to export the image file
            ('png', 'jpg', 'jpeg', 'pdf', 'svg')
        max_Hz (int): Maximum frequency to consider in the mel spectrogram
        trim_silence (bool): Whether to trim silence from the audio
        log_scale (bool): Whether to use a logarithmic scale for the y-axis
        n_mels (int): Number of mel bands to generate
    Returns:
        None
    """
    # Check if input is a path or a loaded audio array
    if isinstance(input, str):  # If the input is a file path
        audio, sr = librosa.load(input, sr=sr)
    else:  # If the input is an already loaded audio array
        audio, sr = input, sr

    # Compute the mel spectrogram
    mel_spectrogram, sr, freqs, times = compute_mel_spectrogram(
        audio, trim_silence=trim_silence, n_mels=n_mels)
    # no max_Hz arg due to mel filter calculation

    # Convert to dB scale
    mel_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

    # Apply logarithmic scale if specified
    y_axis = 'log' if log_scale else 'hz'

    # Plotting the mel spectrogram
    librosa.display.specshow(
        mel_db,
        x_axis='time',
        y_axis=y_axis,
        fmax=max_Hz if max_Hz else None
    )
    plt.title('Mel Spectrogram')
    plt.xlim(0, times.max())
    if not log_scale:
        plt.ylim(0, freqs.max())
    plt.colorbar(label='dB')
    plt.show()

    # After plt.show() (or instead of it if you don't want to show)
    if export_path and export_format:
        if export_format.lower() not in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            raise ValueError(
                f"Unsupported export format: '{export_format}'. "
                "Supported formats: png, jpg, jpeg, pdf, svg."
            )
        # Export the plot to the specified path
        plt.savefig(export_path, format=export_format.lower())
    elif export_path or export_format:
        raise ValueError(
            "Both export_path and export_format must be provided for export."
        )


def compute_mfcc(
        input,
        sr=None,
        n_fft=2048,
        hop_length=512,
        window='hann',
        max_Hz=None,
        trim_silence=False,
        n_mfcc=13
):
    """
    Compute MFCC features from a WAV audio file.
    Parameters:
        input (str): Path to the input audio file
        sr (int): Sample rate, defaults to the sample rate of the audio file
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        window (str): Window function type
        max_Hz (int): Maximum frequency to consider in the spectrogram
        trim_silence (bool): Whether to trim silence from the audio
        n_mfcc (int): Number of MFCC features to compute
    Returns:
        tuple: (mfcc, sr, freqs, times)
            - mfcc (ndarray): MFCC features of the audio
            - sr (int): Sample rate of the audio
            - freqs (ndarray): Frequencies corresponding to the MFCC features
            - times (ndarray): Time frames corresponding to the MFCC features
    """
    # Check if input is a path or a loaded audio array
    if isinstance(input, str):  # If the input is a file path
        audio, sr = librosa.load(input, sr=sr)
    else:  # If the input is an already loaded audio array
        audio, sr = input, sr

    # Compute the mel spectrogram
    mel_spectrogram, sr, freqs, times = compute_mel_spectrogram(
        audio, sr, n_fft, hop_length, window, max_Hz, trim_silence
    )

    # Apply MFCC transformation to mel spectrogram
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_spectrogram), n_mfcc=n_mfcc
        )

    return mfcc, sr, freqs, times


def plot_mfcc(
        input,
        export_path=None,
        export_format=None,
        sr=None,
        max_Hz=None,
        trim_silence=False,
        log_scale=False,
        n_mfcc=13
):
    """
    Plot the MFCC features of a WAV audio file.
    Parameters:
        input (str): Path to the input audio file
        export_path (str): Path to save the exported image file
        export_format (str): Format to export the image file
            ('png', 'jpg', 'jpeg', 'pdf', 'svg')
        max_Hz (int): Maximum frequency to consider in the MFCC features
        trim_silence (bool): Whether to trim silence from the audio
        log_scale (bool): Whether to use a logarithmic scale for the y-axis
        n_mfcc (int): Number of MFCC features to compute
    Returns:
        None
    """
    # Check if input is a path or a loaded audio array
    if isinstance(input, str):  # If the input is a file path
        audio, sr = librosa.load(input, sr=sr)
    else:  # If the input is an already loaded audio array
        audio, sr = input, sr

    # Compute MFCC features
    mfcc, sr, freqs, times = compute_mfcc(
        audio, sr=sr, max_Hz=max_Hz, trim_silence=trim_silence, n_mfcc=n_mfcc
    )

    # Convert to dB scale
    mfcc_db = librosa.amplitude_to_db(mfcc, ref=np.max)

    # Apply logarithmic scale if specified
    y_axis = 'log' if log_scale else 'hz'

    # Plotting the MFCC features
    librosa.display.specshow(
        mfcc_db,
        x_axis='time',
        y_axis=y_axis,
        fmax=max_Hz if max_Hz else None
    )
    plt.title('MFCCs')
    plt.xlim(0, times.max())
    if not log_scale:
        plt.ylim(0, freqs.max())
    plt.colorbar(label='dB')
    plt.show()

    # After plt.show() (or instead of it if you don't want to show)
    if export_path and export_format:
        if export_format.lower() not in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            raise ValueError(
                f"Unsupported export format: '{export_format}'. "
                "Supported formats: png, jpg, jpeg, pdf, svg."
            )
        # Export the plot to the specified path
        plt.savefig(export_path, format=export_format.lower())
    elif export_path or export_format:
        raise ValueError(
            "Both export_path and export_format must be provided for export."
        )


def compute_cochleagram(
        input,
        sr=None,
        n_filters=64,
        low_lim=50,
        hi_lim=8000,
        trim_silence=False
):
    # Load audio
    if isinstance(input, str):
        audio, sr = librosa.load(input, sr=sr)
    else:
        audio, sr = input, sr

    # Trim silence from the beginning and end of the audio
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=30)

    # Get filter bank
    fcoefs = erb.make_erb_cos_filters(
        audio, sr, n=n_filters, low_lim=low_lim, hi_lim=hi_lim
    )

    # Compute cochleagram
    cochleagram = cgram.cochleagram(audio, sr, fcoefs)

    return cochleagram


def plot_cochleagram(
        input,
        export_path=None,
        export_format=None,
        sr=None,
        n_filters=None,
        min_cf=20,
        max_cf=None,
        trim_silence=False,
        log_scale=False
):
    """
    Plot the cochleagram of a WAV audio file.
    Parameters:
        input (str): Path to the input audio file
        export_path (str): Path to save the exported image file
        export_format (str): Format to export the image file
            ('png', 'jpg', 'jpeg', 'pdf', 'svg')
        sr (int): Sample rate, defaults to the sample rate of the audio file
        n_filters (int): Number of filters to use in the filterbank
        min_cf (float): Minimum center frequency in Hz
        max_cf (float): Maximum center frequency in Hz
        trim_silence (bool): Whether to trim silence from the audio
        log_scale (bool): Whether to use a logarithmic scale for the y-axis
    Returns:
        None
    """
    # Check if input is a path or a loaded audio array
    if isinstance(input, str):  # If the input is a file path
        audio, sr = librosa.load(input, sr=sr)
    else:  # If the input is an already loaded audio array
        audio, sr = input, sr

    # Compute the cochleagram
    cochleagram, sr, cfs, times = compute_cochleagram(
        audio,
        sr=sr,
        n_filters=n_filters,
        min_cf=min_cf,
        max_cf=max_cf,
        trim_silence=trim_silence
    )

    # Convert to dB scale
    cochleagram_db = librosa.amplitude_to_db(
        np.abs(cochleagram), ref=np.max
    )

    # Apply logarithmic scale if specified
    y_axis = 'log' if log_scale else 'hz'

    # Plot the cochleagram
    librosa.display.specshow(
        cochleagram_db,
        x_coords=times,
        y_coords=cfs,
        x_axis='time',
        y_axis=y_axis,
        cmap='viridis'
    )

    plt.title('Cochleagram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.xlim(0, times.max())

    # Add colorbar
    plt.colorbar(label='dB')
    plt.show()

    # After plt.show() (or instead of it if you don't want to show)
    if export_path and export_format:
        if export_format.lower() not in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            raise ValueError(
                f"Unsupported export format: '{export_format}'. "
                "Supported formats: png, jpg, jpeg, pdf, svg."
            )
        # Export the plot to the specified path
        plt.savefig(export_path, format=export_format.lower())
    elif export_path or export_format:
        raise ValueError(
            "Both export_path and export_format must be provided for export."
        )