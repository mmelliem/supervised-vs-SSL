# extract milliseconds is from lynette

def extract_milliseconds(audio, sample_rate):
    """
    Extract the duration in milliseconds from an audio signal.
    Parameters:
        audio (np.ndarray): Audio signal
        sample_rate (int): Sample rate of the audio in Hz
    Returns:
        float: Duration in milliseconds
    """

    # Extract the length of audio input
    duration_seconds = float(len(audio)) / sample_rate

    # Convert to milliseconds
    duration_ms = duration_seconds * 1000

    # Return the duration in milliseconds
    return duration_ms