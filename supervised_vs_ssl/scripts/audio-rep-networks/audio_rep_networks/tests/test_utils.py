import pytest
import numpy as np
from audio_rep_networks.utils import generate_dB_distribution, download_data
from pathlib import Path
import os


def test_generate_dB_distribution():
    """
    Test the generate_dB_distribution function.
    """
    # Generate the distribution
    dB_dist = generate_dB_distribution()

    # Check the distribution
    assert len(dB_dist) == 1000
    assert np.round(dB_dist.mean()) == 12
    assert np.round(dB_dist.std()) == 2
    assert isinstance(dB_dist, np.ndarray)


@pytest.mark.parametrize("mean,sd,n", [(4, 2, 100), (6, 3, 200), (3, 3, 1000)])
def test_generate_dB_distribution_matrix(mean, sd, n):
    """
    Test the generate_dB_distribution function.
    """
    # Generate the distribution
    dB_dist = generate_dB_distribution(mean, sd, n)

    # Check the distribution
    assert len(dB_dist) == n
    assert np.round(dB_dist.mean()) == mean
    assert np.round(dB_dist.std()) == sd
    assert isinstance(dB_dist, np.ndarray)


def test_download_with_demo_files():
    """
    Test downloading and processing .wav and .npy files from
    a GitHub repository.
    """
    url_wav = (
        "https://github.com/mcdermottLab/kelletal2018/blob/master/demo_stim/"
        "example_1.wav"
    )
    url_npy = (
        "https://github.com/mcdermottLab/kelletal2018/blob/master/demo_stim/"
        "example_cochleagram_0.npy"
    )

    # Get the current directory
    current_dir = Path(os.getcwd())

    # Download the files
    for url in [url_wav, url_npy]:
        # Download the actual file from the provided URL
        download_data(url, current_dir)

        # Assert that the file is downloaded correctly
        expected_file = (
            current_dir / 'audio_rep_networks_dataset' / Path(url).name
        )
        assert expected_file.exists()

        # Check that the file is not empty
        if expected_file.is_file():
            with open(expected_file, 'rb') as f:
                assert len(f.read()) > 0
        else:
            raise AssertionError(
                f"Expected a file but found a directory: {expected_file}"
            )

        # Clean up after the test
        if expected_file.exists() and expected_file.is_file():
            expected_file.unlink()
