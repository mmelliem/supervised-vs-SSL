import os
import shutil
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm
import numpy as np
import torch


def generate_dB_distribution(mean=12, sd=2, n=1000):
    """
    This function generates a distribution of dB values.
    The defaults are set based on the Kell et al. (2018)
    paper: a Gaussian with a mean of 12 and a standard
    deviation of 2. The sample size is set to 1000.

    Parameters
    ----------
    mean : int
        An integer indicating the desired mean of the
        Gaussian.
    sd : int
        An integer indicating the desired standard
        deviation of the Gaussian.

    Returns
    -------
    np.array entailing the generated Gaussian.

    Examples
    --------
    Generate a Gaussian using the defaults.

    >>> dB_dist = generate_dB_distribution()
    """

    # Generate the distribution based on the parameters
    np.random.seed(42)
    dB_dist = np.random.normal(mean, sd, n)

    return dB_dist


def download_data(url, data_path=None):
    """
    Download data(set) from a specified link. If the data
    is in a zip file, it will be unzipped.

    Parameters
    ----------
    url : string
        The URL of the data to download.
    data_path : string
        Path where the file will be saved. If None, the file will be saved
        in the current working directory. Default = None.

    Returns
    -------
    file_local : PosixPath
        PosixPath indicating the path to the downloaded data(set).
    """

    # Determine the directory to save the file
    if data_path is None:
        path = Path(os.curdir) / 'audio_rep_networks_dataset'
    else:
        path = Path(data_path) / 'audio_rep_networks_dataset'

    # Ensure the directory exists
    if not path.exists():
        os.makedirs(path)

    # Provide a name for the downloaded file
    file_name = Path(url).name
    file_path = path / file_name

    # Download the file if it doesn't already exist
    if not file_path.exists():
        print(f"Downloading data to {file_path}")

        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            file_size = int(response.headers.get('Content-Length', 0))

            # Download with progress bar
            with open(file_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

        # If the file is a zip file, unzip it
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path)
            os.remove(file_path)

    # Remove __MACOSX folder if it exists
    macosx_path = path / '__MACOSX'
    if macosx_path.exists():
        shutil.rmtree(macosx_path, ignore_errors=True)

    return file_path if file_path.exists() else path

def get_device():
    """
    Get the device (mps, cuda, or cpu) to run the model on.

    Returns:
    --------
    device : str
        Device to run the model on.
    num_gpus : int
        Number of GPUs available.
    """
    if torch.backends.mps.is_available():
        return 'mps', 1
    elif torch.cuda.is_available():
        return 'cuda', torch.cuda.device_count()
    else:
        return 'cpu', 0