# %% Import necessary libraries
import os, yaml

from audio_rep_networks.utils import download_data
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create dictionary of noise URLs and paths from YAML file
with open(os.path.join(os.getcwd(), 'noise_data.yaml'), 'r') as file:
    NOISE_DICT = yaml.safe_load(file)


if __name__ == "__main__":
    for noise_name, noise_urls in NOISE_DICT.items():
        # Define the path to save the noise data
        noise_path = os.path.join(os.environ['DATA_DIR'], noise_name)

        # Download the noise data if it does not exist
        if not os.path.exists(noise_path):
            # Loop through all URLs in list
            for url in noise_urls:
                try:
                    # Get noise data
                    download_data(url, noise_path)
                except Exception as e:
                    # Print error message if download fails
                    print(f'Error downloading {url} dataset: {e}')
        else:
            print(f'{noise_name} dataset already exists in {noise_path}.')
