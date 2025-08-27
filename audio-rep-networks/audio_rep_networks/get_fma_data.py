# Import necessary libraries
import os

from audio_rep_networks.utils import download_data
from dotenv import load_dotenv
from argparse import ArgumentParser

# Load environment variables
load_dotenv()

# Specify URLs for FMA datasets

# Create dictionary of noise URLs and paths
FMA_DICT = {
    'fma_metadata': 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip',
    'fma_small': 'https://os.unil.cloud.switch.ch/fma/fma_small.zip',
    'fma_medium': 'https://os.unil.cloud.switch.ch/fma/fma_medium.zip',
    'fma_large': 'https://os.unil.cloud.switch.ch/fma/fma_large.zip',
    'fma_full': 'https://os.unil.cloud.switch.ch/fma/fma_full.zip'
}

if __name__ == "__main__":
    
    # Create an argument parser
    parser = ArgumentParser(description='Download FMA data')
    # Add an argument to specify the size of the dataset to download
    parser.add_argument(
        '--dataset', '-d', type=str, default=None, 
        help='Specify download: "small", "medium", "large",\
             "full", or "metadata"'
    )
    # Parse the arguments
    args = parser.parse_args()
    
    # Print dataset that will be downloaded
    if args.dataset is None:
        args.dataset = 'metadata'
        print('No dataset specified. Defaulting to "metadata" dataset.\n')
        print(f'Downloading FMA {args.dataset} dataset...')

    # Define the path to save the noise data
    fma_path = os.path.join(os.environ['DATA_DIR'], 'fma')
    # Get the noise data name
    dataset_name = f'fma_{args.dataset}'

    # Download the noise data if it does not exist
    if not os.path.exists(
        os.path.join(fma_path, dataset_name)
        ):
        # Get noise data
        download_data(
            url=FMA_DICT[dataset_name],
            data_path=fma_path
        )
    else:
        print(f'{dataset_name} dataset already exists in {fma_path}.')
