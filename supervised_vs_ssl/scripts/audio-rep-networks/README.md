# audio-rep-networks
# About
[tbd]

# Setup
## Setting up submodules
After cloning this repository, run the following commands to initialize the submodules in the `libs` folder:
```
git submodule init
git submodule update
```

## Setting up Anaconda Environment
All necessary libraries are in the `environment.yml` and `requirements.txt` 
file. Run the following command to create the environment:
```
conda env create -n audio_rep
conda activate audio_rep
pip install -e .
```

From the main repository directory, add these submodules and folders to your 
conda env by running the following:
```
cd libs/chcochleagram
pip install -e .
```

### Setting up .env file
In the `audio-rep-networks` directory, create a `.env` file specifying the following fields:
```
HOME_DIR = # Absolute path to audio-rep-networks
DATA_DIR = # Absolute path where you want to store the data
```

## Data
You can download the data used for training and evaluation by running the
`get_fma_data.py` and `get_noise_data.py` files in the `audio_rep_networks`
folder. (NOTE: make sure .env file is setup for this).

The data used to train these models comes from the Free Music Archive (FMA)
dataset (learn more about it [here](https://github.com/mdeff/fma)). 
Additionally, background noise of auditory scenes and speech were from the 
[DCASE Challenge](https://dcase.community/challenge2013/task-acoustic-scene-classification) and 
[LibreVox](https://librivox.org), respectively.

Specific links can be found in the `noise_data.yaml` file in the
`audio_rep_networks` folder.

