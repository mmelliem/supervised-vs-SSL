import os
from setuptools import setup, find_packages

# Get repo root directory
root_dir = os.path.abspath(os.path.dirname(__file__))
# Get requirements from txt file
with open(os.path.join(root_dir, 'requirements.txt')) as f:
    requirements = [line for line in f.read().splitlines()
                    if not line.startswith('#')]
    
#Setup, TODO: Add license
setup(
    name='audio_rep_networks',
    packages=find_packages(),
    install_requires=requirements,
    version='0.1.0',
    description='Audio representation learning networks',
    author='Bareesh Bhaduri and Peer Herholz',
)
