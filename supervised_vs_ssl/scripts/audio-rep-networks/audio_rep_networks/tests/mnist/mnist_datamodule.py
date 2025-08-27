import lightning as L

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

"""
TODO: Parameters to specify in Hydra config
- batch size
- data dir (maybe)
"""


# Create DataModule class for MNIST dataset
class MNISTDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            batch_size: int = 64
            ):
        super().__init__()
        # Store data directory and batch size
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Download the MNIST dataset
        MNIST(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        # Define transforms for the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.reshape(-1, 1).squeeze(),
            lambda x: x.float()
        ])
        # Load the MNIST dataset for training
        self.mnist_train = MNIST(
            self.data_dir,
            train=True,
            download=False,
            transform=transform
        )
        # Load the MNIST dataset for validation
        self.mnist_val = MNIST(
            self.data_dir,
            train=False,
            download=False,
            transform=transform
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)
