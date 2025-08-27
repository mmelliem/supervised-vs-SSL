# Import necessary libraries
import lightning as L
import torch
import hydra

from omegaconf import DictConfig, OmegaConf
from mlp import MLP
from mnist_datamodule import MNISTDataModule
from audio_rep_networks.utils import get_device
from lightning.pytorch.loggers import WandbLogger

# Set torch precision for fast training on QUEST A100s, H100s
torch.set_float32_matmul_precision('high')

# Define the main function using hydra decorator
@hydra.main(version_base=None, config_path="./configs",
            config_name="submitit_single")
def train(cfg: DictConfig):
    # Get device
    device, num_gpus = get_device()

    print(f"Device: {device}, Number of GPUs: {num_gpus}")

    # Create datamodule
    dm = MNISTDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size
    )
    # Create logger
    wandb_logger = WandbLogger(
        project="audio_representations",
        save_dir=cfg.trainer.root_dir,
        offline=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    # Create model
    model = MLP(cfg)
    # Create trainer
    trainer = L.Trainer(
        accelerator=device,
        logger=wandb_logger,
        strategy="ddp" if num_gpus > 1 else "auto",
        devices=num_gpus if num_gpus > 1 else "auto",
        max_epochs=cfg.trainer.max_epochs,
        default_root_dir=cfg.trainer.root_dir
    )
    # Train the model
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
