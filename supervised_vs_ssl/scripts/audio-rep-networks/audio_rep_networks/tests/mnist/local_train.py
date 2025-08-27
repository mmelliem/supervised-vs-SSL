import lightning as L
import hydra

from omegaconf import DictConfig
from mlp import MLP
from mnist_datamodule import MNISTDataModule
from audio_rep_networks.utils import get_device
from lightning.pytorch.loggers import WandbLogger

# Define the main function using hydra decorator
@hydra.main(config_path="./configs", config_name="single_model")
def main(cfg: DictConfig):
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
        offline=True
    )
    # Create model
    model = MLP(cfg)
    # Create trainer
    trainer = L.Trainer(
        accelerator=device,
        logger=wandb_logger,
        strategy="ddp" if num_gpus > 1 else "auto",
        devices=num_gpus if num_gpus > 1 else "auto",
        max_epochs=cfg.trainer.max_epochs
    )
    # Train the model
    trainer.fit(model, dm)

    # Print training is finished
    print("Training finished!!!")

# Run the main function
if __name__ == "__main__":
    main()
