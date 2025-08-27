# Import necessary libraries
import lightning as L
import torch
import torch.nn as nn

from torchmetrics.classification import MultilabelHammingDistance

# Create model class for Kell et. al 2018
#(https://doi.org/10.1016/j.neuron.2018.03.044)

class KellCNN(L.LightningModule):
    def __init__(self):
        # TODO: Add parameters to Hydra config instead of hardcoding.
        # Hardcoded parameters taken from kelletal2018 repo in libs folder
        # Parameters
        self.rnorm_bias = 1.
        self.rnorm_alpha = 1e-3
        self.rnorm_beta = 0.75

        # Input Cochleagram: (B, 1, 256, 256)
        self.agg_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 96, kernel_size=9, stride=3),
            nn.ReLU(),
            nn.LocalResponseNorm(size=2, alpha=self.rnorm_alpha, 
                                 beta=self.rnorm_beta, k=self.rnorm_bias),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 2
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=2, alpha=self.rnorm_alpha, 
                                 beta=self.rnorm_beta, k=self.rnorm_bias),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv Layer 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            # Conv Layer 4
            nn.Conv2d(512, 1024, kernel_size=3, stride=1),
            nn.ReLU(),
            # Conv Layer 5
            nn.Conv2d(1024, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            # Final Pooling Layer
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            # Fully Connected 1
            nn.Linear(512 * 6 * 6, 1024),
            nn.Dropout(p=0.5), # NOTE: Seems steep, but is what they used
            # Fully Connected 2
            nn.Linear(1024, 40)
        )

        # Define the loss function
        # TODO: Add to Hydra config, and add util function to get loss function
        self.loss = nn.BCEWithLogitsLoss()
        # Define the accuracy metric
        self.acc = MultilabelHammingDistance(
            num_labels=40,
            threshold=0.5
        )

    def forward(self, x):
        # Aggregate blocks
        x = self.agg_blocks(x)
        # Convolutional layers
        x = self.conv_layers(x)
        # Flatten
        x = torch.flatten(x, 1)
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x

    def training_step(self, batch, batch_idx):
        # Get input and labels from batch
        x, y = batch
        # Forward pass
        train_y_hat = self(x)
        # Calculate loss
        train_loss = self.loss(train_y_hat, y)

        # Log loss
        self.log_dict(
            {
                "train_loss": train_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return train_loss


    def validation_step(self, batch, batch_idx):
        # Get input and labels from batch
        x, y = batch
        # Forward pass
        val_y_hat = self(x)
        # Calculate loss
        val_loss = self.loss(val_y_hat, y)

        # Calculate accuracy (Auto-applies sigmoid)
        val_acc = self.acc(val_y_hat, y)

        # Log loss and accuracy
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_acc": val_acc
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return val_loss

    def configure_optimizers(self):
        # Use Adam optimizer
        # NOTE: This might be different than how Kell did it.
        return torch.optim.Adam(self.parameters(), lr=1e-3) 