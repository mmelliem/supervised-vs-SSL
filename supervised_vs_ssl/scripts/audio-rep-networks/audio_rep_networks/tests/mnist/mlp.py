import lightning as L
import torch
import torch.nn as nn


class MLP(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Store input size, hidden size, and output size
        self.input_size = config.model.input_size
        self.hidden_size = config.model.hidden_size
        self.output_size = config.model.output_size
        
        # Define the layers of the MLP
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()

        # Define the loss function
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # Flatten the input tensor
        # x = x.view(-1, self.input_size) #NOTE: don't think I need this
        # Pass the input through the first fully connected layer and apply
        # ReLU activation
        x = self.relu(self.fc1(x))
        # Pass the output through the second fully connected layer
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        # Extract input and target from the batch
        x, y = batch
        # Forward pass through the model
        y_hat = self(x)
        # Calculate the loss using cross entropy
        loss = self.loss(y_hat, y)
        # Log the loss
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # Extract input and target from the batch
        x, y = batch
        # Forward pass through the model
        y_hat = self(x)
        # Calculate the loss using cross entropy
        loss = self.loss(y_hat, y)
        # Calculate the accuracy
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        # Log the accuracy and loss
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        return loss

    def configure_optimizers(self):
        # Use Adam
        return torch.optim.Adam(self.parameters(), lr=1e-3)
