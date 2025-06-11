import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# Loss function
def nt_xent_loss(z1, z2, temperature=0.07):
    """
    Contrastive loss using implicit negatives (NT-Xent).
    Args:
        z1: Tensor of shape (N, D) – embeddings from view 1 (e.g., anchors)
        z2: Tensor of shape (N, D) – embeddings from view 2 (e.g., positives)
    Returns:
        Scalar contrastive loss
    """
    batch_size = z1.size(0)

    # Concatenate for full 2N x D
    z = torch.cat([z1, z2], dim=0)  # shape: (2N, D)

    # Cosine similarity (2N x 2N)
    sim = F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)

    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -float("inf"))  # ignore similarity to self

    # Targets: for i in 0..N-1, positive pair is i<->i+N and i+N<->i
    targets = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
    ).to(z.device)

    loss = F.cross_entropy(sim / temperature, targets)

    # Get correct answers for computing accuracy
    # Divided by 2 as both z1 -> z2 and z2 -> z1 are checked
    correct = torch.sum(torch.argmax(sim, dim=1) == targets).float() / 2.0

    return loss, correct


# Training function
def train_model(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    temperature=0.07,
    epochs=10,
    device=None,
    verbose=True,
    batch_size=32,
    num_workers=0,
    best_model_path="models/best_model.pt",
    destination=None,
):
    """Train a pytorch model

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    train_dataset : torch.utils.data.Dataset
        The dataset from which the model is trained
    val_dataset : torch.utils.data.Dataset
        The dataset that is evaluated at each epoch to follow the training process
    optimizer : torch.optim.Optimizer
        The optimizer that is used for model training
    temperature: float, default: 0.07
        The temperature for scaling the cosine similarity loss
    epochs : int, default: 10
        The number of epochs that are used to train the model
    device : torch.device, default: None
        The device onto which the model is trained
    verbose : bool, default: True
        Whether a summary of the training process is printed at each epoch or not.
    batch_size: int, default: 32
        The number of data points to load with each batch.
    best_model_path: str, default: "models/best_model.pt"
        The path to save the best model.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the average training and validation loss at each epoch,
        and the model accuracy on the training and validation data.

    """
    # Path to save the best model
    best_model_path = pathlib.Path(best_model_path)
    best_model_path.parent.mkdir(exist_ok=True)

    # Definition of data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    train_loss_array = np.zeros(epochs)
    val_loss_array = np.zeros(epochs)
    train_accuracy_array = np.zeros(epochs)
    val_accuracy_array = np.zeros(epochs)

    # Start time counter
    start_time = time.perf_counter()

    # Write destination file
    if destination:
        destination = pathlib.Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not destination.exists():
            with open(destination, "w") as f:
                f.write("epoch,train_loss,val_loss,train_accuracy,val_accuracy")

    # Iteration over the epochs
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0.0

        for i, (x1, x2) in enumerate(train_loader):
            if verbose:
                print(f"Train batch {i+1}/{len(train_loader)}", end="\r")

            # Send to device
            if device is not None:
                x1 = x1.to(device)
                x2 = x2.to(device)

            optimizer.zero_grad()

            # Compute embeddings
            z1 = model(x1)
            z2 = model(x2)

            # Compute loss
            batch_loss, batch_correct = nt_xent_loss(z1, z2, temperature=temperature)

            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()
            correct += batch_correct.item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / len(train_dataset)

        train_loss_array[epoch] = avg_train_loss
        train_accuracy_array[epoch] = train_accuracy

        # Validation
        model.eval()

        val_loss = 0.0
        correct = 0.0

        with torch.no_grad():
            for i, (x1, x2) in enumerate(val_loader):
                if verbose:
                    print(f"Validation batch {i+1}/{len(val_loader)}", end="\r")

                if device is not None:
                    x1 = x1.to(device)
                    x2 = x2.to(device)

                # Compute embeddings
                z1 = model(x1)
                z2 = model(x2)

                # Compute loss
                batch_loss, batch_correct = nt_xent_loss(
                    z1, z2, temperature=temperature
                )

                val_loss += batch_loss.item()
                correct += batch_correct.item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / len(val_dataset)

        val_loss_array[epoch] = avg_val_loss
        val_accuracy_array[epoch] = val_accuracy

        if avg_val_loss == np.min(val_loss_array[: epoch + 1]):
            torch.save(model.state_dict(), best_model_path)

        # Printing a summary
        if verbose:
            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.4f} | "
                f"Valid. Loss: {avg_val_loss:.4f} | "
                f"Valid. Accuracy: {val_accuracy:.4f} | "
            )

        # Exporting summary to destination
        if destination:
            with open(destination, "a") as f:
                f.write(
                    f"{epoch},{avg_train_loss},{avg_val_loss},{train_accuracy},{val_accuracy}"
                )

    # End time counter
    end_time = time.perf_counter()
    duration = end_time - start_time

    if verbose:
        print(f"Training time for {epochs} epochs: {duration:0.2f} seconds")

    # Regroup results into dataframe
    res = pd.DataFrame(
        {
            "train_loss": train_loss_array,
            "val_loss": val_loss_array,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy_array,
        },
        index=pd.RangeIndex(len(train_loss_array), name="epoch"),
    )

    return res
