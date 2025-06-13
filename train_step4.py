import pathlib
import numpy as np
import pandas as pd
import torch

from thesession.dataset import TheSessionDataset
from thesession.model import TheSessionModel
from thesession.training import train_model, NTXentLoss

# Datasets
dataset = pd.read_csv("dataset.csv")

test_subset = dataset.loc[dataset["dataset"] == "test", "tune"]
val_subset = dataset.loc[dataset["dataset"] == "validation", "tune"]
train_subset = dataset.loc[dataset["dataset"] == "train", "tune"]

train_dataset = TheSessionDataset(
    "audio_flac",
    subset=train_subset,
    sampling_rate=48000,
    fmt=".flac",
    backend="soundfile",
)
val_dataset = TheSessionDataset(
    "audio_flac",
    subset=val_subset,
    sampling_rate=48000,
    fmt=".flac",
    backend="soundfile",
)
test_dataset = TheSessionDataset(
    "audio_flac",
    subset=test_subset,
    sampling_rate=48000,
    fmt=".flac",
    backend="soundfile",
)

print("Training data", len(train_dataset))
print("Validation data", len(val_dataset))
print("Test data", len(test_dataset))

# Model
model = TheSessionModel()
model.load("models/step3_best.pt")

model.toggle_gradients(False, verbose=False)
model.toggle_gradients(
    True,
    [
        "clap_model.model.audio_branch.layers.3",
        "clap_model.model.audio_branch.norm",
        "clap_model.model.audio_branch.tscam_conv",
        "clap_model.model.audio_branch.head",
        "clap_model.model.audio_projection",
    ],
    verbose=True,
)

# Training
criterion = NTXentLoss(temperature=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

train_model(
    model,
    train_dataset,
    val_dataset,
    criterion,
    optimizer,
    epochs=20,
    device="cuda",
    batch_size=32,
    num_workers=6,
    history_path="training/step4.csv",
    best_model_path="models/step4_best.pt",
    last_model_path="models/step4_last.pt",
)
