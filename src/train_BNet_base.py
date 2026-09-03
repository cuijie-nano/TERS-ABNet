# -*- coding: utf-8 -*-

"""
Training script for ANet.

Author:
    Jie Cui (崔杰)
    University of Science and Technology of China
    Hefei National Laboratory for Physical Science at the Microscale
"""

import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset, DataLoader

from models.attention_unet import AttUnet


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    weights,
    epoch,
):
    """Train the model for one epoch."""

    model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        loss = torch.mean(loss * weights)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print(
            f"Train Epoch: {epoch} "
            f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
            f"({100. * batch_idx / len(train_loader):.0f}%)] "
            f"Loss: {loss.item():.9f}"
        )

    return train_loss / len(train_loader)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    test_loader,
    criterion,
    weights,
):
    """Evaluate the model on the test set."""

    model.eval()

    test_loss = 0.0

    predictions = None
    targets = None

    for data, target in test_loader:

        prediction = model(data)

        loss = criterion(prediction, target)
        loss = torch.mean(loss * weights)

        test_loss += loss.item()

        predictions = prediction
        targets = target

    test_loss /= len(test_loader)

    print(
        f"\nTest set: Average loss: "
        f"{test_loss:.9f}\n"
    )

    return predictions, targets, test_loss


# ============================================================
# Data loading
# ============================================================

def load_data(data_files, target_files):
    """Load and concatenate input and target data."""

    print("Loading data...")

    data = [
        torch.from_numpy(
            np.load(path).astype(np.float32)
        )
        for path in data_files
    ]

    targets = [
        torch.from_numpy(
            np.load(path).astype(np.float32)
        )
        for path in target_files
    ]

    data = torch.cat(data, dim=0)
    targets = torch.cat(targets, dim=0)

    targets = targets[:, [2, 3, 4, 5, 6, 7, 8,9,  10], :, :]
    print(f"Data shape:    {tuple(data.shape)}")
    print(f"Target shape:  {tuple(targets.shape)}")

    return data, targets


# ============================================================
# Data normalization
# ============================================================

def normalize_data(data, self_normalized=True):
    """Normalize input TERS mappings."""

    print("Normalizing data...")

    if not self_normalized:

        mean = 0.0
        std = 0.0

        for i in range(data.shape[1]):
            mean += torch.mean(data[:, i, :, :])
            std += torch.std(data[:, i, :, :])

        mean /= data.shape[1]
        std /= data.shape[1]

        data = (data - mean) / std

    else:

        for i in range(data.shape[0]):

            data_min = torch.min(data[i])
            data_max = torch.max(data[i])

            data[i] = (
                (data[i] - data_min)
                / (data_max - data_min)
            )

    return data


# ============================================================
# Data augmentation
# ============================================================

def augment_data(data, targets, noise=0.02):
    """Apply rotational augmentation and Gaussian noise."""

    print("Applying data augmentation...")

    data_aug = torch.cat(
        [
            data,
            torch.rot90(data, k=1, dims=(2, 3)),
            torch.rot90(data, k=2, dims=(2, 3)),
            torch.rot90(data, k=3, dims=(2, 3)),
        ],
        dim=0,
    )

    targets_aug = torch.cat(
        [
            targets,
            torch.rot90(targets, k=1, dims=(2, 3)),
            torch.rot90(targets, k=2, dims=(2, 3)),
            torch.rot90(targets, k=3, dims=(2, 3)),
        ],
        dim=0,
    )

    if noise > 0:
        data_aug = data_aug + (
            torch.randn_like(data_aug) * noise
        )

    return data_aug, targets_aug


# ============================================================
# Dataset splitting
# ============================================================

def split_dataset(
    data,
    targets,
    train_ratio,
    indices_file,
):
    """Split data into training and test sets."""

    print("Splitting dataset...")

    full_dataset = TensorDataset(
        data,
        targets,
    )

    # Original molecules before rotational augmentation
    total_mols = data.shape[0] // 4

    train_size = int(
        train_ratio * total_mols
    )

    test_size = total_mols - train_size

    # --------------------------------------------------------
    # Generate or load molecular indices
    # --------------------------------------------------------

    if not os.path.exists(indices_file):

        indices0 = torch.randperm(total_mols)

        indices = torch.cat(
            [
                indices0[:train_size],
                indices0[:train_size] + total_mols,
                indices0[:train_size] + 2 * total_mols,
                indices0[:train_size] + 3 * total_mols,

                indices0[-test_size:],
                indices0[-test_size:] + total_mols,
                indices0[-test_size:] + 2 * total_mols,
                indices0[-test_size:] + 3 * total_mols,
            ]
        )

        torch.save(indices, indices_file)

    else:

        indices = torch.load(
            indices_file,
            weights_only=True,
        )

    # --------------------------------------------------------
    # Create subsets
    # --------------------------------------------------------

    train_indices = indices[
        : train_size * 4
    ]

    test_indices = indices[
        -test_size * 4 :
    ]

    train_dataset = Subset(
        full_dataset,
        train_indices,
    )

    test_dataset = Subset(
        full_dataset,
        test_indices,
    )

    return train_dataset, test_dataset


# ============================================================
# Save training history
# ============================================================

def save_loss(losses, filename):
    """Save loss values to a text file."""

    with open(filename, "w") as f:

        for loss in losses:
            f.write(f"{loss}\n")


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Train TERS-ABNet"
    )

    # Data
    parser.add_argument(
        "--data_file",
        nargs="+",
        default=[
            "QM_FG_CNO05A_H05_08A_mapping_data.npy"
        ],
    )

    parser.add_argument(
        "--targets_file",
        nargs="+",
        default=[
            "QM_FG_CNO05A_H05_08A_bondmapping_decay5.npy"
        ],
    )

    parser.add_argument(
        "--indices_file",
        type=str,
        default="indices_QM5678_FG5678_x4.pth",
    )

    parser.add_argument(
        "--savedir",
        type=str,
        default="./checkpoints/BNet_base/",
    )

    # Data processing
    parser.add_argument(
        "--self_normalized",
        action="store_true",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
    )

    # Model
    parser.add_argument(
        "--out_channels",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--x_channels",
        nargs="+",
        type=int,
        default=[64, 64, 96],
    )

    parser.add_argument(
        "--q_channels",
        nargs="+",
        type=int,
        default=[64, 64, 64],
    )

    parser.add_argument(
        "--attention_type",
        type=str,
        default="sigmoid",
        choices=["softmax", "sigmoid"],
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.92,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    os.makedirs(
        args.savedir,
        exist_ok=True,
    )

    # Save arguments
    with open(
        os.path.join(args.savedir, "args.pkl"),
        "wb",
    ) as f:
        pickle.dump(args, f)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AttUnet(
        out_channels=args.out_channels,
        x_channels=args.x_channels,
        q_channels=args.q_channels,
        attention_type=args.attention_type,
    )


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data, targets = load_data(
        args.data_file,
        args.targets_file,
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    data = normalize_data(
        data,
        self_normalized=args.self_normalized,
    )

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    data, targets = augment_data(
        data,
        targets,
        noise=args.noise,
    )

    # --------------------------------------------------------
    # Dataset splitting
    # --------------------------------------------------------

    train_dataset, test_dataset = split_dataset(
        data,
        targets,
        train_ratio=args.train_ratio,
        indices_file=args.indices_file,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
    )


    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = nn.MSELoss(
        reduction="none"
    )

    weights = torch.tensor(
        [
            0.64866437, 
            3.1267838 , 
            1.94359842, 
            5.37902654, 
            0.46735217, 
            0.3940476 , 
            0.9373045 , 
            2.66315383, 
            3.14194538
        ],
        dtype=torch.float32,
    ).view(1, 9, 1, 1)

    # --------------------------------------------------------
    # Optimizer and scheduler
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_losses = []
    test_losses = []

    print("\nStart training...\n")

    for epoch in range(1, args.num_epochs + 1):

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            weights=weights,
            epoch=epoch,
        )

        _, _, test_loss = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            weights=weights,
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        scheduler.step()

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        checkpoint_path = os.path.join(
            args.savedir,
            f"model_{epoch}.pth",
        )

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        # ----------------------------------------------------
        # Save losses
        # ----------------------------------------------------

        save_loss(
            train_losses,
            os.path.join(
                args.savedir,
                "train_loss.txt",
            ),
        )

        save_loss(
            test_losses,
            os.path.join(
                args.savedir,
                "test_loss.txt",
            ),
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.9f} | "
            f"Test Loss: {test_loss:.9f}"
        )


if __name__ == "__main__":
    main()