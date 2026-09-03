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

def augment_data_norot(data, noise=0.02):
    """Apply rotational augmentation and Gaussian noise."""

    print("Applying data augmentation...")

   

    if noise > 0:
        data_aug = data + (
            torch.randn_like(data) * noise
        )
     
    return data_aug
# ============================================================
# Dataset splitting
# ============================================================

def split_dataset_FG_QM(
    data,
    targets,
    data1_mols,
    data2_mols,
    train_ratio,
    indices_file1,
    indices_file2,

):
    """Split data into training and test sets."""

    print("Splitting dataset...")

    full_dataset = TensorDataset(
        data,
        targets,
    )


    train1_size = int(
        train_ratio * data1_mols
    )
    test1_size = data1_mols - train1_size
    # --------------------------------------------------------
    # Generate or load molecular indices
    # --------------------------------------------------------
    if not os.path.exists(indices_file1):

        indices1 = torch.randperm(data1_mols)  #
        indices1 = torch.cat([indices1[:train1_size],                      #train0
                             indices1[:train1_size]+data1_mols,           #train90
                             indices1[:train1_size]+data1_mols*2,         #train180
                             indices1[:train1_size]+data1_mols*3,         #train270
                             indices1[-test1_size:],                      #test0
                             indices1[-test1_size:]+data1_mols,           #test90
                             indices1[-test1_size:]+data1_mols*2,         #test180
                             indices1[-test1_size:]+data1_mols*3,         #test270
            ])
        torch.save(indices1, indices_file1)

    else:

        indices1 = torch.load(
            indices_file1,
            weights_only=True,
        )
        
        
    train2_size = int(
        train_ratio * data2_mols
    )
    test2_size = data2_mols - train2_size
    
    if not os.path.exists(indices_file2):
        indices2 = torch.randperm(data2_mols)  # ])
        torch.save(indices2, indices_file2)
    else:
        indices2 = torch.load(indices_file2)
        
    indices_tot = torch.cat([indices1, indices2+data1_mols*4])

    # --------------------------------------------------------
    # Create subsets
    # --------------------------------------------------------

    train_indices = torch.cat([ indices_tot[:int(train1_size*4)], 
                                indices_tot[int(data1_mols*4) : int(data1_mols*4)+int(train2_size)] 
                                ])
    test_indices = torch.cat([  indices_tot[int(train1_size*4)  :int(data1_mols*4)],
                                indices_tot[int(data1_mols*4)+int(train2_size) :]
                                ])
    
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
        "--data_file1",
        nargs="+",
        default=[
            "FG_CNO05A_09A_mapping_data.npy"
        ],
    )

    parser.add_argument(
        "--targets_file1",
        nargs="+",
        default=[
            "FG_CNO05A_H09A_atommapping_decay5.npy"
        ],
    )

    parser.add_argument(
        "--data_file2",
        nargs="+",
        default=[
            "QM_CNO05A_09A_mapping_data.npy"
        ],
    )

    parser.add_argument(
        "--targets_file2",
        nargs="+",
        default=[
            "QM_CNO05A_H09A_atommapping_decay5.npy"
        ],
    )

    parser.add_argument(
        "--indices_file1",
        type=str,
        default="indices_FG9_x4.pth",
    )
    
    parser.add_argument(
        "--indices_file2",
        type=str,
        default="indices_QM9_x1.pth",
    )
    
    parser.add_argument(
        '--predir', 
        type=str, 
        default='../checkpoints/ANet_base/')

    parser.add_argument(
        "--savedir",
        type=str,
        default="../checkpoints/ANet_transfer/",
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
        default=4,
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
        default="softmax",
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

    model.load_state_dict(torch.load(args.predir+"model_100.pth"))  
    # 冻结前层和后层
    for param in model.conv2d_1.parameters():
        param.requires_grad = True
    for param in model.conv2d_2.parameters():
        param.requires_grad = True
    for param in model.conv2d_3.parameters():
        param.requires_grad = True
    for param in model.conv2d_4.parameters():
        param.requires_grad = True
    for param in model.conv2d_5.parameters():
        param.requires_grad = True
    for param in model.ag1.parameters():
        param.requires_grad = True 
    for param in model.conv2d_6.parameters():
        param.requires_grad = True  
    for param in model.conv2d_7.parameters():
        param.requires_grad = True
    for param in model.ag2.parameters():
        param.requires_grad = True 
    for param in model.conv2d_8.parameters():
        param.requires_grad = True
    for param in model.conv2d_9.parameters():
        param.requires_grad = True
    for param in model.ag3.parameters():
        param.requires_grad = True 
    for param in model.conv2d_10.parameters():
        param.requires_grad = True
    for param in model.conv2d_11.parameters():
        param.requires_grad = True
    for param in model.conv2d_12.parameters():
        param.requires_grad = True
        
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data1, targets1 = load_data(
        args.data_file1,
        args.targets_file1,
    )
    
    data2, targets2 = load_data(
        args.data_file2,
        args.targets_file2,
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    data1 = normalize_data(
        data1,
        self_normalized=args.self_normalized,
    )
    
    data2 = normalize_data(
        data2,
        self_normalized=args.self_normalized,
    )

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------
    data1, targets1 = augment_data(
        data1,
        targets1,
        noise=args.noise,
    )
    data1_mols = int(data1.size(0)/4)

    data2= augment_data_norot(
        data2,
        noise=args.noise,
    )
    data2_mols = int(data2.size(0))    

    data = torch.cat(
        [
            data1,
            data2,
        ],
        dim=0,
    )
    
    targets = torch.cat(
        [
            targets1,
            targets2,
        ],
        dim=0,
    )
    

    # --------------------------------------------------------
    # Dataset splitting
    # --------------------------------------------------------

    train_dataset, test_dataset = split_dataset_FG_QM(
        data,
        targets,
        data1_mols,
        data2_mols,
        train_ratio=args.train_ratio,
        indices_file1=args.indices_file1,
        indices_file2=args.indices_file2,
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
            0.7121132,
            1.13728947,
            2.13719746,
            0.80093372,
        ],
        dtype=torch.float32,
    ).view(1, 4, 1, 1)

    # --------------------------------------------------------
    # Optimizer and scheduler
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)


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
            f"model_epoch_{epoch}.pth",
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