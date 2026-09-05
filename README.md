# TERS-ABNet

**TERS-ABNet** is an AI-based framework for molecular structure reconstruction from tip-enhanced Raman spectroscopy (TERS) mappings. The framework integrates two neural networks, **ANet** and **BNet**, to identify atomic positions and chemical-bond/functional-group features from TERS mapping data and reconstruct the corresponding molecular structures.

This repository provides the source code, trained model checkpoints, and representative examples used in the main text.

---

## Repository Structure

```text
TERS-ABNet/
│
├── checkpoints/
│   ├── plane/
│   │   ├── ANet.pth
│   │   └── BNet.pth
│   │
│   └── nonplane/
│       ├── ANet.pth
│       └── BNet.pth
│
├── examples/
│   ├── plane/
│   │   └── ...
│   │
│   ├── nonplane/
│   │   └── ...
│   │
│   └── MgP/
│       ├── theory/
│       └── experiment/
│
├── scripts/
│   ├── train_ABNet.sh
│   ├── test_ABNet_plane_mole.sh
│   ├── test_ABNet_nonplane_mole.sh
│   └── test_ABNet_plane_MgP.sh
│
├── src/
│   ├── models/
│   │   ├── attention_unet.py
│   │   └── attention_gate.py
│   │
│   ├── train_ABNet.py
│   ├── test_ABNet.py
│   └── ...
│
├── README.md
├── requirements.txt
└── LICENSE
```

### `checkpoints/`

This directory contains the trained model parameters (`.pth`) used in this work.

* **`plane/`**: models trained for planar molecules.

  * `ANet.pth`: trained ANet for planar molecules.
  * `BNet.pth`: trained BNet for planar molecules.

* **`nonplane/`**: models obtained after transfer learning for non-planar molecules.

  * `ANet.pth`: transfer-learned ANet for non-planar molecules.
  * `BNet.pth`: transfer-learned BNet for non-planar molecules.

The provided checkpoints allow users to directly reproduce the molecular reconstruction examples without retraining the networks.

---

### `examples/`

This directory contains the input data and representative prediction results corresponding to the examples presented in the main text.

The examples include:

* TERS mappings used as model inputs;
* predicted atomic probability maps;
* predicted chemical-bond/functional-group maps;
* predicted atomic and bond information;
* reconstructed molecular structures;
* theoretical and experimental MgP examples.

The example results can be regenerated using the shell scripts in the `scripts/` directory.

---

### `scripts/`

This directory contains shell scripts for model training and testing.

| Script                        | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| `train_ABNet.sh`              | Train ANet and BNet                                                         |
| `test_ABNet_plane_mole.sh`    | Test the trained models on all planar molecules in `examples/`              |
| `test_ABNet_nonplane_mole.sh` | Test the transfer-learned models on all non-planar molecules in `examples/` |
| `test_ABNet_plane_MgP.sh`     | Test theoretical and experimental MgP examples                              |

---

### `src/`

This directory contains the core Python source code of TERS-ABNet.

```text
src/
├── models/
│   ├── attention_unet.py
│   └── attention_gate.py
│
├── train_ABNet.py
├── test_ABNet.py
└── ...
```

The `models/` directory contains the neural-network architectures, including the Attention U-Net architecture and attention-gate modules.

The training and testing source code provides the core implementation for model training, inference, and molecular structure reconstruction.

---

# Installation

## Requirements

The code requires Python and the following major packages:

* PyTorch
* NumPy
* SciPy
* Matplotlib
* scikit-learn

The complete package environment is provided in:

```text
requirements.txt
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for model training and large-scale inference.

---

# Model Architecture

TERS-ABNet consists of two neural networks: **ANet** and **BNet**.

## ANet

ANet predicts atomic-level information from the input TERS mapping.

For the models used in this work:

```text
Input:
160-channel TERS mapping
128 × 128 spatial resolution

Output:
4-channel atomic probability maps
128 × 128 spatial resolution
```

The four output channels correspond to the predefined atomic classes considered by ANet.

## BNet

BNet predicts chemical-bond and functional-group information from the TERS mapping.

```text
Input:
160-channel TERS mapping
128 × 128 spatial resolution

Output:
9-channel bond/functional-group probability maps
128 × 128 spatial resolution
```

The outputs of ANet and BNet are subsequently used for molecular graph reconstruction.

---

# Input Data

The model input consists of TERS mapping data represented as a multi-channel hyperspectral image.

For the examples used in this work, the input has the form:

```text
160 × 128 × 128
```

where:

* `160` represents the spectral channels;
* `128 × 128` represents the spatial dimensions.

The TERS mappings are generated by integrating the processed Raman spectra into predefined spectral windows.

---

# Inference Workflow

The general inference workflow is:

```text
                 TERS mapping
                      │
                      ▼
                    ANet
                      │
                      ▼
          Atomic probability maps
                      │
                      │
                      ▼
                    BNet
                      │
                      ▼
     Bond / functional-group probability maps
                      │
                      ▼
        Molecular graph reconstruction
                      │
                      ▼
       Reconstructed molecular structure
```

The provided testing scripts automate this workflow for planar molecules, non-planar molecules, and MgP.

---

# Using the Pre-trained Models

The repository provides trained checkpoints, allowing users to directly perform inference without retraining the models.

## 1. Planar molecules

To reproduce the planar-molecule examples from the main text, run:

```bash
bash scripts/test_ABNet_plane_mole.sh
```

This script uses the trained planar-molecule ANet and BNet models in:

```text
checkpoints/plane/
```

and processes the corresponding input TERS mappings in:

```text
examples/plane/
```

The predicted atomic information, bond information, and reconstructed molecular structures are generated in the corresponding example directories.

---

## 2. Non-planar molecules

The non-planar molecule examples use the transfer-learned models.

Run:

```bash
bash scripts/test_ABNet_nonplane_mole.sh
```

The script uses:

```text
checkpoints/nonplane/ANet.pth
checkpoints/nonplane/BNet.pth
```

and processes the examples stored in:

```text
examples/nonplane/
```

---

## 3. MgP

Theoretical and experimental MgP TERS data can be tested using:

```bash
bash scripts/test_ABNet_plane_MgP.sh
```

This script processes the MgP examples stored under:

```text
examples/MgP/
```

including both theoretical and experimental TERS data.

---

# Training

The training pipeline for ANet and BNet is provided in:

```text
src/train_ABNet.py
```

The corresponding shell script is:

```text
scripts/train_ABNet.sh
```

Run:

```bash
bash scripts/train_ABNet.sh
```

The training script supports training of both ANet and BNet using a unified training pipeline, with network-specific configurations for the output dimensions, target classes, attention mechanism, and loss weights.

The trained model parameters are saved as PyTorch `.pth` files.

---

# Testing

The general testing scripts are provided in the `scripts/` directory.

### Planar molecules

```bash
bash scripts/test_ABNet_plane_mole.sh
```

### Non-planar molecules

```bash
bash scripts/test_ABNet_nonplane_mole.sh
```

### MgP

```bash
bash scripts/test_ABNet_plane_MgP.sh
```

These scripts load the corresponding pre-trained model checkpoints and process the TERS mappings stored in `examples/`.

---

# Output

The inference pipeline produces several types of output, including:

1. **Atomic probability maps**
2. **Chemical-bond/functional-group probability maps**
3. **Predicted atomic positions**
4. **Predicted chemical-bond information**
5. **Reconstructed molecular graphs**
6. **Reconstructed molecular structures**

Representative results corresponding to the examples in the main text are included in the `examples/` directory.

---

# Reproducing the Main-Text Examples

The main-text examples can be reproduced using the provided shell scripts without retraining the models.

### Planar molecules

```bash
bash scripts/test_ABNet_plane_mole.sh
```

### Non-planar molecules

```bash
bash scripts/test_ABNet_nonplane_mole.sh
```

### Theoretical and experimental MgP

```bash
bash scripts/test_ABNet_plane_MgP.sh
```

The scripts automatically load the appropriate pre-trained models from `checkpoints/` and process the corresponding input data in `examples/`.

---

# Citation

If you use TERS-ABNet in your research, please cite the corresponding publication:

```text
[Add the final publication citation here]
```

---

# License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

The **pre-trained model weights and benchmark datasets** are released under the **CC-BY-NC 4.0** license.

For details of the CC-BY-NC 4.0 license, see:

https://creativecommons.org/licenses/by-nc/4.0/

### MIT License Text

```text
MIT License

Copyright (c) 2026 JieCui@USTC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Insert the remaining MIT License text here.]
```
