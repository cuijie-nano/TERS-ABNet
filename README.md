# TERS-ABNet

**TERS-ABNet** is an AI-based framework for molecular structure reconstruction from tip-enhanced Raman spectroscopy (TERS) mappings. The framework integrates two neural networks, **ANet** and **BNet**, to identify atomic positions and chemical-bond/functional-group features from TERS mapping data and reconstruct the corresponding molecular structures. 

The corresponding publication is in
```text
https://arxiv.org/abs/2603.21579
```

This repository provides the source code, trained model checkpoints, and representative examples used in the main text.

---


### `checkpoints/`

This directory contains the trained model parameters (`.pth`) used in this work.

* **`ANet_base/`**: trained ANet parameters for planar molecules.
* **`BNet_base/`**: trained BNet parameters for planar molecules.


* **`ANet_transfer/`**: trained ANet parameters after transfer learning for non-planar molecules.
* **`BNet_transfer/`**: trained ANet parameters after transfer learning for non-planar molecules.


The provided checkpoints allow users to directly reproduce the molecular reconstruction examples without retraining the networks.

---

### `examples/`

This directory contains the input data and representative prediction results corresponding to the examples presented in the main text.

The examples include:

* TERS mappings used as model inputs;
* predicted atom and bond maps;
* reconstructed molecular structures;

The example results can be regenerated using the shell scripts in the `scripts/` directory.

---

### `scripts/`

This directory contains shell scripts for model training and testing.

| Script                        | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| `train_ABNet.sh`              | Train ANet and BNet                                                         |
| `test_ABNet_plane_mole.sh`    | Test the trained models on all planar molecules in `examples/`              |
| `test_ABNet_nonplane_mole.sh` | Test the transfer-learned models on all non-planar molecules in `examples/` |
| `test_ABNet_plane_MgP.sh`     | Test simulated and experimental MgP examples                              |

---

### `src/`

This directory contains the core Python source code of TERS-ABNet.

```text
src/
├── models/
│   ├── attention_unet.py
│   └── attention_gate.py
│
├── train_ANet_base.py
├── test_ABNet_base.py
└── ...
```

The `models/` directory contains the neural-network architectures, including the Attention U-Net architecture and attention-gate modules.

The training and testing source code provides the core implementation for model training, testing.

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


# Using the Pre-trained Models

The repository provides trained checkpoints, allowing users to directly perform inference without retraining the models.

## 1. Planar molecules

To reproduce the planar-molecule examples from the main text, run:

```bash
bash scripts/test_ABNet_plane_mole.sh
```

The predicted atom, bond maps, and reconstructed molecular structures are generated in the corresponding example directories.

---

## 2. Non-planar molecules

To reproduce the nonplanar-molecule examples from the main text, run:

```bash
bash scripts/test_ABNet_nonplane_mole.sh
```

The predicted atom, bond maps, and reconstructed molecular structures are generated in the corresponding example directories.

---

## 3. MgP

Theoretical and experimental MgP TERS data can be tested using:

```bash
bash scripts/test_ABNet_plane_MgP.sh
```

This script processes the MgP examples stored under:

```text
examples/**MgP**/
```

including both theoretical and experimental TERS data.

---


# Citation

If you use TERS-ABNet in your research, please cite the corresponding publication:

```text
https://arxiv.org/abs/2603.21579
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
