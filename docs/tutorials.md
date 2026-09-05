# TERS-ABNet 使用与结果复现教程

本教程面向初次接触 TERS-ABNet 的科研人员和代码使用者。它说明项目的科学问题、代码和数据的组织方式，以及如何用已提供的权重复现论文中的代表性结果。默认仓库根目录为 `<repo>`，即本项目的 `code_upstall_new` 目录。

## 1. 项目概览与研究背景

### 1.1 科学问题

TERS-ABNet 面向一个高维反问题：由单分子 tip-enhanced Raman spectroscopy（TERS）mapping 推断显式的分子原子-化学键图（atom-bond graph）。单个振动模式的 TERS 图通常同时包含邻近原子和化学键的叠加、干涉贡献，不能被直接一一解释为某个原子或某条键；项目因此将多个波数窗口的二维 TERS 图联合输入深度网络，再由网络输出可用于构图的原子和键的空间概率图。

模型包含两条并行路径：

- **ANet**：预测 C、N、O、H 四种原子的多通道概率图；
- **BNet**：预测键/官能团类型的多通道概率图。平面模型使用 9 个通道；迁移学习模型使用 11 个通道，额外包含甲基和乙基；
- **后处理**：在概率图中检测高斯峰，获得原子和键的位置与类别；随后按距离、键/官能团约束和价态评分组合为分子图，并输出 SDF 结构文件。

论文的训练输入为 `128 × 128 × 160` 的 TERS mapping：空间覆盖范围为 `2.5 × 2.5 nm²`，160 个通道来自 20 cm⁻¹ 波数窗口积分。对应标签是以真实位置为中心的高斯斑点图。论文报告的独立测试集结果包括约 `0.233 Å` 的原子坐标 MAE、约 `0.199 Å` 的键位置 MAE、`94.2%` 的原子类型平均准确率和 `92.0%` 的键类型平均准确率。

### 1.2 代码、样例和论文结果的对应关系

| 论文结果 | 样例目录 | 使用入口 | 预期生成/查看的内容 |
| --- | --- | --- | --- |
| Fig. 1 工作流及 Supplementary Fig. 3 的 2,6-dihydroxypyridine 例子 | `examples/1_plane_mole/` | `test_ABNet_plane_mole.sh` | 160 通道 TERS 拼图、原子/键图、坐标、重建几何 |
| Fig. 3 的四个平面分子及 Supplementary Fig. 4 | `examples/3-1_plane_mole/` 至 `3-4_plane_mole/` | `test_ABNet_plane_mole.sh` | 平面分子的概率图和 `geometry.sdf` |
| Fig. 4 的分辨率泛化案例 | `examples/4-1_plane_mole/` 至 `4-3_plane_mole/` | `test_ABNet_plane_mole.sh` | 三种 TERS 输入条件的 atom/bond 概率图 |
| Fig. 5 的四个非平面案例及 Supplementary Fig. 6 | `examples/5-1_nonplane_mole/` 至 `5-4_nonplane_mole/` | `test_ABNet_nonplane_mole.sh` | 11 通道 BNet 的输出、坐标与几何 |
| Fig. 6 的模拟及实验 MgP | `examples/6-1_MgP_sim/`、`6-2_MgP_exp/` | `test_ABNet_plane_MgP.sh` | MgP 的 atom/bond 概率图 |

`docs/Maunscript-20260316-arxiv.docx` 是主论文初稿；`docs/SI-TERS-ABNet-20260316-clean.docx` 给出了网络结构、训练超参数、分辨率泛化与非平面体系补充结果。表中“论文结果”用于定位对应图，而非意味着脚本会自动排版生成论文整图。

## 2. 目录结构与模块说明

```text
<repo>/
├── checkpoints/                 # 4 个可直接推理的模型权重
│   ├── ANet_base/                # 平面分子原子网络
│   ├── BNet_base/                # 平面分子键网络（9 通道）
│   ├── ANet_transfer/            # 非平面迁移学习原子网络
│   └── BNet_transfer/            # 非平面迁移学习键网络（11 通道）
├── examples/                     # 论文代表性输入及可覆盖的结果文件
├── scripts/                      # 训练和三类推理的 Bash 入口
├── src/
│   ├── models/attention_unet.py  # Attention U-Net 主体
│   ├── models/attention_gate.py  # 跳连注意力门
│   ├── test_ABNet_base.py        # 平面分子推理、峰检测和构图
│   ├── test_ABNet_transfer.py    # 非平面分子推理、峰检测和构图
│   ├── test_ABNet_base_MgPsim.py # 模拟 MgP 推理
│   ├── test_ABNet_base_MgPexp.py # 实验 MgP 推理
│   ├── train_ANet_base.py        # 平面 ANet 训练
│   ├── train_BNet_base.py        # 平面 BNet 训练
│   ├── train_ANet_transfer.py    # ANet 迁移学习
│   ├── train_BNet_transfer.py    # BNet 迁移学习
│   └── c/                        # 模板匹配的 C++/Python 绑定
├── docs/                         # 论文初稿和补充信息
├── requirements.txt              # 锁定的 Python 依赖版本
└── README.md                     # 原始项目说明
```

### 2.1 `examples/` 中的关键文件

每个样例目录以 `ters_map.pt` 作为模型输入。运行推理后，脚本在同一目录写入或覆盖下列结果：

| 文件 | 含义 |
| --- | --- |
| `ters_map.tif` | 160 个输入 TERS 图的拼图，按每幅图自身范围显示 |
| `ters_map_same_scale.tif` | 同一输入，但按统一 `0–1` 色标显示 |
| `atom_map.tif` | ANet 输出的元素分辨概率图合并图 |
| `bond_map.tif` | BNet 输出的键/官能团概率图合并图 |
| `atom_coord.txt` | 峰检测后的原子坐标与类别 |
| `bond_coord.txt` | 峰检测后的键/官能团中心坐标与类别 |
| `atom_coord.svg` | 原子位置的矢量图 |
| `bond_coord.svg` | 由键/官能团预测推断的连接示意图 |
| `geometry.sdf` | 后处理选出的图结构，以 V3000 SDF 保存 |
| `args.pkl` | 本次推理使用的命令行参数 |

平面 `4-*` 和 MgP 推理入口没有启用 `--geom`，因此其重点输出是 atom/bond 概率图；不会自动写出上述坐标、SVG 与 SDF 几何文件。

## 3. 环境配置与快速开始

### 3.1 依赖

建议使用 Python 3.8，并按仓库锁定版本安装依赖。核心版本为 PyTorch `1.12.0`、NumPy `1.21.5`、SciPy `1.7.3`、scikit-image `0.19.2`、Matplotlib `3.5.1`、Pandas `1.4.2`、scikit-learn `1.0.2` 和 pybind11 `2.10.1`。训练或批量推理推荐配置与 PyTorch `1.12.0` 匹配的 CUDA；给定推理脚本本身不显式调用 CUDA，因此应先在目标环境试运行一个小样例。

在 Linux、WSL 或 Git Bash 中执行：

```bash
cd <repo>
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果你的 PyTorch 安装需要专用 CUDA 或 CPU wheel，请先按照目标平台的 PyTorch 安装方式安装 `torch==1.12.0`、`torchvision==0.13.0`、`torchaudio==0.12.0`，再执行其余依赖安装。

### 3.2 最小可用测试

先进入 `scripts/` 再运行。这样脚本中的 `../src`、`../examples` 和 `../checkpoints` 才会正确解析到仓库内的目录：

```bash
cd <repo>/scripts
bash test_ABNet_plane_mole.sh
```

完成后检查 `../examples/1_plane_mole/` 中时间戳最新的 `atom_map.tif`、`bond_map.tif`、`atom_coord.txt`、`bond_coord.txt` 与 `geometry.sdf`。这就是最小的端到端复现：TERS tensor → ANet/BNet → 概率图 → 峰位置 → 分子图。

在 Windows PowerShell 中且已安装 Git Bash/WSL 的 `bash` 时，可使用：

```powershell
Push-Location '<repo>\scripts'
bash .\test_ABNet_plane_mole.sh
Pop-Location
```

不要在仓库根目录直接运行 `bash scripts/test_ABNet_plane_mole.sh`：脚本内部相对路径会向根目录的上一级解析，从而找不到 `src` 和 `checkpoints`。

## 4. 核心工作流与数据复现指南

### Step 0：确认输入与模型配置

1. 选择任务类型：平面分子使用 `ANet_base` + `BNet_base`；带甲基/乙基等受控非平面案例使用两个 `*_transfer` 权重；MgP 使用 base 权重和专用 MgP 脚本。
2. 确认 `ters_map.pt` 是 PyTorch 可加载的 tensor，且具有批次维度并与训练输入的 160 个光谱通道、`128 × 128` 空间网格兼容。
3. 不要混用 BNet 权重与输出通道数：base BNet 必须配置 `--out_channels_b 9`，transfer BNet 必须配置 `--out_channels_b 11`。ANet 均为 4 通道。
4. 为自己的数据新建一个空输出目录，而非覆盖论文样例。例如将输入复制/链接到 `examples/my_sample/ters_map.pt`，并在命令中把 `--savedir` 改为该目录。

### Step 1：复现平面分子结果（Fig. 1 / Fig. 3 / Fig. 4）

执行：

```bash
cd <repo>/scripts
bash test_ABNet_plane_mole.sh
```

脚本依次处理 `1_plane_mole`、`3-1` 至 `3-4`、`4-1` 至 `4-3`。其中：

- `1_plane_mole` 对应 Fig. 1 所用的 2,6-dihydroxypyridine 示例，并可与 Supplementary Fig. 3 的完整输出链比较；
- `3-1` 至 `3-4` 对应 Fig. 3 的短链、六元杂环、稠合杂环和较大支化分子；通过 `geometry.sdf` 与 `atom_coord.svg`/`bond_coord.svg` 复核重建结果；
- `4-1` 至 `4-3` 是 Fig. 4 的不同分辨率条件。它们用于比较原子/键概率图的清晰度和骨架保持情况，而脚本没有请求自动构图。

若只想运行一个平面样例，可在 `<repo>/scripts` 中执行：

```bash
python ../src/test_ABNet_base.py \
  --savedir ../examples/3-1_plane_mole/ \
  --input_file ../examples/3-1_plane_mole/ters_map.pt \
  --out_channels_a 4 --x_channels_a 64 64 96 --q_channels_a 64 64 64 \
  --attention_type_a softmax --model_dir_a ../checkpoints/ANet_base/model_100.pth \
  --out_channels_b 9 --x_channels_b 64 64 96 --q_channels_b 64 64 64 \
  --attention_type_b sigmoid --model_dir_b ../checkpoints/BNet_base/model_100.pth \
  --geom
```

`--geom` 是触发峰检测、连接推断和 SDF 写出的开关。没有该参数时，程序仍会生成 TERS、atom 与 bond 的 TIFF 图，但不生成坐标和几何文件。

### Step 2：复现非平面迁移学习结果（Fig. 5）

执行：

```bash
cd <repo>/scripts
bash test_ABNet_nonplane_mole.sh
```

该脚本对 `5-1` 至 `5-4` 四个案例运行 `test_ABNet_transfer.py`，均启用 `--geom`。重点检查：

1. `atom_map.tif` 是否仍能分辨 C/N/O/H；
2. `bond_map.tif` 是否包含 11 通道体系下的键/官能团特征；
3. `geometry.sdf` 与 `bond_coord.svg` 是否给出合理的主骨架和取代基连接。

论文指出，对更强三维化的体系，TERS 的取向选择定则和局域场限制会使部分振动模式采样不完整；因此对超出迁移学习分布的三维分子，不应把输出理解为完整且唯一的真结构。Supplementary Fig. 7 展示了这一适用边界。

### Step 3：复现 MgP 模拟与实验结果（Fig. 6）

执行：

```bash
cd <repo>/scripts
bash test_ABNet_plane_MgP.sh
```

该入口分别调用模拟 MgP 和实验 MgP 的专用脚本。对比 `6-1_MgP_sim/` 与 `6-2_MgP_exp/` 中的 `atom_map.tif`、`bond_map.tif`，即可对应 Fig. 6a-c 和 Fig. 6d-f。

实验 MgP 的原始 mapping 为 `25 × 25` 像素、覆盖同样的 `2.5 × 2.5 nm²` 区域；论文中先去噪并线性插值到网络所需的 `128 × 128`。模型可恢复部分氢、外围碳和宏环特征，但因空间失配、噪声、离域化，以及 Mg 未被定义为输出原子类别，不能期待完整的原子级 SDF 重建。这个案例应以概率图中的化学约束为主要结果，而不是用它评估完整结构恢复率。

### Step 4：训练或重新训练

训练入口为：

```bash
cd <repo>
bash scripts/train_ABNet.sh
```

该脚本依次训练平面 ANet/BNet，再加载平面权重进行 ANet/BNet 迁移学习。配置为 batch size `32`、100 epochs、Adam、初始学习率 `3e-4`、每 10 个 epoch 衰减为原来的 `0.92`；训练损失为按通道样本数加权的 MSE。

**重要：当前仓库未包含训练脚本引用的完整 `.npy` 训练数据和索引文件**（如 `QM_FG_CNO05A_H05_08A_mapping_data.npy`、`indices_QM5678_FG5678_x4.pth`）。因此，发布版本可以直接复现推理样例，但不能在不补齐这些文件的情况下从头训练。获得相同文件后，应将其放在训练命令所引用的位置，或显式修改 `--data_file`、`--targets_file`、`--indices_file`（迁移学习还包括 `*file1` 与 `*file2`）的路径。

### Step 5：核验结果

对带 `--geom` 的案例，建议按以下顺序检查：

1. 打开 `ters_map_same_scale.tif`，确认输入的 160 个通道没有全黑、全白、尺寸错位或方向异常；
2. 打开 `atom_map.tif` 和 `bond_map.tif`，确认预测峰/结构没有明显整体平移或镜像；
3. 检查 `atom_coord.txt` 和 `bond_coord.txt` 是否非空，并与两个 SVG 中的位置一致；
4. 用支持 V3000 SDF 的分子可视化工具打开 `geometry.sdf`，检查键连接和价态是否合理；
5. 与对应的论文图及 SI 中的 ground truth 比较时，区分“网络概率图”“峰检测结果”和“最终构图结果”：它们是三个不同的处理层级。

## 5. 常见问题与注意事项

### 脚本报找不到 `../src/...` 或权重文件

原因通常是工作目录不对。请先 `cd <repo>/scripts` 后再运行 Bash 脚本，或将脚本中所有相对路径改为相对于仓库根目录的路径。

### `ModuleNotFoundError`、PyTorch 版本不兼容或无法加载 `.pth`

使用 `requirements.txt` 锁定的依赖组合，并让 `torch`、`torchvision` 与 `torchaudio` 的版本相互匹配。若因 CUDA 环境不同而先安装了平台专用 PyTorch，请保持版本与仓库指定版本一致。推理模型构造参数也必须保持脚本中给出的 `x_channels`、`q_channels`、attention 类型和输出通道数，否则 `load_state_dict` 会因权重形状不匹配而失败。

### 输出被覆盖

所有样例会将结果写回其 `--savedir`。为保留基准结果或比较不同参数，请为新任务建立独立目录，并显式传入新的 `--savedir`。

### 只得到 TIFF，没有 `geometry.sdf`

检查命令是否带有 `--geom`。平面 `4-*` 和 MgP 的预置命令故意不启用该参数；这与这些案例的论文展示目标一致。

### 模型对实验数据或低分辨率数据出现额外原子、模糊键或不完整结构

这不仅是程序错误的信号，也可能反映输入超出训练分布。SI 中的 zero-shot 分辨率测试显示：训练基准约为 `0.87 nm` 局域场约束；在更低约束、约 `1.12 nm` 的条件下，原子/键图会扩散并可能出现额外预测。可考虑先进行去噪、尺度/分辨率校正，或以目标仪器条件的数据做迁移学习；同时保留概率图和坐标结果，而不要只依赖单一 SDF 文件。

### 需要处理新的实验 TERS mapping

先使预处理后的数据与训练表示对齐：相同空间范围、`128 × 128` 网格、160 个 20 cm⁻¹ 窗口通道和相近的归一化/方向约定。然后从一个单样例推理开始，人工检查输入与概率图；验证通过后再批量运行。对于含有未训练元素、强离域的大共轭体系或明显三维构型，输出应被视为结构候选与化学约束，而非无条件的唯一答案。

## 6. 推荐阅读顺序

1. 先阅读主论文 Fig. 1 与 Supplementary Fig. 3，理解 “TERS 输入 → 两个概率图 → 峰 → 图结构” 的完整链路；
2. 运行 `1_plane_mole`，逐一打开其 TIFF、TXT、SVG 和 SDF；
3. 运行 Fig. 3 与 Fig. 5 的批量样例，观察平面与非平面模型的通道和后处理差异；
4. 最后运行 MgP，并结合论文 Fig. 6 理解实验域差异和当前限制；
5. 只有在取得完整训练数据后，再运行 `train_ABNet.sh` 进行重训或领域迁移。

## 7. 许可与引用

代码采用 MIT License；预训练权重与基准数据采用 CC-BY-NC 4.0。使用项目时请引用 README 指向的论文：<https://arxiv.org/abs/2603.21579>。
