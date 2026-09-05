#!/bin/bash 

python src/train_ANet_base.py \
    --data_file QM_FG_CNO05A_H05_08A_mapping_data.npy \
    --targets_file QM_FG_CNO05A_H05_08A_atommapping_decay5.npy \
    --indices_file indices_QM5678_FG5678_x4.pth \
    --savedir ./checkpoints/ANet_base/ \
    --self_normalized \
    --out_channels 4 \
    --x_channels 64 64 96 \
    --q_channels 64 64 64 \
    --attention_type softmax \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 0.0003 \
    --step_size 10 \
    --gamma 0.92 \
    --noise 0.02 \
    --train_ratio 0.9

python src/train_BNet_base.py \
    --data_file QM_FG_CNO05A_H05_08A_mapping_data.npy \
    --targets_file QM_FG_CNO05A_H05_08A_bondmapping_decay5.npy \
    --indices_file indices_QM5678_FG5678_x4.pth \
    --savedir ./checkpoints/BNet_base/ \
    --self_normalized \
    --out_channels 9 \
    --x_channels 64 64 96 \
    --q_channels 64 64 64 \
    --attention_type sigmoid \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 0.0003 \
    --step_size 10 \
    --gamma 0.92 \
    --noise 0.02 \
    --train_ratio 0.9
	
	
python src/train_ANet_transfer.py \
    --data_file1 FG_CNO05A_09A_mapping_data.npy \
    --targets_file1 FG_CNO05A_H09A_atommapping_decay5.npy \
    --data_file2 QM_CNO05A_09A_mapping_data.npy \
    --targets_file2 QM_CNO05A_H09A_atommapping_decay5.npy \	
    --indices_file1 indices_FG9_x4.pth \
	--indices_file2 indices_QM9_x1.pth \
    --predir ./checkpoints/ANet_base/ \
    --savedir ./checkpoints/ANet_transfer/ \
    --self_normalized \
    --out_channels 4 \
    --x_channels 64 64 96 \
    --q_channels 64 64 64 \
    --attention_type softmax \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 0.0003 \
    --step_size 10 \
    --gamma 0.92 \
    --noise 0.02 \
    --train_ratio 0.9
	
python src/train_BNet_transfer.py \
    --data_file1 FG_CNO05A_09A_mapping_data.npy \
    --targets_file1 FG_CNO05A_H09A_bondmapping_decay5.npy \
    --data_file2 QM_CNO05A_09A_mapping_data.npy \
    --targets_file2 QM_CNO05A_H09A_bondmapping_decay5.npy \	
    --indices_file1 indices_FG9_x4.pth \
	--indices_file2 indices_QM9_x1.pth \
    --predir ./checkpoints/BNet_base/ \
    --savedir ./checkpoints/BNet_transfer/ \
    --self_normalized \
    --out_channels 11 \
    --x_channels 64 64 96 \
    --q_channels 64 64 64 \
    --attention_type sigmoid \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 0.0003 \
    --step_size 10 \
    --gamma 0.92 \
    --noise 0.02 \
    --train_ratio 0.9