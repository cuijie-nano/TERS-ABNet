#!/bin/bash 

python ../src/test_ABNet_base_MgPsim.py \
    --savedir 				../examples/6-1_MgP_sim/ \
    --input_file 			../examples/6-1_MgP_sim/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_base/model_100.pth		\
    --out_channels_b 		9 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_base/model_100.pth
    


python ../src/test_ABNet_base_MgPexp.py \
    --savedir 				../examples/6-2_MgP_exp/ \
    --input_file 			../examples/6-2_MgP_exp/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_base/model_100.pth		\
    --out_channels_b 		9 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_base/model_100.pth
    
