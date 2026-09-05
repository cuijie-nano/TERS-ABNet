#!/bin/bash 

python ../src/test_ABNet_transfer.py \
    --savedir 				../examples/5-1_nonplane_mole/ \
    --input_file 			../examples/5-1_nonplane_mole/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_transfer/model_100.pth		\
    --out_channels_b 		11 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_transfer/model_100.pth  \
	--geom
    
python ../src/test_ABNet_transfer.py \
    --savedir 				../examples/5-2_nonplane_mole/ \
    --input_file 			../examples/5-2_nonplane_mole/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_transfer/model_100.pth		\
    --out_channels_b 		11 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_transfer/model_100.pth  \
	--geom
    
python ../src/test_ABNet_transfer.py \
    --savedir 				../examples/5-3_nonplane_mole/ \
    --input_file 			../examples/5-3_nonplane_mole/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_transfer/model_100.pth		\
    --out_channels_b 		11 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_transfer/model_100.pth  \
	--geom
    
python ../src/test_ABNet_transfer.py \
    --savedir 				../examples/5-4_nonplane_mole/ \
    --input_file 			../examples/5-4_nonplane_mole/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_transfer/model_100.pth		\
    --out_channels_b 		11 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_transfer/model_100.pth  \
	--geom
    
