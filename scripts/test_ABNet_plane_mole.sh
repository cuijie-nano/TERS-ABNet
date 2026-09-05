#!/bin/bash 

python ../src/test_ABNet_base.py \
    --savedir 				../examples/1_plane_mole/ \
    --input_file 			../examples/1_plane_mole/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_base/model_100.pth		\
    --out_channels_b 		9 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_base/model_100.pth  \
	--geom					  
    
python ../src/test_ABNet_base.py \
    --savedir 				../examples/3-1_plane_mole/ \
    --input_file 			../examples/3-1_plane_mole/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_base/model_100.pth		\
    --out_channels_b 		9 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_base/model_100.pth  \
	--geom					  
    
python ../src/test_ABNet_base.py \
    --savedir 				../examples/3-2_plane_mole/ \
    --input_file 			../examples/3-2_plane_mole/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_base/model_100.pth		\
    --out_channels_b 		9 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_base/model_100.pth  \
	--geom					  
    
python ../src/test_ABNet_base.py \
    --savedir 				../examples/3-3_plane_mole/ \
    --input_file 			../examples/3-3_plane_mole/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_base/model_100.pth		\
    --out_channels_b 		9 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_base/model_100.pth  \
	--geom					  
    
python ../src/test_ABNet_base.py \
    --savedir 				../examples/3-4_plane_mole/ \
    --input_file 			../examples/3-4_plane_mole/ters_map.pt \
    --out_channels_a 		4 \
    --x_channels_a 			64  64  96  \
    --q_channels_a 			64  64  64  \
    --attention_type_a   	softmax  \
    --model_dir_a   	 	../checkpoints/ANet_base/model_100.pth		\
    --out_channels_b 		9 \
    --x_channels_b 		 	64  64  96 \
    --q_channels_b  		64  64  64 \
    --attention_type_b      sigmoid \
    --model_dir_b 			../checkpoints/BNet_base/model_100.pth  \
	--geom					  
    
python ../src/test_ABNet_base.py \
    --savedir 				../examples/4-1_plane_mole/ \
    --input_file 			../examples/4-1_plane_mole/ters_map.pt \
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
    
python ../src/test_ABNet_base.py \
    --savedir 				../examples/4-2_plane_mole/ \
    --input_file 			../examples/4-2_plane_mole/ters_map.pt \
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
    
python ../src/test_ABNet_base.py \
    --savedir 				../examples/4-3_plane_mole/ \
    --input_file 			../examples/4-3_plane_mole/ters_map.pt \
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
    