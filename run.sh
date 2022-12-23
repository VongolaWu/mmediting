CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh \
configs/nafnet/nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_adobex8.py \
4 \
--resume
