GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=16 ./tools/slurm_test.sh \
aide_lol \
nafnet \
configs/nafnet/nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro_diy.py \
checkpoint/NAFNet-GoPro-midc64.pth
