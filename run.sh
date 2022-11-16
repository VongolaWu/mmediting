#!/usr/bin/env bash
# test
# GPUS=4 GPUS_PER_NODE=4 ./tools/slurm_test.sh \
# mm_lol \
# airnet_o \
# configs/airnet/airnet_cfg_allsets_air.py \
# work_dirs/airnet/epoch_1500.pth

# all in one
# GPUS=4 GPUS_PER_NODE=4 ./tools/slurm_train.sh \
# mm_lol \
# airnet_o \
# configs/airnet/airnet_cfg_allsets_air.py \
# work_dirs/airnet \
# --resume

# derain
GPUS=2 GPUS_PER_NODE=2 ./tools/slurm_train.sh \
mm_lol \
air_rain \
configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1000E_rain.py \
# --resume

# dehaze
# GPUS=2 GPUS_PER_NODE=2 ./tools/slurm_train.sh \
# mm_lol \
# air_haze \
# configs/airnet/airnet_cbde256drgn55643_2xb8_lr1e-3_1000E_sots.py \
# --resume
