GPUS=8 CPUS_PER_TASK=16 ./tools/slurm_test.sh \
mm_lol \
mmp-rnn \
configs/mmp-rnn/mmp-rnn_adobe.py \
checkpoint/mmp-rnn.pth
