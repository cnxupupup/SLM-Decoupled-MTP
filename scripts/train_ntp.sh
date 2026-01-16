NNODE=1
NPROC_PER_NODE=8

torchrun \
--nnode $NNODE \
--nproc_per_node $NPROC_PER_NODE \
--master_port 29501 \
src/train/train_ntp.py \
--cfg_file config/ntp_demo.yaml