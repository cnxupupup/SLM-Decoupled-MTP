NNODE=1
NPROC_PER_NODE=8

torchrun \
--nnode $NNODE \
--nproc_per_node $NPROC_PER_NODE \
--master_port 29501 \
src/train/train_mtp.py \
--cfg_file config/mtp_demo.yaml