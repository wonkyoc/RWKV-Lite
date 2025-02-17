# https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf
# https://enterprise-support.nvidia.com/s/article/Useful-nvidia-smi-Queries-2

# goal: set env vars: 
# VRAM_MB, NGPUS, GPU0_NAME
#   GPUID is a known valid GPU, for detecting VRAM size

NGPUS=`nvidia-smi  --list-gpus |wc -l`; GPUID=0

VRAM_MB=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$GPUID`

GPU0_NAME=`nvidia-smi --query-gpu=name --format=csv,noheader,nounits --id=$GPUID`
