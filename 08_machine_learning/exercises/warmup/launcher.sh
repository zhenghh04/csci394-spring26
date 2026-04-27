export PYTHONNOUSERSITE=1
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${master_port}'
export WORLD_SIZE='${ngpus}'
export RANK=${PMI_RANK}
export LOCAL_RANK=${PMI_LOCAL_RANK}   
