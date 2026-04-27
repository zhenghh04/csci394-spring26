#!/bin/sh
export LOCAL_RANK=$PMIX_LOCAL_RANK
export RANK=$PMIX_RANK
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
#export SIZE=$((PALS_LOCAL_SIZE*PBS_JOBSIZE))
export SIZE=$PMI_SIZE
export LOCAL_RANK=$PALS_LOCAL_RANKID
export RANK=$PALS_RANKID
export WORLD_SIZE=$SIZE
if [ -z "${WORLD_SIZE}" ]; then
    export WORLD_SIZE=1
fi
if [ -z "${RANK}" ]; then
    export RANK=0
fi

echo "I am $RANK of $SIZE: $LOCAL_RANK on `hostname`"

export MASTER_ADDR=$(head -1 $PBS_NODEFILE)
export PYTHONNOUSERSITE=1
export MASTER_PORT=8892
$@
