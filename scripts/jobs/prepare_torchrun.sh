set -x
set -e

# Make sure env vars ENV_PREFIX and WORKDIR are set
if [ -z "$ENV_PREFIX" ]; then
    echo "ENV_PREFIX is not set. It must be set in job environment config while job launch"
    exit 1
fi
if [ -z "$WORKDIR" ]; then
    echo "WORKDIR is not set. It must be set in job environment config while job launch"
    exit 1
fi

echo "ENV_PREFIX=$ENV_PREFIX"
echo "WORKDIR=$WORKDIR"

# Add environment prefix to PATH
PATH=$ENV_PREFIX:$PATH

# Jobs run under MPI, extract master node info for DDP
MASTER_HOST_PREFIX=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$//; print \$x ")
MASTER_HOST=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$/-mpimaster-0/; print \$x ")

MASTER_HOST_FULL="$MASTER_HOST.$MASTER_HOST_PREFIX"
echo "MASTER_HOST_FULL $MASTER_HOST_FULL"

# Set DDP environment variables
export MASTER_ADDR=$MASTER_HOST_FULL
export MASTER_PORT=12345
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
export RANK=${OMPI_COMM_WORLD_RANK:-0}

NUM_GPUS=$(nvidia-smi -L | grep -c "GPU")

echo "NUM_GPUS $NUM_GPUS"
echo "WORLD_SIZE $WORLD_SIZE"
echo "RANK $RANK"
echo "MASTER_ADDR $MASTER_ADDR"

cd $WORKDIR

echo "ARGS: $@"

# Disable trace and errexit before launching to avoid spurious errors during cleanup
# The "set -e" causes NFS "Stale file handle" errors during shell cleanup to fail the job
# even though training completed successfully
set +x
set +e

exec ${ENV_PREFIX}/torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $@
