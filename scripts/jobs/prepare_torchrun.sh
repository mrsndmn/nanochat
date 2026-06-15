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

# Point nanochat at the worktree's artifacts/ dir (prepared tokenizer, training data,
# checkpoints, eval bundle). nanochat.common resolves this to a symlink onto the absolute
# shared store (SHARED_ARTIFACTS_DIR), auto-creating it for new worktrees and resolving it
# inside worker containers. The MPI launch does not reliably forward NANOCHAT_BASE_DIR from
# the job config to the worker ranks, so derive it here from WORKDIR (which is forwarded).
# Any value that *was* forwarded takes precedence.
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$WORKDIR/artifacts}"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"

# Add environment prefix to PATH
PATH=$ENV_PREFIX:$PATH

# Jobs run under MPI, extract master node info for DDP
MASTER_HOST_PREFIX=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$//; print \$x ")
MASTER_HOST=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$/-mpimaster-0/; print \$x ")

MASTER_HOST_FULL="$MASTER_HOST.$MASTER_HOST_PREFIX"
echo "MASTER_HOST_FULL $MASTER_HOST_FULL"

# Job name (lm-mpi-job-<UUID>) for Arkhip's /status progress file. MASTER_HOST_PREFIX
# already strips the -mpimaster-N / -worker-N suffix off the MPI hostname, leaving the
# bare job name Arkhip keys on. Exported so JobProgress writes <job_name>.json into the
# metrics dir; absent (e.g. local runs) -> no progress file is written.
export ARKHIP_JOB_NAME=$MASTER_HOST_PREFIX
echo "ARKHIP_JOB_NAME=$ARKHIP_JOB_NAME"

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

# Invoke torchrun via the env's python instead of the `torchrun` console-script.
# The console-script has a hard-coded shebang pointing at the path where the env
# was originally created (e.g. /home/jovyan/.mlspace/envs/...), which does not
# exist on worker nodes that mount the env under a different path (/workspace-...).
# Running `python -m torch.distributed.run` uses the real interpreter at ENV_PREFIX
# and avoids the stale shebang entirely.
exec ${ENV_PREFIX}/python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $@
