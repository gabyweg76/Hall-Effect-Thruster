#!/bin/bash
#SBATCH --nodes 2              # Request 2 nodes so all resources are in two nodes.
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”. You will get 2 per node.

#SBATCH --tasks-per-node=2   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.

#SBATCH --mem=8G      
#SBATCH --time=0-00:30
#SBATCH --account=def-soulaima
#SBATCH --output=%N.%j.out

module load StdEnv/2020
module load python/3.7
module load scipy-stack
module load hdf5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install torch --no-index
pip install torchvision --no-index

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"


echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python Softmax_DDP.py --init_method tcp://$MASTER_ADDR:8103 --world_size $SLURM_NTASKS --batch_size 128
