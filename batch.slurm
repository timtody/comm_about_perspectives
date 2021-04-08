#!/bin/bash
#SBATCH --job-name=hypsearch-tt-01              # name of job
#SBATCH -C v100-16g                             # reserving 16 GB GPUs only
                                                # (use "-C v100-32g" for 32 GB GPUs only)
#SBATCH -A imi@gpu
#SBATCH --qos=qos_gpu-dev                       # QoS
#SBATCH --output=hypsearch-tt-01%j.out          # output file (%j = job ID)
#SBATCH --error=hypsearch-tt-01%j.err           # error file (%j = job ID)
#SBATCH --time=00:10:00                         # maximum allocation time "(HH:MM:SS)"
#SBATCH --nodes=1                               # reserving 1 node
#SBATCH --ntasks=9                              # reserving 4 tasks (or MPI processes)
#SBATCH --gres=gpu:4                            # reserving 4 GPUs
#SBATCH --cpus-per-task=4                       # reserving 10 CPUs per task (and associated memory)

module purge                                    # purging modules inherited by default
conda deactivate                                # deactivating environments inherited by default
module load python/3.8.2
module load pytorch-gpu/py3/1.8.0               # loading modules


set -x                                          # activating echo of launched commands

srun python -u main.py --grid_size 3 --nprocs 4 