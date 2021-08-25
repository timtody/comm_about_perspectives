#!/bin/bash
#SBATCH --job-name=reprieve              # name of job
#SBATCH -C v100-32g                             # reserving 16 GB GPUs only
                                                # (use "-C v100-32g" for 32 GB GPUs only)
#SBATCH -A imi@gpu
#SBATCH --output=out/%j                         # output file (%j = job ID)
#SBATCH --error=err/%j                          # error file (%j = job ID)
#SBATCH --time=20:00:00                         # maximum allocation time "(HH:MM:SS)"
#SBATCH --nodes=1                               # reserving 1 node
#SBATCH --ntasks=1                              # reserving 9 tasks (or MPI processes)
#SBATCH --gres=gpu:1                            # reserving 4 GPUs
#SBATCH --cpus-per-task=2                      # reserving 40 CPUs per task (and associated memory)

module purge                                    # purging modules inherited by default
module load python/3.8.8


set -x                                          # activating echo of launched commands

srun python eval_with_reprieve_cifar10.py