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
#SBATCH --gres=gpu:1                            # reserving 1 GPU
#SBATCH --cpus-per-task=2                      # reserving 2 CPUs per task (and associated memory)

module purge                                    # purging modules inherited by default
module load python/3.8.8
module load cudnn/8.0.4.30-cuda-10.2


set -x                                          # activating echo of launched commands

srun python eval_with_reprieve_cifar10.py --mtm_path $SCRATCH/results/sweeps/shared_ref_mnist/2021-08-24/19-16-00/eta_ae:0.81-eta_lsa:0.14-eta_msa:0.95-eta_dsa:0.01-sigma:0.67-nagents:3-/params/step_49999/ --dti_path $SCRATCH/results/sweeps/shared_ref_mnist/2021-08-24/19-16-00/eta_ae:0.0-eta_lsa:0.0-eta_msa:1.0-eta_dsa:0.0-sigma:0.67-nagents:3-/params/step_49999/ --bsize 32 --points 4