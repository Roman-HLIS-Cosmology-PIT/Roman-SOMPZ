#!/bin/bash -l
#SBATCH --account=des
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --output=JOB/JOB_OUT_check_%j.txt
#SBATCH --error=JOB/JOB_ERR_check_%j.txt
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END

set -x

export HDF5_USE_FILE_LOCKING='FALSE'

module load python 
conda activate sompz

srun python -u selection_nz_final_rushift_zpshift_0_25.py
