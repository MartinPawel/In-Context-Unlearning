#!/usr/bin/env bash
#SBATCH -n 1                                      # Number of cores
#SBATCH -N 1                                      # Ensure that all cores are on one machine
#SBATCH -t 0-06:00                                # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu                                # Request gpu resources
#SBATCH -p gpu                                    # Partition: gpu
#SBATCH -v                                        # Make out more verbose 
#SBATCH --mem=20000                               # RAM requirement
#SBATCH -o myoutput_%j.out                        # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err                        # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=martin.pawelczyk.1@gmail.com  # info when job finishes

# Load required modules
module load python/3.10.9-fasrc01
module load cuda/11.3.1-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01

source activate unlearn

# Run program
python run.py --dataset_name "sst2" --dataset_size 25000 --config config_run_ubs1_bloom1b1.json