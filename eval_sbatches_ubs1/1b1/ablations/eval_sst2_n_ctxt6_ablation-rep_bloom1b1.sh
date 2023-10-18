#!/usr/bin/env bash
#SBATCH -n 1                            # Number of cores
#SBATCH -N 1                            # Ensure that all cores are on one machine
#SBATCH -t 1-06:00                      # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu
#SBATCH -p gpu
#SBATCH -v                              # make outout more verbose 
#SBATCH --mem=20000
#SBATCH -o myoutput_%j.out              # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err              # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=martin.pawelczyk.1@gmail.com

# Load required modules
module load ncf/1.0.0-fasrc01
module load parallel/20230422-rocky8_x64-ncf
module load python/3.10.9-fasrc01
module load cuda/11.3.1-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01

conda activate unlearn

# Run program
python eval.py --dataset_name "sst2" --lfm "first-k" --batch_sizes 1 --n_ctxt 6 --ctxt_style "ablation-rep" --K_models 1 --model_path "bigscience/bloom-1b1" --rng_offset ${SLURM_ARRAY_TASK_ID} --config config_eval_rep.json

