#!/bin/bash
#SBATCH --job-name=m1_train                            # Job name
#SBATCH --gres=gpu:1                                   # Request 1 GPU
#SBATCH --mem=32G                                      # Request 32GB of memory
#SBATCH --output=dedup_results_m1/m1_train.log               # Standard output file
#SBATCH --error=dedup_results_m1/m1_train_error.txt          # Standard error file
#SBATCH --time=00:15:00                                # Max job time
#SBATCH --cpus-per-task=4                              # CPU cores per task
#SBATCH --mail-user=rosma012@uottawa.ca                # Email for job updates

module load StdEnv/2023 python/3.11 mpi4py scipy-stack/2023b cuda cudnn gcc/13.3 arrow/19.0.1 git-lfs/3.4.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install transformers --no-index
pip install torch --no-index
pip install numpy --no-index
pip install tqdm --no-index
pip install pandas --no-index
pip install tokenizers --no-index
pip install datasets --no-index
pip install gdown --no-index
pip install tensorboard --no-index
pip install scikit-learn --no-index

# Move to the directory
cd M1_VulRepair_PL-NL

python dedup_vulrepair_main.py \
    --model_name=dedup_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../codet5-base \
    --model_name_or_path=../codet5-base \
    --do_train \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log