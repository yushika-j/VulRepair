#!/bin/bash
#SBATCH --job-name=m2_train         # Job name
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                    # Request 32GB of memory
#SBATCH --output=m2_train.log        # Standard output file
#SBATCH --error=m2_train_error.txt   # Standard error file
#SBATCH --time=11:00:00              # Max job time
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mail-user=rosma012@uottawa.ca  # Email for job updates

# Load required modules
module load StdEnv/2023
module load python/3.12.4
module load gcc/13.3
module --ignore_cache load cuda/12.2
module load arrow/19.0.1
module load git-lfs/2.11.0
module load scipy-stack  # Optional: Provides numpy, pandas, matplotlib, etc.

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install dependencies (only if not installed)
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

# Move to the Model 2 directory
cd M2_CodeBERT_PL-NL

# Run training
python codebert_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

