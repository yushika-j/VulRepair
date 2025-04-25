#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-00:10:00
#SBATCH --account=def-pariya
#SBATCH --mail-user=onaim017@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/olena/scratch/logoutput_%A_%a.out
#SBATCH -e /home/olena/scratch/logerror_%A_%a.err

# Load necessary modules
module load StdEnv/2023 python/3.11 mpi4py scipy-stack/2023b cuda cudnn gcc/13.3 arrow/19.0.1 git-lfs/3.4.0

# Create and activate virtual environment
VENV_DIR=$SLURM_TMPDIR/env
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

# Install dependencies from requirements.txt
if [ -f requirements.txt ]; then
    if ! pip install --no-index -r requirements.txt; then
        echo "Failed to install requirements from requirements.txt" >&2
        exit 1
    fi
fi

# Verify Git LFS is installed
git lfs install

# Navigate to working directory
cd $SCRATCH/VulRepair

MODEL_DIR=$SCRATCH/VulRepair/roberta-base
if [ ! -d "$MODEL_DIR" ]; then
    echo "Cloning RoBERTa-base model..."
    git clone https://huggingface.co/roberta-base $MODEL_DIR
    cd $MODEL_DIR
    git lfs pull
fi


# Check if model files exist
if [ ! -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "ERROR: Model files are missing! Exiting..."
    exit 1
fi

# Navigate to the correct directory for training
cd M9_CodeBERT_word_level|| exit 1

# Run training script
python codebert_wordlevel_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --config_name=$MODEL_DIR \
    --do_train \
    --train_data_file=../data/fine_tune_data/train.csv \
    --eval_data_file=../data/fine_tune_data/val.csv \
    --test_data_file=../data/fine_tune_data/test.csv \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
