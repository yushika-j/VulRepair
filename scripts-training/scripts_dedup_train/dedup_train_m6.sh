#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-10:10:00
#SBATCH --account=def-pariya
#SBATCH --mail-user=onaim017@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/olena/scratch/logoutput_m6_%A_%a.out
#SBATCH -e /home/olena/scratch/logerror_m6_%A_%a.err

# Load modules
module load StdEnv/2023 python/3.11 mpi4py scipy-stack/2023b cuda cudnn gcc/13.3 arrow/19.0.1 git-lfs/3.4.0

# Create and activate virtual environment
VENV_DIR=$SLURM_TMPDIR/env
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

# Install dependencies (offline)
pip install torch transformers numpy tqdm pandas tokenizers datasets gdown tensorboard scikit-learn --no-index
pip list

# Navigate to working directory
cd $SCRATCH/VulRepair

# Ensure RoBERTa model is available locally
MODEL_DIR=$SCRATCH/VulRepair/roberta-base
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ ERROR: RoBERTa model directory not found!"
    exit 1
fi
cd $MODEL_DIR
git lfs pull

# Check model weights
if [ ! -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "❌ ERROR: Model weights are missing!"
    exit 1
fi

# Move into M6 model directory
cd $SCRATCH/VulRepair/M6_BERT_base_NL || exit 1

# Run training
python roberta_base_main.py \
    --output_dir=./saved_models \
    --model_name=modelD.bin \
    --tokenizer_name=$MODEL_DIR \
    --model_name_or_path=$MODEL_DIR \
    --do_train \
    --train_data_file=../VulRepair/cleaned_train.csv \
    --eval_data_file=../VulRepair/cleaned_val.csv\
    --test_data_file=../VulRepair/cleaned_test.csv\
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
