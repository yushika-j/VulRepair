#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-10:10:00
#SBATCH --account=def-pariya
#SBATCH --mail-user=onaim017@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/olena/scratch/logoutput_m8_%A_%a.out
#SBATCH -e /home/olena/scratch/logerror_m8_%A_%a.err

# Load necessary modules
module load StdEnv/2023 python/3.11 mpi4py scipy-stack/2023b cuda cudnn gcc/13.3 arrow/19.0.1 git-lfs/3.4.0

# Create and activate virtual environment
VENV_DIR=$SLURM_TMPDIR/env
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

# Install dependencies
pip install torch transformers numpy tqdm pandas tokenizers datasets gdown tensorboard scikit-learn --no-index
pip list

# Navigate to project root
cd $SCRATCH/VulRepair

# Optional: Pull tokenizer model from Hugging Face if needed (offline mode should be pre-fetched)
MODEL_DIR=$SCRATCH/VulRepair/codet5-base
if [ ! -d "$MODEL_DIR" ]; then
    echo "Cloning CodeT5-base tokenizer..."
    git clone https://huggingface.co/Salesforce/codet5-base $MODEL_DIR
    cd $MODEL_DIR
    git lfs pull
fi

# Navigate to M8 directory
cd M8_VRepair_subword || exit 1

# Run training script
python vrepair_subword_main.py \
    --model_name=modelD.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../codet5-base \
    --config_name=../codet5-base \
    --do_train \
    --train_data_file=../VulRepair/cleaned_train.csv \
    --eval_data_file=../VulRepair/cleaned_val.csv \
    --test_data_file=../VulRepair/cleaned_test.csv \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
