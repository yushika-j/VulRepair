#!/bin/bash
#SBATCH --job-name=t3_training_deduped         # Job name
#SBATCH --gres=gpu:4                   # Request 1 GPU
#SBATCH --mem=50G                      # Request 16GB of memory
#SBATCH --output=train3_deduped_gh.log      # Standard output file
#SBATCH --error=train3_deduped_gh.txt        # Standard error file
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=pjhun035@uottawa.ca

module load StdEnv/2023
module load python/3.12.4
module load gcc/13.3
module --ignore_cache load cuda/12.2
module load arrow/19.0.1
module load git-lfs/2.11.0
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
pip list
nvidia-smi
# Navigate to the appropriate directory
cd M3_T5_no_pretrain_subword

# Set PyTorch CUDA configuration to handle GPU memory better
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the training command
python t5_no_pretraining_main.py \
    --model_name=model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=./tokenizer/ \
    --config_name=../t5-base \
    --do_train \
    --train_data_file=../data/fine_tune_data/deduped/VulRepair/cleaned_train.csv \
    --eval_data_file=../data/fine_tune_data/deduped/VulRepair/cleaned_test.csv \
    --test_data_file=../data/fine_tune_data/deduped/VulRepair/cleaned_val.csv \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee train.log
