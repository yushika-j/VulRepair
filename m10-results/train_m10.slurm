#!/bin/bash
#SBATCH --job-name=m10_train         # Job name
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                    # Request 32GB of memory
#SBATCH --output=train_m10.log        # Standard output file
#SBATCH --error=train_error_m10.txt   # Standard error file
#SBATCH --time=12:00:00              # Max job time
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mail-user=rosma012@uottawa.ca  # Email for job updates

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


# Navigate to the model directory
cd M10_T5_no_pretrain_word_level

# Run the training script
python t5_no_pretraining_wordlevel_main.py \
    --model_name=model.bin \
    --output_dir=./saved_models \
    --config_name=../t5-base \
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

