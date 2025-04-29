#!/bin/bash 
#SBATCH --job-name=t5_training         # Job name
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=32G                      # Request 16GB of memory
#SBATCH --output=train5.log      # Standard output file
#SBATCH --error=train5.txt        # Standard error file
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=pjhun035@uottawa.ca
#SBATCH --mail-type=ALL

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

cd M5_BERT_no_pretrain_subword
python roberta_no_pretraining_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=./tokenizer/ \
    --config_name=../roberta-base \
    --do_train \
    --train_data_file=../data/fine_tune_data/train.csv \
    --eval_data_file=../data/fine_tune_data/val.csv \
    --test_data_file=../data/fine_tune_data/test.csv \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
