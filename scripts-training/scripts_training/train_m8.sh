#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-10:10:00
#SBATCH --account=def-pariya
#SBATCH --mail-user=onaim017@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/olena/scratch/logoutput_%A_%a.out
#SBATCH -e /home/olena/scratch/logerror_%A_%a.err

module load StdEnv/2023 python/3.11 mpi4py scipy-stack/2023b cuda cudnn
module load StdEnv/2023

module load python/3.12.4
module load gcc/13.3
module --ignore_cache load cuda/12.2
module load arrow/19.0.1
module load git-lfs/2.11.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
cd $SCRATCH/VulRepair

# Create the virtual environment on each node:
srun --ntasks=$SLURM_NNODES --tasks-per-node=1 bash << EOF
if ! virtualenv --no-download $SLURM_TMPDIR/env; then
    echo "Failed to create virtual environment on node $SLURM_NODEID" >&2
    exit 1
fi
source $SLURM_TMPDIR/env/bin/activate

if ! pip install --no-index --upgrade pip; then
    echo "Failed to upgrade pip on node $SLURM_NODEID" >&2
    exit 1
fi

if ! pip install --no-index -r requirements.txt; then
    echo "Failed to install requirements on node $SLURM_NODEID" >&2
    exit 1
fi
EOF

# Activate the environment on the main node
source $SLURM_TMPDIR/env/bin/activate

# Install additional packages
pip install torch --no-index
pip install transformers --no-index
pip install numpy --no-index
pip install tqdm --no-index
pip install pandas --no-index
pip install tokenizers --no-index
pip install datasets --no-index
pip install gdown --no-index
pip install tensorboard --no-index
pip install scikit-learn --no-index
pip list

# Navigate to the correct directory
cd M8_VRepair_subword/transformers || exit
cd ../.. || exit
cd M8_VRepair_subword || exit

# Run training script
python vrepair_subword_main.py \
    --model_name=model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=../codet5-base \
    --config_name=../codet5-base \
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

