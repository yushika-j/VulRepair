#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-04:10:00
#SBATCH --account=def-pariya
#SBATCH --mail-user=onaim017@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/olena/scratch/logoutput_m9_%A_%a.out
#SBATCH -e /home/olena/scratch/logerror_m9_%A_%a.err

# Load modules
module load StdEnv/2023 python/3.11 mpi4py scipy-stack/2023b cuda cudnn gcc/13.3 arrow/19.0.1 git-lfs/3.4.0

# Create and activate virtual environment
VENV_DIR=$SLURM_TMPDIR/env
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

# Install dependencies (offline mode)
if [ -f requirements.txt ]; then
    if ! pip install --no-index -r requirements.txt; then
        echo "❌ Failed to install requirements." >&2
        exit 1
    fi
fi

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

# Enable Git LFS
git lfs install

# Navigate to working directory
cd $SCRATCH/VulRepair

# Load CodeBERT model locally
MODEL_DIR=/home/olena/scratch/VulRepair/codebert-base
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ ERROR: CodeBERT model directory not found!" >&2
    exit 1
fi

cd $MODEL_DIR
git lfs pull

if [ ! -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "❌ ERROR: Model weights missing! Exiting..."
    exit 1
fi

# Go to M9 model test directory
cd $SCRATCH/VulRepair/M9_CodeBERT_word_level || exit 1

# Run inference
python inter9.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --config_name=$MODEL_DIR \
    --tokenizer_name=$MODEL_DIR \
    --do_test \
    --test_data_file=../data/fine_tune_data/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1

# Check for test output
JSON_OUTPUT=./saved_models/test_intermediate_outputs_${SLURM_JOB_ID}.json
if [ -f "$JSON_OUTPUT" ]; then
    echo "✅ Test outputs found: $JSON_OUTPUT"
    head -n 20 "$JSON_OUTPUT"
else
    echo "❌ ERROR: Test output file not found! Exiting..." >&2
    exit 1
fi

echo "🎉 SLURM job for M9 test completed successfully."
