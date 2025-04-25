#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-04:10:00
#SBATCH --account=def-pariya
#SBATCH --mail-user=onaim017@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/olena/scratch/logoutput_%A_%a.out
#SBATCH -e /home/olena/scratch/logerror_%A_%a.err

# Load necessary modules
module load StdEnv/2023 python/3.11 mpi4py scipy-stack/2023b cuda cudnn gcc/13.3 arrow/19.0.1 git-lfs/3.4.0

# Activate virtual environment
VENV_DIR=$SLURM_TMPDIR/env
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

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

git lfs install

# Navigate to working directory
cd $SCRATCH/VulRepair

# Ensure RoBERTa model is properly cloned
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

# Navigate to correct directory
cd M6_BERT_base_NL || exit 1

# Run inference
python roberta_base_modified.py \
    --output_dir=./saved_models \
    --model_name=modelD.bin \
    --tokenizer_name=$MODEL_DIR \
    --model_name_or_path=$MODEL_DIR \
    --do_test \
    --test_data_file=../VulRepair/cleaned_test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1

# Check for JSON output using SLURM_JOB_ID
JSON_OUTPUT=./saved_models/test_intermediate_outputs_${SLURM_JOB_ID}.json

if [ -f "$JSON_OUTPUT" ]; then
    echo "Test outputs found: $JSON_OUTPUT"
    head -n 20 "$JSON_OUTPUT"
else
    echo "ERROR: Test output file not found! Exiting..." >&2
    exit 1
fi

echo "SLURM job completed successfully."

