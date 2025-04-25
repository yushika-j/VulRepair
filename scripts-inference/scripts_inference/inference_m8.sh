#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-04:10:00
#SBATCH --account=def-pariya
#SBATCH --mail-user=onaim017@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/olena/scratch/logoutput_test_m8_%A_%a.out
#SBATCH -e /home/olena/scratch/logerror_test_m8_%A_%a.err

# Load modules
module load StdEnv/2023
module load python/3.12.4
module load gcc/13.3
module --ignore_cache load cuda/12.2
module load arrow/19.0.1
module load git-lfs/2.11.0

# Create virtual environment and activate
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install packages
pip install --no-index -r $SCRATCH/VulRepair/requirements.txt || exit 1
pip install torch transformers numpy tqdm pandas tokenizers datasets gdown tensorboard scikit-learn --no-index
pip list

# Navigate to model directory
cd $SCRATCH/VulRepair/M8_VRepair_subword || exit

# Run test
python inter8.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=../codet5-base \
    --config_name=../codet5-base \
    --do_test \
    --test_data_file=../data/fine_tune_data/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1

# Confirm test output
JSON_OUTPUT=./saved_models/test_intermediate_outputs_${SLURM_JOB_ID}.json
if [ -f "$JSON_OUTPUT" ]; then
    echo "✅ Test outputs found: $JSON_OUTPUT"
    head -n 20 "$JSON_OUTPUT"
else
    echo "❌ ERROR: Test output file not found! Exiting..." >&2
    exit 1
fi

echo "🎉 SLURM job for M8 test completed successfully."
