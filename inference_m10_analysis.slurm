#!/bin/bash
#SBATCH --job-name=m10_inference_analysis         # Job name
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                    # Request 32GB of memory
#SBATCH --output=inference_m10_analysis.log        # Standard output file
#SBATCH --error=inference_error_m10_analysis.txt   # Standard error file
#SBATCH --time=00:20:00              # Max job time
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
pip install sentencepiece

# Navigate to the model directory
cd M10_T5_no_pretrain_word_level

# Run the training script
python analysis.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --config_name=../t5-base-model \
    --do_test \
    --test_data_file=../data/fine_tune_data/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1
