#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=24:00:00 
#SBATCH --job-name=cyberner_train
#SBATCH --output=/home/yasir.ech-chammakhy/lustre/cyber_cc-lcbfvhtc9qm/users/yasir.ech-chammakhy/logs_gpu/cyberner_train_%j.log
#SBATCH --account=CYBER_CC-LCBFVHTC9QM-DEFAULT-GPU
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules - using exactly the same approach as the working script
module purge
module load CUDA/11.7.0
module load Anaconda3

# Environment setup - using the same approach as the working script
source ~/.bashrc
conda activate yasir

# Make sure conda libs are prioritized over system libs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Define the NER script path for the unified model
NER_SCRIPT="/home/yasir.ech-chammakhy/lustre/cyber_cc-lcbfvhtc9qm/users/yasir.ech-chammakhy/CyberNER/scripts/models_cyber.py"

# Define base directory for outputs
BASE_DIR="/home/yasir.ech-chammakhy/lustre/cyber_cc-lcbfvhtc9qm/users/yasir.ech-chammakhy/CyberNER"
mkdir -p "${BASE_DIR}/logs_cyberner"
mkdir -p "${BASE_DIR}/outputs_cyberner"

# Define models using the same naming convention as the working script
MODELS=("bert-base-cased" "roberta-base" "securebert" "darkbert" "cysecbert")

# Set common training parameters
BATCH_SIZE=16
EPOCHS=50
MAX_SEQ_LENGTH=256
LEARNING_RATE=5e-5
LR_CRF_FC=8e-5
WEIGHT_DECAY_FINETUNE=1e-5
WEIGHT_DECAY_CRF_FC=5e-6
WARMUP_PROPORTION=0.1
EARLY_STOPPING_PATIENCE=5
TEST_SIZE=0.15
VAL_SIZE=0.15

# Create a unique timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting Unified CyberNER Training Jobs at ${TIMESTAMP}"

# Process each model
for model in "${MODELS[@]}"; do
    # Create a descriptive name for this run
    RUN_NAME="CyberNER_${model//\//_}_${TIMESTAMP}"
    LOG_FILE="${BASE_DIR}/logs_cyberner/${RUN_NAME}.log"
    
    echo "Running: Model=${model}"
    echo "Log file: ${LOG_FILE}"
    
    # Run NER training with the unified model script
    python ${NER_SCRIPT} \
        --model_type "${model}" \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --learning_rate ${LEARNING_RATE} \
        --lr_crf_fc ${LR_CRF_FC} \
        --weight_decay_finetune ${WEIGHT_DECAY_FINETUNE} \
        --weight_decay_crf_fc ${WEIGHT_DECAY_CRF_FC} \
        --warmup_proportion ${WARMUP_PROPORTION} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        --test_size ${TEST_SIZE} \
        --val_size ${VAL_SIZE} \
        --output_dir "${BASE_DIR}/outputs_cyberner" \
        --dataset_path "${BASE_DIR}/dataset/cyberner_combined_stix.csv" \
        > "${LOG_FILE}" 2>&1
    
    # Report status
    if [ $? -eq 0 ]; then
        echo "✓ Completed: Model=${model}"
    else
        echo "✗ Failed: Model=${model}. Check log: ${LOG_FILE}"
    fi
done

echo "All Unified CyberNER Training Jobs completed."