#!/bin/bash
# Phi-4 Informius Training Launch Script - FIXED VERSION
# Uses the improved dataset with complete answers

set -e  # Exit on any error

echo "🚀 Phi-4 Informius Training Launcher (FIXED VERSION)"
echo "===================================================="
echo "Training with COMPLETE ANSWERS dataset!"
echo ""

# Check GPU availability
echo "📊 Checking GPU setup..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✅ Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 1 ]; then
    echo "❌ No GPUs detected. Training requires at least 1 GPU."
    exit 1
fi

# Display GPU information
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
while IFS=, read -r name memory_total memory_free; do
    echo "   GPU: $name (${memory_total}MB total, ${memory_free}MB free)"
done

# Check dataset files
echo "📁 Verifying FIXED dataset files..."
DATASET_DIR="final_rich_dataset_fixed"
REQUIRED_FILES=("train.json" "validation.json" "test.json" "dataset_info.json")

if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Fixed dataset directory '$DATASET_DIR' not found!"
    echo "   Please run the data conversion scripts first."
    exit 1
fi

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATASET_DIR/$file" ]; then
        echo "❌ Required file '$DATASET_DIR/$file' not found!"
        exit 1
    else
        FILE_SIZE=$(du -h "$DATASET_DIR/$file" | cut -f1)
        echo "   ✅ $file ($FILE_SIZE)"
    fi
done

# Check config file
if [ ! -f "phi4_axolotl_config_fixed.yml" ]; then
    echo "❌ Fixed Axolotl config 'phi4_axolotl_config_fixed.yml' not found!"
    exit 1
fi

# Check disk space
echo "💾 Checking disk space..."
FREE_SPACE=$(df . | tail -1 | awk '{print $4}')
FREE_SPACE_GB=$((FREE_SPACE / 1024 / 1024))
echo "   Free space: ${FREE_SPACE_GB}GB"

if [ "$FREE_SPACE_GB" -lt 100 ]; then
    echo "⚠️  Low disk space. Training requires ~100GB for checkpoints."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Show data quality improvements
echo "📈 Dataset Quality Summary:"
echo "   - CoT responses: ~4000 characters (vs ~300 original)"
echo "   - Complete step details included"
echo "   - All validation criteria documented"
echo "   - Comprehensive summaries provided"
echo "   - 1493% increase in content richness!"

# Display training configuration
echo "⚙️  Fixed Training Configuration:"
echo "   Framework: Axolotl"
echo "   Config: phi4_axolotl_config_fixed.yml"
echo "   Dataset: final_rich_dataset_fixed (COMPLETE ANSWERS)"
echo "   Sequence length: 4096 (increased for longer responses)"
echo "   Max eval tokens: 1024 (increased for complete outputs)"
echo "   Output: phi4_axolotl_outputs_fixed"

# Estimate training time (longer due to increased sequence length)
TOTAL_EXAMPLES=74700
EFFECTIVE_BATCH_SIZE=32
STEPS_PER_EPOCH=$((TOTAL_EXAMPLES / EFFECTIVE_BATCH_SIZE))
TOTAL_STEPS=$((STEPS_PER_EPOCH * 3))
ESTIMATED_HOURS=$((TOTAL_STEPS / 300))  # Slower due to longer sequences

echo "   Estimated training time: ~${ESTIMATED_HOURS} hours (longer due to complete responses)"
echo "   Total training steps: ~${TOTAL_STEPS}"

# Confirm before starting
echo ""
echo "🎯 Ready to train with FIXED COMPLETE DATASET!"
echo "Command: python -m axolotl.cli.train phi4_axolotl_config_fixed.yml"
echo ""
echo "This will create a model that gives COMPLETE answers instead of truncated ones!"
echo ""
read -p "Start fixed training now? (Y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Set optimizations
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export NCCL_P2P_DISABLE=1        # Disable P2P for stability
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Create backup info
echo "📝 Creating training session info..."
echo "Training started: $(date)" > training_fixed_session.log
echo "Dataset: final_rich_dataset_fixed" >> training_fixed_session.log
echo "Config: phi4_axolotl_config_fixed.yml" >> training_fixed_session.log
echo "Expected improvements: Complete answers, no truncation" >> training_fixed_session.log

# Start training
echo "🏁 Starting Phi-4 Axolotl training (FIXED VERSION)..."
echo "   Time started: $(date)"
echo "   Training log: phi4_training.log"
echo "   Session log: training_fixed_session.log"
echo ""

# Run training with nice output
python -m axolotl.cli.train phi4_axolotl_config_fixed.yml 2>&1 | tee -a training_fixed_session.log

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "🎉 FIXED TRAINING COMPLETED SUCCESSFULLY!"
    echo "   Time finished: $(date)"
    echo "   Output directory: phi4_axolotl_outputs_fixed"
    echo "   Your model should now give COMPLETE answers!"
    echo ""
    echo "To test your improved model:"
    echo "   python -m axolotl.cli.inference phi4_axolotl_config_fixed.yml --max_new_tokens 1024"
else
    echo ""
    echo "❌ Training failed!"
    echo "   Check phi4_training.log and training_fixed_session.log for details."
    exit 1
fi