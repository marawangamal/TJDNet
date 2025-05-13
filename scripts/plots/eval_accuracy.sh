#!/bin/bash

# eval.sh
# Evaluate accuracy for all experiments in the experiments directory

# Set experiments directory as a constant
EXPERIMENTS_DIR="checkpoints"

# Check if experiments directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Error: experiments directory not found"
    exit 1
fi

# Find all experiment directories
experiments=$(find $EXPERIMENTS_DIR -maxdepth 1 -type d | grep -v "^$EXPERIMENTS_DIR$")

# If no experiments found
if [ -z "$experiments" ]; then
    echo "No experiments found in the $EXPERIMENTS_DIR directory"
    exit 0
fi

echo "Found experiments:"
for exp in $experiments; do
    echo "  - $(basename $exp)"
done
echo ""

# Run evaluation for each experiment
echo "Starting evaluation..."
for exp in $experiments; do
    exp_name=$(basename $exp)
    echo "Evaluating experiment: $exp_name"
    python scripts/eval_accuracy.py -e $EXPERIMENTS_DIR/$exp_name --max_num_samples 100
    
    # Check if the evaluation was successful
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed for $exp_name"
    else
        echo "✗ Evaluation failed for $exp_name"
    fi
    echo "----------------------------------"
done

echo "Evaluation complete for all experiments"