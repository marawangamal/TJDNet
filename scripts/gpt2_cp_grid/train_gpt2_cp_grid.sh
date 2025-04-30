#!/bin/bash
# gpt2-grid.sh
# This script runs the gpt2 model with cp head with a grid search over {rank, horizon}

# Define the ranks and horizons to search over
RANKS=(1 2 4 8 16)
HORIZONS=(2 4 8)

# Run all configurations
for rank in "${RANKS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
        echo "Running configuration: rank=$rank, horizon=$horizon"
        
        python train.py \
            --epochs 20 \
            --horizon "$horizon" \
            --horizon_eval "$horizon" \
            --rank "$rank" \
            --disable_wandb \
            --use_memory_efficient_loss \
            --compute_acc 
        
        echo "Completed: rank=$rank, horizon=$horizon"
        echo "------------------------"
    done
done

echo "Grid search completed."