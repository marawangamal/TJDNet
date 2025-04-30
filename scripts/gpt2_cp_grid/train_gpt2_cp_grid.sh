#!/bin/bash
# gpt2-cp-umps-grid.sh
# This script runs the gpt2 model with both cp and umps heads with a grid search over {rank, horizon}
# Different learning rates are used for each head type: lr=1e-3 for cp and lr=1e-4 for umps

# Define the ranks and horizons to search over
RANKS=(1 2 4 8 16)
HORIZONS=(2 4 8)
MODEL_HEADS=("cp" "umps")

# Run all configurations
for model_head in "${MODEL_HEADS[@]}"; do
    # Set the learning rate based on model head type
    if [ "$model_head" == "cp" ]; then
        lr="1e-3"
    else  # umps
        lr="5e-4"
    fi
    
    echo "Starting grid search for model_head=$model_head with lr=$lr"
    echo "========================================================"
    
    for rank in "${RANKS[@]}"; do
        for horizon in "${HORIZONS[@]}"; do
            echo "Running configuration: model_head=$model_head, rank=$rank, horizon=$horizon, lr=$lr"
            
            python train.py \
                --epochs 20 \
                --horizon "$horizon" \
                --horizon_eval "$horizon" \
                --rank "$rank" \
                --model_head "$model_head" \
                --lr "$lr" \
                --disable_wandb \
                --use_memory_efficient_loss \
                --compute_acc 
            
            echo "Completed: model_head=$model_head, rank=$rank, horizon=$horizon, lr=$lr"
            echo "------------------------"
        done
    done
    
    echo "Completed grid search for model_head=$model_head"
    echo "========================================================"
done

echo "Full grid search completed."