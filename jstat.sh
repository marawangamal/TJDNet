#!/bin/bash

# Script to get GPU memory usage for a SLURM job ID
# Usage: ./gpu_mem_usage.sh <jobid>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <jobid>"
    echo "Example: $0 12345"
    exit 1
fi

JOBID=$1

# Check if job exists and is running
JOB_STATE=$(squeue -j $JOBID -h -o "%T" 2>/dev/null)
if [ -z "$JOB_STATE" ]; then
    echo "Error: Job $JOBID not found or not running"
    exit 1
fi

if [ "$JOB_STATE" != "RUNNING" ]; then
    echo "Error: Job $JOBID is not in RUNNING state (current state: $JOB_STATE)"
    exit 1
fi

# Get the node(s) where the job is running
NODES=$(squeue -j $JOBID -h -o "%N")
if [ -z "$NODES" ]; then
    echo "Error: Could not determine nodes for job $JOBID"
    exit 1
fi

echo "Job ID: $JOBID"
echo "Node(s): $NODES"
echo "Job State: $JOB_STATE"
echo ""
echo "GPU Memory Usage:"
echo "=================="

# Expand node list if it's in compressed format (e.g., node[01-03])
EXPANDED_NODES=$(scontrol show hostnames $NODES)

for NODE in $EXPANDED_NODES; do
    echo ""
    echo "Node: $NODE"
    echo "-------------------"
    
    # Use srun to execute nvidia-smi on the specific node for this job
    # This ensures we're checking the GPUs allocated to this specific job
    srun --jobid=$JOBID --nodelist=$NODE --pty nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=', ' read -r index name mem_used mem_total gpu_util; do
        # Calculate memory percentage
        if [ "$mem_total" -gt 0 ]; then
            mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "N/A")
        else
            mem_percent="N/A"
        fi
        
        printf "GPU %s (%s):\n" "$index" "$name"
        printf "  Memory: %s MB / %s MB (%.1f%%)\n" "$mem_used" "$mem_total" "$mem_percent"
        printf "  GPU Utilization: %s%%\n" "$gpu_util"
        echo ""
    done
    
    # If nvidia-smi fails, try alternative approach
    if [ $? -ne 0 ]; then
        echo "  Could not retrieve GPU info (nvidia-smi failed or no GPUs allocated)"
    fi
done

echo ""
echo "Summary:"
echo "========"
# Get total GPU memory usage across all nodes
TOTAL_GPU_MEM=$(srun --jobid=$JOBID nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | \
    awk '{sum += $1} END {print sum}')

if [ ! -z "$TOTAL_GPU_MEM" ] && [ "$TOTAL_GPU_MEM" -gt 0 ]; then
    echo "Total GPU memory used by job $JOBID: ${TOTAL_GPU_MEM} MB"
else
    echo "Could not determine total GPU memory usage"
fi