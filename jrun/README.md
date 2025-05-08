# Enhanced SLURM JobRunner

A powerful tool for defining, submitting, and tracking complex SLURM job workflows.

## Overview

The Enhanced SLURM JobRunner allows you to define complex workflows with sequential and parallel jobs, parameter sweeps, and dependencies between jobs. It provides a clean interface for submitting jobs to a SLURM cluster and tracking their status.

## Features

- **Complex Job Hierarchies**
  - Sequential job chains (jobs that depend on each other)
  - Parallel job execution
  - Nested job structures (sequential within parallel and vice versa)
  - Parameter sweeps with automatic job generation

- **Template System**
  - Reusable preamble templates
  - Automatic inheritance from base templates

- **Variable Substitution**
  - Experiment IDs automatically generated and propagated
  - Group name propagation
  - Parameter value substitution
  - Dynamic variable replacement with `{{VARIABLE}}` syntax

- **Dependency Management**
  - Automatic tracking of job dependencies
  - SLURM dependency flags automatically added
  - Visualization of dependency graphs

- **Status Tracking**
  - SQLite database for efficient storage and querying
  - Advanced filtering and reporting capabilities
  - Status summaries and visualizations

## Installation

```
pip install slurm-jobrunner
```

## Quick Start

1. Define your job workflow in a YAML file:

```yaml
# Define reusable preamble templates
preambles:
  # Common base preamble
  base:
    - "#!/bin/bash"
    - "#SBATCH --partition=short-unkillable"
    - "#SBATCH --output=slurm/slurm-%j.out"
    - "#SBATCH --error=slurm/slurm-%j.err"
  
  # Training job preamble
  train:
    - "#SBATCH --gres=gpu:a100l:4"
    - "#SBATCH --cpus-per-task=12"
    - "#SBATCH --mem=128G"
    - "#SBATCH --time=3:00:00"

groups:
  - name: example-workflow
    sequentialjobs:
      - paralleljobs:
          - job:
              preamble: train
              command: python train.py --model model1 --jobrunner_id {{EXP_ID}} --tag {{GROUP_NAME}}
          - job:
              preamble: train
              command: python train.py --model model2 --jobrunner_id {{EXP_ID}} --tag {{GROUP_NAME}}
      - job:
          preamble: base
          command: python aggregate_results.py --jobrunner_id {{EXP_ID}}
```

2. Submit jobs to SLURM:

```
jrun submit --config workflow.yaml
```

3. Check job status:

```
jrun status --group example-workflow
```

## Usage

### Submit Jobs

```
# Submit all jobs from configuration
jrun submit --config workflow.yaml

# Submit only jobs from a specific group
jrun submit --config workflow.yaml --group example-workflow

# Submit a single job
jrun submit --command "python train.py" --preamble "#!/bin/bash" --preamble "#SBATCH --gres=gpu:1"
```

### Check Status

```
# Check status of all jobs
jrun status

# Filter by group
jrun status --group example-workflow

# Filter by experiment ID
jrun status --exp-id abc123

# Filter by status
jrun status --status RUNNING

# Sort results
jrun status --sort-by created_at
```

### Cancel Jobs

```
# Cancel specific jobs
jrun cancel job_id1 job_id2

# Cancel all jobs in a group
jrun cancel --group example-workflow

# Cancel all jobs with experiment ID
jrun cancel --exp-id abc123
```

### Resubmit Jobs

```
# Resubmit specific jobs
jrun resubmit job_id1 job_id2

# Resubmit all failed jobs in a group
jrun resubmit --group example-workflow --failed-only

# Resubmit all jobs with experiment ID
jrun resubmit --exp-id abc123
```

### Visualize Dependencies

```
# Generate dependency graph for a group
jrun visualize --group example-workflow --output deps.png

# Generate dependency graph for an experiment
jrun visualize --exp-id abc123 --output deps.png
```

## Configuration Format

The configuration file uses YAML format and supports:

- **Preamble Templates**: Reusable SLURM directives
- **Groups**: Collections of related jobs
- **Sequential Jobs**: Jobs that run in sequence
- **Parallel Jobs**: Jobs that run in parallel
- **Parameter Sweeps**: Automatically generate jobs for combinations of parameters
- **Variable Substitution**: Replace variables in commands

## License

MIT License