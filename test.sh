# Submit a job that will fail after a few seconds
job1=$(sbatch --wrap="echo 'Job 1 starting'; sleep 10; echo 'Job 1 failing'; exit 1" | grep -o '[0-9]*')
echo "Job 1 ID: $job1"

# Submit a job that depends on it  
job2=$(sbatch --dependency=afterok:$job1 --wrap="echo 'Job 2 started after job 1'" | grep -o '[0-9]*')
echo "Job 2 ID: $job2"

# # Check the queue - job 2 should show as pending with reason "Dependency"
squeue -j $job1,$job2