# jrun/job_submitter.py
import os
import tempfile
import time
from typing import List, Dict

from .job_spec import JobSpec, JobSequence, JobGroup
from .job_tracker import JobTracker


class JobSubmitter:
    """Submit jobs to a SLURM cluster with support for dependencies and job sequences."""

    def __init__(self, tracker: JobTracker):
        """Initialize the job submitter.

        Args:
            tracker: JobTracker instance for recording job information
        """
        self.tracker = tracker

    def submit_job(self, job_spec: JobSpec) -> str:
        """Submit a single job to SLURM and return the job ID.

        Args:
            job_spec: The job specification to submit

        Returns:
            The job ID as a string
        """
        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            script_path = f.name
            f.write(job_spec.to_script())

        try:
            # Submit the job
            result = os.popen(f"sbatch {script_path}").read()

            # Parse the job ID
            try:
                job_id = result.split(" ")[-1].strip()
                print(f"Submitted job {job_spec.name} with ID {job_id}")

                # Record the job submission
                self.tracker.record_job(job_id, job_spec)

                return job_id
            except Exception as e:
                print(f"Error parsing job ID from output: {result}")
                print(f"Error details: {e}")
                return "ERROR"
        finally:
            # Clean up the temporary file
            os.unlink(script_path)

    def submit_sequence(self, sequence: JobSequence) -> Dict[str, str]:
        """Submit a sequence of jobs with dependencies.

        Args:
            sequence: JobSequence to submit

        Returns:
            Dictionary mapping job names to job IDs
        """
        job_ids = {}

        if sequence.is_parallel:
            # Submit sub-sequences in parallel
            for sub_seq in sequence.sub_sequences:
                sub_ids = self.submit_sequence(sub_seq)
                job_ids.update(sub_ids)
        else:
            # Submit jobs in sequence, with dependencies
            prev_job_id = None

            # First, submit all direct job specs
            for job_spec in sequence.job_specs:
                if prev_job_id:
                    job_spec.depends_on.append(prev_job_id)

                job_id = self.submit_job(job_spec)
                job_ids[job_spec.name] = job_id
                prev_job_id = job_id

            # Then, submit sub-sequences with dependency on the last job
            last_id = prev_job_id
            for sub_seq in sequence.sub_sequences:
                if last_id:
                    # Add dependency to first job in sub-sequence
                    if sub_seq.job_specs:
                        sub_seq.job_specs[0].depends_on.append(last_id)

                sub_ids = self.submit_sequence(sub_seq)
                job_ids.update(sub_ids)

                # Find the last job ID in this sub-sequence
                if sub_ids:
                    last_sub_id = list(sub_ids.values())[-1]
                    prev_job_id = last_sub_id

        return job_ids

    def submit_group(self, group: JobGroup) -> Dict[str, str]:
        """Submit a group of job sequences.

        Args:
            group: JobGroup to submit

        Returns:
            Dictionary mapping job names to job IDs
        """
        job_ids = {}

        for sequence in group.job_sequences:
            seq_ids = self.submit_sequence(sequence)
            job_ids.update(seq_ids)

        return job_ids

    def wait_for_jobs(self, job_ids: List[str], check_interval: int = 30) -> None:
        """Wait for a list of jobs to complete.

        Args:
            job_ids: List of job IDs to wait for
            check_interval: Time in seconds between status checks
        """
        while job_ids:
            # Update job statuses
            self.tracker.update_status()

            # Check which jobs are still running
            running_jobs = []
            for job_id in job_ids:
                status = self.tracker.get_job_status(job_id)
                if status in ["PENDING", "RUNNING", "REQUEUED"]:
                    running_jobs.append(job_id)

            # Update the list of jobs to wait for
            job_ids = running_jobs

            if job_ids:
                # Some jobs are still running
                jobs_str = ", ".join(job_ids)
                print(f"Waiting for {len(job_ids)} jobs to complete: {jobs_str}")
                time.sleep(check_interval)
            else:
                print("All jobs have completed.")
