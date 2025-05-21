import os
import tempfile
from jrun._base import JobDB
from jrun.interfaces import JobRecord, JobSpec


class JobSubmitter(JobDB):
    """Submit jobs to a SLURM cluster with support for dependencies and job sequences."""

    def __init__(self, db_path: str = "~/.cache/jobrunner/jobs.db"):
        super().__init__(db_path)

    def _parse_job_id(self, result: str) -> str:
        job_id = result.split(" ")[-1].strip()
        if job_id is None:
            raise ValueError("Failed to parse job ID from sbatch output.")
        return job_id

    def _submit_jobspec(self, job_spec: JobSpec) -> str:
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
            result = os.popen(f"sbatch {script_path}").read()
            job_id = self._parse_job_id(result)
            print(f"Submitted job with ID {job_id}")
            self.add_record(JobRecord(**job_spec.to_dict(), job_id=job_id))
            return job_id
        finally:
            # Clean up the temporary file
            os.unlink(script_path)

    def submit(self, file: str, dry: bool = False):
        pass

    def sbatch(self, sbatch_args: list):
        # Call sbatch with the provided arguments
        result = os.popen(f"sbatch {' '.join(sbatch_args)}").read()
        job_id = self._parse_job_id(result)
        print(f"Submitted job with ID {job_id}")
        self.add_record(
            JobRecord(
                job_id=job_id,
                group="sbatch",
                command=" ".join(sbatch_args),
                preamble="",
            )
        )
