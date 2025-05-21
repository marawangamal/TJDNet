import os
import re
import shlex
import subprocess
import tempfile
from jrun._base import JobDB
from jrun.interfaces import JobSpec

JOB_RE = re.compile(r"Submitted batch job (\d+)")


class JobSubmitter(JobDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_job_id(self, result: str) -> int:
        # job_id = result.split(" ")[-1].strip()
        # if not job_id:
        #     raise ValueError("Failed to parse job ID from sbatch output.")
        # return job_id
        # Typical line: "Submitted batch job 123456"
        m = JOB_RE.search(result)
        if m:
            jobid = int(m.group(1))
            print(f"→ jobid = {jobid}")
            return jobid
        else:
            raise RuntimeError(f"Could not parse job id from sbatch output:\n{result}")

    def _submit_jobspec(self, job_spec: JobSpec) -> int:
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
            self.add_record(JobSpec(**job_spec.to_dict(), job_id=job_id))
            return job_id
        finally:
            # Clean up the temporary file
            os.unlink(script_path)

    def submit(self, file: str, dry: bool = False):
        pass

    def sbatch(self, args: list):
        # Call sbatch with the provided arguments
        result = subprocess.run(
            ["sbatch"] + args, check=True, capture_output=True, text=True
        ).stdout.strip()
        job_id = self._parse_job_id(result)
        self.add_record(
            JobSpec(
                job_id=job_id,
                group_name="sbatch",
                command=" ".join(args),
                preamble="",
                depends_on=[],
            )
        )
