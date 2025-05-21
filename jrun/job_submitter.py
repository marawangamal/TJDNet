import itertools
import os
import random
import re
import shlex
import subprocess
import tempfile
from typing import Any, Dict

import yaml
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
        """Parse the YAML file and submit jobs."""
        cfg = yaml.safe_load(open(file))

        preamble_map = {
            name: "\n".join(lines) for name, lines in cfg["preambles"].items()
        }
        root = cfg["group"]
        # recursively walk the tree
        self.walk(node=cfg["group"], preamble_map=preamble_map, dry=dry)

    def walk(
        self,
        node: Dict[str, Any],
        preamble_map: Dict[str, str],
        dry: bool = False,
        depends_on: list[int] = [],
        group_name: str = "",
    ):
        """Recursively walk the job tree and submit jobs."""

        # Base case (single leaf)
        if hasattr(node, "command"):
            # Leaf node
            # generate rand job id int
            job_id = random.randint(100000, 999999)
            job = JobSpec(
                job_id=job_id,
                group_name=node.get("name", group_name),
                command=node["command"],
                preamble=preamble_map.get(node["preamble"], ""),
                depends_on=[str(_id) for _id in depends_on],
            )
            if dry:
                print(f"DRY-RUN: {job.to_script()}")
                return [job_id]
            else:
                return [self._submit_jobspec(job)]

        # Base case (sweep)
        elif node["type"] == "sweep":
            cmd_template = node["command"]
            sweep = node["sweep"]
            # Generate all combinations of the sweep parameters
            keys = list(sweep.keys())
            values = list(sweep.values())
            # Generate all combinations of the sweep parameters
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            # Iterate over the combinations
            for i, params in enumerate(combinations):
                job_id = random.randint(100000, 999999)
                cmd = cmd_template.format(**params)
                job = JobSpec(
                    job_id=job_id,
                    group_name=node.get("name", group_name),
                    command=cmd,
                    preamble=preamble_map.get(node["preamble"], ""),
                    depends_on=[str(_id) for _id in depends_on],
                )
                if dry:
                    print(f"DRY-RUN: {job.to_script()}")
                    return [job_id]
                else:
                    return [self._submit_jobspec(job)]

        # Recursive case:
        elif node["type"] == "sequential":
            # Parallel group
            depends_on = []
            for entry in node["jobs"]:
                job_ids = self.walk(
                    entry, dry=dry, preamble_map=preamble_map, depends_on=depends_on
                )
                if job_ids:
                    depends_on.extend(job_ids)

        elif node["type"] == "parallel":
            # Parallel group
            for entry in node["jobs"]:
                self.walk(
                    entry, dry=dry, preamble_map=preamble_map, depends_on=depends_on
                )

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
