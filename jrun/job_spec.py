# jrun/job_spec.py
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class JobSpec:
    """Specification for a SLURM job."""

    name: str
    command: str
    preamble: str
    group_name: str = ""
    exp_id: str = ""
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_script(self) -> str:
        """Convert job spec to a SLURM script.

        Returns:
            String containing the complete SLURM script
        """
        script_lines = [self.preamble]

        # Add dependency information if needed
        if self.depends_on:
            dependencies = ":".join(self.depends_on)
            script_lines.append(f"#SBATCH --dependency=afterok:{dependencies}")

        # Add the command
        script_lines.append(self.command)

        return "\n".join(script_lines)


@dataclass
class JobSequence:
    """A sequence of jobs that may run in parallel or sequentially."""

    id: str
    job_specs: List[JobSpec] = field(default_factory=list)
    sub_sequences: List["JobSequence"] = field(default_factory=list)
    is_parallel: bool = False
    is_sweep: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def get_all_job_specs(self) -> List[JobSpec]:
        """Get all job specs in this sequence, including those in sub-sequences.

        Returns:
            Flattened list of all JobSpec objects
        """
        specs = list(self.job_specs)

        for sub_seq in self.sub_sequences:
            specs.extend(sub_seq.get_all_job_specs())

        return specs


@dataclass
class JobSweep:
    """A parameter sweep that generates multiple job sequences."""

    name: str
    parameters: Dict[str, List[Any]]
    job_sequences: List[JobSequence] = field(default_factory=list)

    def get_all_job_specs(self) -> List[JobSpec]:
        """Get all job specs in this sweep.

        Returns:
            Flattened list of all JobSpec objects
        """
        specs = []

        for seq in self.job_sequences:
            specs.extend(seq.get_all_job_specs())

        return specs


@dataclass
class JobGroup:
    """A group of job sequences with a common name."""

    name: str
    job_sequences: List[JobSequence] = field(default_factory=list)

    def get_all_job_specs(self) -> List[JobSpec]:
        """Get all job specs in this group.

        Returns:
            Flattened list of all JobSpec objects
        """
        specs = []

        for seq in self.job_sequences:
            specs.extend(seq.get_all_job_specs())

        return specs
