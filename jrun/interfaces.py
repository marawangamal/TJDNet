from dataclasses import asdict, dataclass, field
from typing import List, Dict, Any

###############################################################################
# Core data structures                                                        #
###############################################################################


@dataclass
class JobRecord:
    """Specification for a SLURM job."""

    job_id: str
    command: str
    preamble: str
    group: str = ""
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass instance to a dictionary."""
        return asdict(self)


@dataclass
class JobSpec(JobRecord):
    job_id: str = ""

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
