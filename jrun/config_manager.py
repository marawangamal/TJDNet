# jrun/config_manager.py
import os
import yaml
import uuid
import itertools
from typing import Dict, List, Any, Optional

from .job_spec import JobSpec, JobGroup, JobSequence, JobSweep


class ConfigManager:
    """Manage complex SLURM job configurations from YAML files with dependencies."""

    def __init__(self, config_path: str):
        """Initialize the configuration manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.preambles = self._process_preambles()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _process_preambles(self) -> Dict[str, List[str]]:
        """Process preamble templates."""
        return self.config.get("preambles", {})

    def get_job_groups(self) -> List[JobGroup]:
        """Build job groups from configuration.

        Returns:
            List of JobGroup objects
        """
        groups = []

        for group_config in self.config.get("groups", []):
            group_name = group_config.get("name", f"group_{len(groups)}")

            # Process the sequential or parallel jobs within this group
            job_sequences = self._process_sequential_jobs(
                group_config.get("sequentialjobs", []), group_name
            )

            groups.append(JobGroup(name=group_name, job_sequences=job_sequences))

        return groups

    def _process_sequential_jobs(
        self,
        sequential_config: List[Dict],
        group_name: str,
        parent_exp_id: Optional[str] = None,
    ) -> List[JobSequence]:
        """Process sequential jobs configuration.

        Args:
            sequential_config: List of sequential job configurations
            group_name: Name of the parent group
            parent_exp_id: Optional parent experiment ID for variable substitution

        Returns:
            List of JobSequence objects
        """
        # Generate a sequence experiment ID if not provided
        if parent_exp_id is None:
            parent_exp_id = str(uuid.uuid4())[:8]

        # Create one JobSequence to hold all simple "job" entries
        simple_jobs = []

        # Create a list to hold all job sequences (both simple and complex)
        all_sequences = []

        for seq_idx, seq_item in enumerate(sequential_config):
            # Generate a unique ID for this item
            item_id = f"{parent_exp_id}-{seq_idx}"

            if "job" in seq_item:
                # Simple job - add to the collection of simple jobs
                job_spec = self._create_job_spec(seq_item["job"], group_name, item_id)
                simple_jobs.append(job_spec)

            elif "paralleljobs" in seq_item:
                # If we have accumulated simple jobs, create a sequence for them
                if simple_jobs:
                    simple_seq_id = f"{parent_exp_id}-simple-{len(all_sequences)}"
                    all_sequences.append(
                        JobSequence(
                            id=simple_seq_id,
                            job_specs=list(simple_jobs),
                            is_parallel=False,
                        )
                    )
                    simple_jobs = []  # Reset the list

                # Parallel jobs - create a parallel sequence
                parallel_sequences = self._process_parallel_jobs(
                    seq_item["paralleljobs"], group_name, item_id
                )
                all_sequences.append(
                    JobSequence(
                        id=item_id,
                        sub_sequences=parallel_sequences,
                        is_parallel=True,
                    )
                )

            elif "parallellsweep" in seq_item:
                # If we have accumulated simple jobs, create a sequence for them
                if simple_jobs:
                    simple_seq_id = f"{parent_exp_id}-simple-{len(all_sequences)}"
                    all_sequences.append(
                        JobSequence(
                            id=simple_seq_id,
                            job_specs=list(simple_jobs),
                            is_parallel=False,
                        )
                    )
                    simple_jobs = []  # Reset the list

                # Parameter sweep - create a sweep sequence
                sweep_config = seq_item["parallellsweep"]
                sweep_sequences = self._process_parameter_sweep(
                    sweep_config, group_name, item_id
                )
                all_sequences.append(
                    JobSequence(
                        id=item_id,
                        sub_sequences=sweep_sequences,
                        is_parallel=True,
                        is_sweep=True,
                    )
                )

        # If we have any remaining simple jobs, create a sequence for them
        if simple_jobs:
            simple_seq_id = f"{parent_exp_id}-simple-{len(all_sequences)}"
            all_sequences.append(
                JobSequence(
                    id=simple_seq_id, job_specs=list(simple_jobs), is_parallel=False
                )
            )

        return all_sequences

    def _process_parallel_jobs(
        self, parallel_config: List[Dict], group_name: str, parent_exp_id: str
    ) -> List[JobSequence]:
        """Process parallel jobs configuration.

        Args:
            parallel_config: List of parallel job configurations
            group_name: Name of the parent group
            parent_exp_id: Parent experiment ID for variable substitution

        Returns:
            List of JobSequence objects (one for each parallel branch)
        """
        parallel_sequences = []

        for p_idx, p_item in enumerate(parallel_config):
            # Generate a unique ID for this parallel branch
            branch_id = f"{parent_exp_id}-{p_idx}"

            if "job" in p_item:
                # Single job
                job_specs = [
                    self._create_job_spec(p_item["job"], group_name, branch_id)
                ]
                parallel_sequences.append(
                    JobSequence(id=branch_id, job_specs=job_specs, is_parallel=False)
                )

            elif "sequentialjobs" in p_item:
                # Sequential jobs within a parallel branch
                sub_sequences = self._process_sequential_jobs(
                    p_item["sequentialjobs"], group_name, branch_id
                )
                parallel_sequences.append(
                    JobSequence(
                        id=branch_id, sub_sequences=sub_sequences, is_parallel=False
                    )
                )

        return parallel_sequences

    def _process_parameter_sweep(
        self, sweep_config: Dict, group_name: str, parent_exp_id: str
    ) -> List[JobSequence]:
        """Process parameter sweep configuration.

        Args:
            sweep_config: Parameter sweep configuration
            group_name: Name of the parent group
            parent_exp_id: Parent experiment ID

        Returns:
            List of JobSequence objects (one for each parameter combination)
        """
        # Extract sweep parameters and their values
        sweep_params = sweep_config.get("sweep", [])
        param_names = []
        param_values = []

        for param in sweep_params:
            for name, config in param.items():
                param_names.append(name)
                param_values.append(config.get("values", []))

        # Generate all combinations of parameter values
        param_combinations = list(itertools.product(*param_values))

        # Create a sequence for each parameter combination
        sweep_sequences = []

        for combo_idx, combo in enumerate(param_combinations):
            # Create parameter dictionary for this combination
            params = {param_names[i]: combo[i] for i in range(len(param_names))}

            # Generate a unique ID for this combination
            combo_id = f"{parent_exp_id}-{combo_idx}"

            # Process the sequential jobs for this parameter combination
            sub_sequences = []

            if "sequentialjobs" in sweep_config:
                seq_jobs = sweep_config["sequentialjobs"]

                for seq_idx, seq_item in enumerate(seq_jobs):
                    job_id = f"{combo_id}-{seq_idx}"

                    if "job" in seq_item:
                        # Apply parameter substitution to job command
                        job_config = seq_item["job"].copy()
                        job_config["params"] = params

                        job_specs = [
                            self._create_job_spec(job_config, group_name, job_id)
                        ]

                        sub_sequences.append(
                            JobSequence(
                                id=job_id,
                                job_specs=job_specs,
                                is_parallel=False,
                                params=params,
                            )
                        )

            # Create a job sequence for this parameter combination
            sweep_sequences.append(
                JobSequence(
                    id=combo_id,
                    sub_sequences=sub_sequences,
                    is_parallel=False,
                    params=params,
                )
            )

        # Create a job sweep to represent the entire parameter sweep
        return sweep_sequences

    def _create_job_spec(
        self,
        job_config: Dict,
        group_name: str,
        exp_id: str,
        params: Optional[Dict] = None,
    ) -> JobSpec:
        """Create a JobSpec from job configuration.

        Args:
            job_config: Job configuration
            group_name: Name of the parent group
            exp_id: Experiment ID for variable substitution
            params: Optional parameter dictionary for variable substitution

        Returns:
            JobSpec object
        """
        # Get preamble
        preamble_name = job_config.get("preamble", "base")
        preamble_lines = self.preambles.get(preamble_name, [])

        # Get base preamble if it exists and isn't the current preamble
        if preamble_name != "base" and "base" in self.preambles:
            preamble_lines = self.preambles["base"] + preamble_lines

        # Process command with variable substitution
        command = job_config.get("command", "")

        # Apply variable substitution
        variables = {
            "EXP_ID": exp_id,
            "GROUP_NAME": group_name,
        }

        # Add parameters if provided
        if params:
            variables.update(params)

        # Apply substitutions
        for var_name, var_value in variables.items():
            command = command.replace(f"{{{{{var_name}}}}}", str(var_value))

        # Create the job spec
        return JobSpec(
            name=f"{group_name}_{exp_id}",
            command=command,
            preamble="\n".join(preamble_lines),
            group_name=group_name,
            exp_id=exp_id,
        )
