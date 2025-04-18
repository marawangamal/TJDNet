"""SLURM Job Manager for submitting and tracking multiple SLURM jobs.

This module allows you to define jobs in a YAML file and submit them in batches to a SLURM cluster.
It keeps track of job statuses and provides a way to check on all previously submitted jobs.

Usage:
    python jobrunner.py -f config.yaml           # Submit all jobs
    python jobrunner.py -f config.yaml -i        # Select jobs interactively
    python jobrunner.py -j "python train.py"     # Submit a single job

    python jobrunner.py -s                                           # Check status of all jobs
    python jobrunner.py -s --filter train.py --sort_by job_status    # Filter and sort jobs

Example YAML configuration:
    common_preamble_declarations:  # Common declarations for all jobs
        - "#!/bin/bash"
        - "#SBATCH --partition=short-unkillable"
        - "#SBATCH --output=slurm/slurm-%j.out"

    common_preamble_runs:  # Common commands for all jobs
        - "module load python/3.9"
        - "source /path/to/venv/bin/activate"

    groups:
        - name: "experiment1"
        preamble:
            - "#SBATCH --gres=gpu:a100l:4"
            - "#SBATCH --mem=128G"
        paralleljobs:
            - "python train.py --model_type llama --lr 1e-5"
            - "python train.py --model_type llama --lr 3e-5"


Example status table:
    job_id  job_status  command
    123456  RUNNING     python train.py --model_type llama --lr 1e-5
    123457  COMPLETED   python train.py --model_type llama --lr 3e-5
    123458  FAILED      python train.py --model_type llama --lr 5e-5

    UNKNOWN  SUBMIT  PENDING  RUNNING  COMPLETED  FAILED  CANCELLED  TIMEOUT
          0       0        0        1          1       1          0        0
"""

import argparse
import datetime
import os
import sys
import os.path as osp
import pickle
import time
from typing import Callable, Optional

import yaml
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 5000)
pd.set_option("display.max_colwidth", 10000)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def save_object(obj, filename):
    with open(filename, "wb") as out_file:  # Overwrites any existing file.
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, "rb") as inp_file:  # Overwrites any existing file.
        out = pickle.load(inp_file)
    return out


def get_job_statuses(
    job_ids: list, on_add_status: Optional[Callable[[str], str]] = None
):
    """Get the status of a list of job IDs."""
    statuses = []
    for job_id in [str(int(job_id)) for job_id in job_ids]:
        try:
            out = os.popen("sacct -j {} --format state".format(job_id)).read()
            status = out.split("\n")[2].strip()
            statuses.append(on_add_status(status) if on_add_status else status)
        except:
            statuses.append("UNKNOWN")
    return statuses


def get_status_table(
    cache_file: str = "~/.jobrunner/jobrunner_status_table.csv",
):
    """Get the status table from the cache file."""
    outpath = osp.join(osp.expanduser(cache_file))
    if not osp.exists(outpath):
        print("No status table found. Run `jobrunner.py -f <filepath>` first.")
        exit(1)
    status_table = pd.read_csv(outpath)

    def add_asterisk(job_id: str):
        if job_id in status_table["job_status"].values:
            return "{}*".format(job_id)
        return job_id

    job_ids = status_table["job_id"].tolist()
    statuses = get_job_statuses(job_ids, on_add_status=add_asterisk)
    status_table["job_status"] = statuses

    return status_table


class SlurmJobManager:

    def __init__(
        self,
        file,
        overwrite=False,
        interactive_mode=False,
        cache_file: str = "~/.jobrunner/jobrunner_status_table.csv",
        job: Optional[str] = None,  # single job
        preamble_path: Optional[str] = None,  # for single jobs.
    ):
        """A manager for submitting and tracking SLURM jobs defined in a YAML configuration file.

        Args:
            file (str): Path to the YAML configuration file.
            overwrite (bool): Flag to allow overwriting of previously submitted jobs.
        """
        self.cache_file = cache_file
        self.file = file
        self.overwrite = overwrite
        self.interactive_mode = interactive_mode

        # Fix the __init__ method - single job creation
        if job:  # Run a single job
            # Get preamble from file if provided
            if preamble_path:
                with open(preamble_path, "r") as stream:
                    preamble = stream.read()
            else:
                preamble = ""
            self.jobs = pd.DataFrame(
                {
                    "group_name": ["single_job"],
                    "preamble": [preamble],
                    "command": [job],
                    "created_at": [self._get_timestamp()],
                }
            )
        else:
            self.jobs = self.build_jobs_df()

        # Create cache file if it doesn't exist
        if osp.exists(osp.expanduser(self.cache_file)):
            # self.status_table = pd.read_csv(osp.expanduser(self.cache_file))
            self.status_table = get_status_table(cache_file=self.cache_file)

        else:
            self.status_table = pd.DataFrame(
                {
                    "job_id": [],
                    "group_name": [],
                    "job_status": [],
                    "preamble": [],
                    "command": [],
                    "created_at": [],
                    "is_starred": [],
                }
            )
        # Create cache directory if it doesn't exist
        dir_name = osp.dirname(osp.expanduser(self.cache_file))
        os.makedirs(osp.expanduser(dir_name), exist_ok=True)

    @staticmethod
    def parse_file(filepath):
        with open(filepath, "r") as stream:
            return yaml.safe_load(stream)

    @staticmethod
    def _get_timestamp():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _query_select_rows(
        self,
        table: pd.DataFrame,
        columns: list = ["job_id", "job_status", "command"],
        query: str = "Overwrite previous jobs? Enter job(s) to overwrite, or 'n' to skip: ",
        max_col_width: int = 170,
        truncate_middle: bool = True,
        truncate_length: int = 150,  # Total length to show (beginning + end)
    ):
        pd.set_option("display.max_colwidth", max_col_width)

        # Create a copy to avoid modifying the original table
        display_table = table[columns].copy()

        # Truncate strings in the middle if requested
        if truncate_middle:
            for col in columns:
                if display_table[col].dtype == "object":  # Only process string columns
                    display_table[col] = display_table[col].apply(
                        lambda x: (
                            str(x)[: truncate_length // 2]
                            + "..."
                            + str(x)[-truncate_length // 2 :]
                            if isinstance(x, str) and len(str(x)) > truncate_length
                            else x
                        )
                    )

        print(display_table)
        overwrite_string = input(query)

        if overwrite_string.lower() == "n":
            return []

        return [int(s) for s in overwrite_string.split(",") if s.strip()]

    def build_jobs(self):

        parsed = self.parse_file(self.file)
        jobs = []

        common_decl = "\n".join(parsed["common_preamble_declarations"])
        common_runs = "\n".join(parsed["common_preamble_runs"])

        for group in parsed["groups"]:
            group_preamble = "\n".join(group["preamble"])
            for job in group["paralleljobs"]:
                jobs.append(
                    {
                        "group_name": group["name"],
                        "preamble": "{}\n{}\n{}".format(
                            common_decl, group_preamble, common_runs
                        ),
                        "command": "{}".format(job),
                    }
                )

        return jobs

    def build_jobs_df(self):

        parsed = self.parse_file(self.file)
        jobs = []

        common_decl = "\n".join(parsed["common_preamble_declarations"])
        common_runs = "\n".join(parsed["common_preamble_runs"])

        for group in parsed["groups"]:
            group_preamble = "\n".join(group["preamble"])
            for job in group["paralleljobs"]:
                jobs.append(
                    {
                        "group_name": group["name"],
                        "preamble": "{}\n{}\n{}".format(
                            common_decl, group_preamble, common_runs
                        ),
                        "command": "{}".format(job),
                    }
                )

        return pd.DataFrame(jobs)

    def _run_job(self, job: dict):
        print(os.popen("rm tmp.sh").read())
        print(os.popen("touch tmp.sh").read())
        print(os.popen("echo '{}' >> tmp.sh".format(job["preamble"])).read())
        print(os.popen("echo '{}' >> tmp.sh".format(job["command"])).read())

        # Submit job
        out = os.popen("sbatch tmp.sh").read()

        print("Running: \n{}".format(out))
        job_id = out.split(" ")[-1].strip()

        new_row = {
            "job_id": job_id,
            "group_name": job["group_name"],
            "job_status": "SUBMIT",
            "preamble": job["preamble"],
            "command": job["command"],
            "created_at": self._get_timestamp(),
        }

        # Update previous row if exists
        if job["command"] in self.status_table["command"].values:
            # import pdb; pdb.set_trace()
            new_row["job_status"] = "{}*".format(new_row["job_status"])
            # self.status_table.loc[self.status_table["command"] == job["command"]] = pd.DataFrame([new_row])
            mask = self.status_table["command"] == job["command"]
            for col, value in new_row.items():
                self.status_table.loc[mask, col] = value

        else:
            self.status_table = pd.concat(
                [self.status_table, pd.DataFrame([new_row])], ignore_index=True
            )

    def submit_jobs(self):

        # Create filter list
        job_ids_filtered = (
            self._query_select_rows(
                table=self.jobs,
                query="Filter previous jobs? Enter job(s) to filter, or 'n' to skip: ",
                columns=["group_name", "command"],
            )
            if self.interactive_mode
            else range(len(self.jobs))
        )

        # Create overwrite list for status table
        st_ids_overwrite = (
            self._query_select_rows(
                table=self.status_table,
                query="Overwrite previous jobs? Enter job(s) to overwrite, or 'n' to skip: ",
            )
            if self.overwrite
            else []
        )

        NOT_COMPLETED = [
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "UNKNOWN",
            "OUT_OF_MEMORY",
            "OUT_OF_ME",
        ]

        for job_idx in job_ids_filtered:
            job = self.jobs.iloc[job_idx]

            # Look up job with matching command in status table
            st_matching_rows = self.status_table[
                self.status_table["command"] == job["command"]
            ]

            # Run job if any of the follwoing are true
            # (i) no matching rows
            # (ii) job is in overwrite list
            # (iii) has not completed status

            if (
                len(st_matching_rows) == 0
                or st_matching_rows.index[0] in st_ids_overwrite
                or any(
                    [
                        st_matching_rows.iloc[0]["job_status"].startswith(s)
                        for s in NOT_COMPLETED
                    ]
                )
            ):

                self._run_job(job)  # type: ignore

            else:
                print("Skipping job: {}".format(job["group_name"]))
                continue

        outpath = osp.expanduser(self.cache_file)
        df = pd.DataFrame(self.status_table)
        df.to_csv(outpath, index=False)
        print(self.status_table[["job_id", "job_status", "command"]])

    @staticmethod
    def _filter_table(table: pd.DataFrame, filter_str: str) -> pd.DataFrame:
        """Filter the table based on a filter string.

        The filter string can be:
        1. A simple string which will match against the command column
        2. A JSON string to filter on multiple columns
        (e.g. '{"job_status": "RUNNING", "command": "llama"}')

        Args:
            table (pd.DataFrame): DataFrame to filter.
            filter_str (str): Filter string or JSON string to filter the table.

        Returns:
            pd.DataFrame: Filtered table.
        """
        import json

        # Try to parse as JSON first
        try:
            filter_dict = json.loads(filter_str)

            # If it's a JSON object, apply column-specific filtering
            if isinstance(filter_dict, dict):
                print(f"Filtering with JSON: {filter_dict}")
                mask = pd.Series(True, index=table.index)

                for col, pattern in filter_dict.items():
                    if col in table.columns:
                        # Handle different types of columns
                        if table[col].dtype == "object":
                            # String columns
                            col_mask = (
                                table[col]
                                .astype(str)
                                .str.contains(str(pattern), case=False, na=False)
                            )
                        else:
                            # Numeric columns - try exact match first, then substring
                            try:
                                # For exact numeric match
                                if isinstance(pattern, (int, float)):
                                    col_mask = table[col] == pattern
                                else:
                                    # For string pattern on numeric column, convert to string first
                                    col_mask = (
                                        table[col]
                                        .astype(str)
                                        .str.contains(
                                            str(pattern), case=False, na=False
                                        )
                                    )
                            except:
                                # Fallback to string contains
                                col_mask = (
                                    table[col]
                                    .astype(str)
                                    .str.contains(str(pattern), case=False, na=False)
                                )

                        mask = mask & col_mask
                    else:
                        print(
                            f"Warning: Column '{col}' not found in table. Available columns: {', '.join(table.columns)}"
                        )

                filtered_table = table[mask]

                if len(filtered_table) == 0:
                    print(f"No jobs match filter criteria: {filter_str}")
                    return table
                else:
                    print(
                        f"Showing {len(filtered_table)} jobs matching criteria: {filter_str}"
                    )
                    return filtered_table

        except json.JSONDecodeError:
            # If not valid JSON, treat as simple string filter on command column
            print(f"Using simple string filter: '{filter_str}'")

        # Simple string filter on command column
        filtered_table = table[
            table["command"].str.contains(filter_str, case=False, na=False)
        ]

        if len(filtered_table) == 0:
            print(f"No jobs match filter '{filter_str}' in command column.")
            return table
        else:
            print(
                f"Showing {len(filtered_table)} jobs with command containing '{filter_str}'."
            )
            return filtered_table

    @classmethod
    def status(
        cls,
        cache_file: str,
        filter_str: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_ascending: bool = False,
    ):
        """Check the status of all jobs in the status table."""

        # Get job statuses
        status_table = get_status_table(cache_file=cache_file)

        # Apply filter if provided
        if filter_str:
            status_table = cls._filter_table(status_table, filter_str)

        # Apply sorting if requested
        if sort_by and sort_by in status_table.columns:
            if sort_by in status_table.columns:
                status_table = status_table.sort_values(
                    by=sort_by, ascending=sort_ascending
                )
                print(f"Sorted by {sort_by} ({'asc' if sort_ascending else 'desc'})")
            else:
                print(
                    f"Warning: Cannot sort by '{sort_by}'. Column not found. Available columns: {', '.join(status_table.columns)}"
                )

        # Print the status table
        reduced = status_table[["job_id", "job_status", "command", "created_at"]]
        print(reduced)

        # Print totals table
        keys = [
            "UNKNOWN",
            "SUBMIT",
            "PENDING",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
        ]
        totals = {k: 0 for k in keys}
        for status in status_table["job_status"]:
            for k in keys:
                if status.startswith(k):
                    totals[k] += 1

        totals_tbl = pd.DataFrame([totals])
        print(totals_tbl.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        help="Path to the YAML batch job file.",
    )
    parser.add_argument(
        "-i",
        "--interact",
        action="store_true",
        default=False,
        help="Choose jobs to submit interactively.",
    )
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        default=None,
        help="Submit a single job directly.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Use development file (~/.jobrunner/jobrunner_status_table_dev.csv)",
    )
    parser.add_argument(
        "-p",
        "--preamble_path",
        type=str,
        default="configs/slurm/preamble-short-unkillable-g4.txt",
        help="Path to preamble file for single jobs.",
    )

    # -----------------------
    # Status table arguments
    # -----------------------
    parser.add_argument(
        "-s",
        "--status",
        action="store_true",
        default=False,
        help="Interactive mode: view, filter and update rows in the status table.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Submit a single job directly.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default=None,
        help="Sort the status table by a specific column.",
    )
    parser.add_argument(
        "--sort_asc",
        action="store_true",
        default=False,
        help="Sort the status table in ascending order.",
    )
    args = parser.parse_args()

    cache_file = (
        "~/.jobrunner/jobrunner_status_table.csv"
        if not args.dev
        else "~/.jobrunner/jobrunner_status_table_dev.csv"
    )

    if args.status:
        SlurmJobManager.status(
            cache_file=cache_file,
            filter_str=args.filter,
            sort_by=args.sort_by,
            sort_ascending=args.sort_asc,
        )
    else:
        SlurmJobManager(
            file=args.filepath,
            interactive_mode=args.interact,
            cache_file=cache_file,
            job=args.job,
            preamble_path=args.preamble_path,
        ).submit_jobs()
