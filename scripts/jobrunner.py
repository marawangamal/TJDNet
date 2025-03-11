"""SLURM Job Manager for submitting and tracking multiple SLURM jobs.

This module allows you to define jobs in a YAML file and submit them in batches to a SLURM cluster.
It keeps track of job statuses and provides a way to check on all previously submitted jobs.

Usage:
    python jobrunner.py -f config.yaml  # Submit jobs
    python jobrunner.py -s              # Check status of all jobs
    python jobrunner.py -c              # Clear job history cache


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
import os
import sys
import os.path as osp
import pickle

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


class SlurmJobManager:
    cache_dir = "~/.jobrunner"
    cache_file_status = "jobrunner_status_table.csv"

    def __init__(self, file, overwrite=False, filter=False):
        """A manager for submitting and tracking SLURM jobs defined in a YAML configuration file.

        Args:
            file (str): Path to the YAML configuration file.
            overwrite (bool): Flag to allow overwriting of previously submitted jobs.
        """
        self.cache_dir = self.__class__.cache_dir
        self.cache_file_status = self.__class__.cache_file_status
        self.file = file
        self.overwrite = overwrite
        self.filter = filter

        # self.jobs = self.build_jobs()
        self.jobs = self.build_jobs_df()

        # Create cache file if it doesn't exist
        if osp.exists(osp.join(osp.expanduser(self.cache_dir), self.cache_file_status)):
            self.status_table = pd.read_csv(
                osp.join(osp.expanduser(self.cache_dir), self.cache_file_status)
            )

        else:
            self.status_table = pd.DataFrame(
                {
                    "job_id": [],
                    "group_name": [],
                    "job_status": [],
                    "preamble": [],
                    "command": [],
                }
            )

        if not osp.exists(osp.expanduser(self.cache_dir)):
            os.makedirs(osp.expanduser(self.cache_dir))

    @staticmethod
    def parse_file(filepath):
        with open(filepath, "r") as stream:
            return yaml.safe_load(stream)

    def _query_select_rows(
        self,
        table: pd.DataFrame,
        columns: list = ["job_id", "job_status", "command"],
        query: str = "Overwrite previous jobs? Enter job(s) to overwrite, or 'n' to skip: ",
    ):
        pd.set_option("display.max_colwidth", 150)
        print(table[columns])
        overwrite_string = input(query)
        return [int(s) for s in overwrite_string.split(",")]

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

    def submit_jobs(self):

        # Create filter list
        filter_row_ids = (
            self._query_select_rows(
                table=self.jobs,
                query="Filter previous jobs? Enter job(s) to filter, or 'n' to skip: ",
                columns=["group_name", "command"],
            )
            if self.filter
            else range(len(self.jobs))
        )

        # Create overwrite list
        overwrite_row_ids = (
            self._query_select_rows(
                table=self.status_table,
                query="Overwrite previous jobs? Enter job(s) to overwrite, or 'n' to skip: ",
            )
            if self.overwrite
            else []
        )

        for idx in filter_row_ids:
            job = self.jobs.iloc[idx]

            # Look up job with matching command in status table
            matching_rows = self.status_table[
                self.status_table["command"] == job["command"]
            ]

            # Run job if no matching rows or job is in overwrite list and not currently running
            if (
                len(matching_rows) == 0
                or idx in overwrite_row_ids
                and not any(
                    [
                        matching_rows.iloc[0]["job_status"].startswith(s)
                        for s in ["RUNNING", "PENDING", "SUBMIT"]
                    ]
                )
            ):

                print("Running job: {}".format(job["group_name"]))
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

            else:
                print("Skipping job: {}".format(job["group_name"]))
                continue

        outpath = osp.join(osp.expanduser(self.cache_dir), self.cache_file_status)
        df = pd.DataFrame(self.status_table)
        df.to_csv(outpath, index=False)
        print(self.status_table[["job_id", "job_status", "command"]])

    @classmethod
    def status(cls):
        outpath = osp.join(osp.expanduser(cls.cache_dir), cls.cache_file_status)
        if not osp.exists(outpath):
            print("No status table found. Run `jobrunner.py -f <filepath>` first.")
            return
        status_table = pd.read_csv(outpath)

        # Ensure `job_id` is an integer
        status = []

        # import pdb; pdb.set_trace()
        for job_id in status_table["job_id"]:
            try:
                job_id_int = str(int(job_id))
                # Get job status
                out = os.popen("sacct -j {} --format state".format(job_id_int)).read()
                status_i = out.split("\n")[2].strip()
                # If status in table ensds with *, then add * to status_i
                if (
                    status_table[status_table["job_id"] == job_id]["job_status"]
                    .values[0]
                    .endswith("*")
                ):
                    status_i = "{}*".format(status_i)
                status.append(status_i)
            except:
                status.append("UNKNOWN")
        status_table["job_status"] = status
        status_table.to_csv(outpath)

        reduced = status_table[["job_id", "job_status", "command"]]
        print(reduced)

        # totals
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
    )
    parser.add_argument("-o", "--overwrite", action="store_true", default=False)
    parser.add_argument("-s", "--status", action="store_true", default=False)
    parser.add_argument("-c", "--clear", action="store_true", default=False)
    parser.add_argument("--filter", action="store_true", default=False)
    args = parser.parse_args()

    if args.clear:
        res = query_yes_no("Clear cache?", default="no")
        if res:
            os.system("rm -r ~/.jobrunner/jobrunner_status_table.csv")
            print("Cleared cache.")
        else:
            print("Did not clear cache.")
        sys.exit()

    if args.status:
        SlurmJobManager.status()
    else:
        SlurmJobManager(args.filepath, args.overwrite, args.filter).submit_jobs()
