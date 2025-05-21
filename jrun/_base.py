###############################################################################
# Base class – JobTracker                                                     #
###############################################################################


import os
import sqlite3
from typing import Callable, Dict, Optional, Union

from jrun.interfaces import JobRecord


class JobDB:
    """Track SLURM job status with support for complex job hierarchies."""

    def __init__(self, db_path: str = "~/.cache/jobrunner/jobs.db"):
        """Initialize the job tracker.

        Args:
            db_path: Path to SQLite database for job tracking
        """
        self.db_path = os.path.expanduser(db_path)
        dir = os.path.dirname(self.db_path)
        if dir:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create jobs table if it doesn't exist
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            group TEXT,
            command TEXT NOT NULL,
            preamble TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            dependencies TEXT,
            params TEXT
        )
        """
        )

        # Create job dependencies table if it doesn't exist
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS job_dependencies (
            job_id TEXT,
            depends_on TEXT,
            PRIMARY KEY (job_id, depends_on),
            FOREIGN KEY (job_id) REFERENCES jobs (job_id)
        )
        """
        )

        conn.commit()
        conn.close()

    def add_record(self, rec: JobRecord):
        pass

    def update_record(self, rec: JobRecord):
        pass

    def delete_record(self, rec: JobRecord):
        pass

    @staticmethod
    def _get_job_statuses(
        job_ids: list, on_add_status: Optional[Callable[[str], str]] = None
    ) -> Dict[str, str]:
        """Get the status of a list of job IDs."""

        def fmt_job_id(job_id: Union[str, int, float]):
            """Get the job ID as a string."""
            # Could be a NaN
            if isinstance(job_id, float) and job_id != job_id:
                return "NaN"
            else:
                return int(job_id)

        statuses = {}
        for job_id in [fmt_job_id(job_id) for job_id in job_ids]:
            try:
                out = os.popen("sacct -j {} --format state".format(job_id)).read()
                status = out.split("\n")[2].strip()
                statuses[job_id] = on_add_status(status) if on_add_status else status
            except:
                statuses[job_id] = "UNKNOWN"
        return statuses
