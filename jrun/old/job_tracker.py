# jrun/job_tracker.py
import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

from .job_spec import JobSpec


class JobTracker:
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
            group_name TEXT,
            exp_id TEXT,
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

    def record_job(self, job_id: str, job_spec: JobSpec) -> None:
        """Record a job submission in the database.

        Args:
            job_id: SLURM job ID
            job_spec: The job specification
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        dependencies = ",".join(job_spec.depends_on)
        params = str(job_spec.params) if job_spec.params else None

        # Insert job record
        cursor.execute(
            """
            INSERT INTO jobs (
                job_id, name, group_name, exp_id, command, preamble, 
                status, created_at, updated_at, dependencies, params
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_spec.name,
                job_spec.group_name,
                job_spec.exp_id,
                job_spec.command,
                job_spec.preamble,
                "SUBMITTED",
                now,
                now,
                dependencies,
                params,
            ),
        )

        # Record dependencies if any
        for dep_id in job_spec.depends_on:
            cursor.execute(
                """
                INSERT INTO job_dependencies (job_id, depends_on)
                VALUES (?, ?)
                """,
                (job_id, dep_id),
            )

        conn.commit()
        conn.close()

    def update_status(self, job_id: Optional[str] = None) -> None:
        """Update the status of jobs from SLURM.

        Args:
            job_id: Optional job ID to update, or all jobs if None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get jobs to update
        if job_id:
            cursor.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,))
        else:
            cursor.execute(
                """
                SELECT job_id FROM jobs 
                WHERE status NOT IN ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT')
                """
            )

        job_ids = [row[0] for row in cursor.fetchall()]

        # Update job statuses
        for job_id in job_ids:
            try:
                out = os.popen(f"sacct -j {job_id} --format state").read()
                lines = out.strip().split("\n")
                if len(lines) >= 3:
                    status = lines[2].strip()

                    cursor.execute(
                        "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                        (status, datetime.now().isoformat(), job_id),
                    )
            except Exception as e:
                print(f"Error updating status for job {job_id}: {e}")

        conn.commit()
        conn.close()

    def get_job_status(self, job_id: str) -> str:
        """Get the status of a specific job.

        Args:
            job_id: SLURM job ID

        Returns:
            Job status as a string
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
        result = cursor.fetchone()

        conn.close()

        if result:
            return result[0]
        else:
            return "UNKNOWN"

    def get_jobs_by_exp_id(self, exp_id: str) -> pd.DataFrame:
        """Get all jobs with a specific experiment ID.

        Args:
            exp_id: Experiment ID to filter by

        Returns:
            DataFrame of matching jobs
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM jobs WHERE exp_id LIKE ?"
        params = (f"{exp_id}%",)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def get_jobs_by_group(self, group_name: str) -> pd.DataFrame:
        """Get all jobs in a specific group.

        Args:
            group_name: Group name to filter by

        Returns:
            DataFrame of matching jobs
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM jobs WHERE group_name = ?"
        params = (group_name,)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def list_jobs(
        self, filters: Optional[Dict[str, Any]] = None, sort_by: Optional[str] = None
    ) -> pd.DataFrame:
        """List jobs with optional filtering and sorting.

        Args:
            filters: Dictionary of column/value pairs to filter on
            sort_by: Column name to sort by

        Returns:
            DataFrame of jobs
        """
        conn = sqlite3.connect(self.db_path)

        # Build query
        query = "SELECT * FROM jobs"
        params = []

        if filters:
            conditions = []
            for col, value in filters.items():
                if isinstance(value, str) and ("%" in value or "_" in value):
                    conditions.append(f"{col} LIKE ?")
                else:
                    conditions.append(f"{col} = ?")
                params.append(value)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        if sort_by:
            query += f" ORDER BY {sort_by}"

        # Execute query
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def get_job_dependencies(self, job_id: str) -> List[str]:
        """Get the dependencies of a specific job.

        Args:
            job_id: SLURM job ID

        Returns:
            List of job IDs that this job depends on
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT depends_on FROM job_dependencies WHERE job_id = ?", (job_id,)
        )

        dependencies = [row[0] for row in cursor.fetchall()]
        conn.close()

        return dependencies

    def get_dependent_jobs(self, job_id: str) -> List[str]:
        """Get jobs that depend on a specific job.

        Args:
            job_id: SLURM job ID

        Returns:
            List of job IDs that depend on this job
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT job_id FROM job_dependencies WHERE depends_on = ?", (job_id,)
        )

        dependents = [row[0] for row in cursor.fetchall()]
        conn.close()

        return dependents

    def get_status_summary(
        self, group_name: Optional[str] = None, exp_id: Optional[str] = None
    ) -> Dict[str, int]:
        """Get a summary of job statuses.

        Args:
            group_name: Optional group name to filter by
            exp_id: Optional experiment ID to filter by

        Returns:
            Dictionary mapping status to count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT status, COUNT(*) FROM jobs"
        params = []

        conditions = []
        if group_name:
            conditions.append("group_name = ?")
            params.append(group_name)

        if exp_id:
            conditions.append("exp_id LIKE ?")
            params.append(f"{exp_id}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY status"

        cursor.execute(query, params)

        summary = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        return summary
