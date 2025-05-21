import json
import sqlite3
from jrun._base import JobDB
from jrun.interfaces import JobSpec


class JobViewer(JobDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def visualize(self):
        pass

    def status(self):
        """Print a simple status table of all jobs."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get basic job information
        cursor.execute(
            "SELECT job_id, group_name command FROM jobs ORDER BY updated_at DESC"
        )
        jobs = cursor.fetchall()

        if not jobs:
            print("No jobs found.")
            conn.close()
            return

        # Print minimal header
        print("\nJOB STATUS")
        print("-" * 50)
        print(f"{'ID':<10} {'NAME':<25} {'STATUS':<10}")
        print("-" * 50)

        job_statuses = self._get_job_statuses([job[0] for job in jobs])

        # Print each job on one line
        for job in jobs:
            job_id, name, cmd = job

            # Truncate long names
            if len(name) > 22:
                name = name[:22] + "..."

            print(f"{job_id:<10} {name:<25} {job_statuses.get(job_id):<10}")

        conn.close()

    def list_jobs(self) -> list[JobSpec]:
        """List all jobs in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all jobs
        cursor.execute("SELECT * FROM jobs")
        jobs = cursor.fetchall()
        conn.close()

        return [
            JobSpec(
                job_id=row[0],
                command=row[1],
                preamble=row[2],
                group_name=row[3],
                depends_on=json.loads(
                    row[4] if row[4] else "[]"
                ),  # Convert JSON text to list
            )
            for row in jobs
        ]
