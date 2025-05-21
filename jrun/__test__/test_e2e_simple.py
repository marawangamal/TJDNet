#!/usr/bin/env python3
"""
Simple test for jrun package.
This script tests the basic functionality of jrun by submitting a simple job.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from jrun.job_submitter import JobSubmitter


class TestJrunSimple(unittest.TestCase):
    """Simple test for jrun package."""

    def setUp(self):
        """Set up the test by creating a test YAML file."""
        # Create temporary test.yaml file
        with open("test.yaml", "w") as f:
            f.write(
                """
# Define reusable preamble templates
preambles:
  base:
    - "#!/bin/bash"
    - "#SBATCH --partition=debug"
    - "#SBATCH --output=test-%j.out"
    - "#SBATCH --error=test-%j.err"
  
  gpu:
    - "#SBATCH --gres=gpu:1"
    - "#SBATCH --mem=8G"

groups:
  - name: test-group
    sequentialjobs:
      - job:
          preamble: base
          command: "echo 'First job'"
      - job:
          preamble: gpu
          command: "echo 'Second job'"
"""
            )

    def tearDown(self):
        """Clean up after the test."""
        # Remove temporary file
        if os.path.exists("test.yaml"):
            os.remove("test.yaml")

        # Remove database if created
        if os.path.exists("test.db"):
            os.remove("test.db")

    @patch("os.popen")
    def test_submit_jobs(self, mock_popen):
        """Test that jobs are submitted correctly."""
        # Setup mock to return job IDs
        mock_popen.side_effect = [
            MagicMock(read=MagicMock(return_value="Submitted batch job 12345")),
            MagicMock(read=MagicMock(return_value="Submitted batch job 12346")),
        ]
        # Initialize components
        submitter = JobSubmitter("test.db")
        root = {
            "group": {
                "name": "test-group",
                "type": "sequential",
                "jobs": [
                    {
                        "job": {
                            "preamble": "base",
                            "command": "echo 'First job'",
                        }
                    },
                    {
                        "job": {
                            "preamble": "gpu",
                            "command": "echo 'Second job'",
                        }
                    },
                ],
            }
        }
        preamble_map = {
            "base": "\n".join(
                [
                    "#!/bin/bash",
                    "#SBATCH --partition=debug",
                    "#SBATCH --output=test-%j.out",
                    "#SBATCH --error=test-%j.err",
                ]
            ),
            "gpu": "\n".join(
                [
                    "#SBATCH --gres=gpu:1",
                    "#SBATCH --mem=8G",
                ]
            ),
        }

        job_ids = submitter.walk(
            node=root["group"],
            group_name=root["group"]["name"],
            preamble_map=preamble_map,
        )
        if not job_ids:
            raise RuntimeError("No job IDs returned from walk method.")

        # Verify submission
        self.assertEqual(len(job_ids), 2)
        self.assertTrue(any(12345 == value for value in job_ids))
        self.assertTrue(any(12346 == value for value in job_ids))

        # Verify the sbatch command was called twice
        self.assertEqual(mock_popen.call_count, 2)

        # Verify the first call had base preamble
        first_call_args = mock_popen.call_args_list[0][0][0]
        self.assertIn("sbatch", first_call_args)

        # Verify jobs are in the database
        # jobs = tracker.list_jobs()
        # self.assertEqual(len(jobs), 2)

        # Verify job statuses
        # job_ids_list = jobs["job_id"].tolist()
        # self.assertIn("12345", job_ids_list)
        # self.assertIn("12346", job_ids_list)

        # Verify second job depends on first job
        # second_job_row = jobs[jobs["job_id"] == "12346"]
        # if not second_job_row.empty:
        #     dependencies = second_job_row["dependencies"].iloc[0]
        #     self.assertIn("12345", dependencies)

        print("Test completed successfully!")


if __name__ == "__main__":
    unittest.main()
