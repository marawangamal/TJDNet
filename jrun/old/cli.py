#!/usr/bin/env python3
# cli.py - Command-line interface for Enhanced JobRunner

import os
import sys
import argparse
import uuid
import json
from typing import Any

import pandas as pd

from jrun.job_spec import JobSpec
from jrun.job_submitter import JobSubmitter
from jrun.job_tracker import JobTracker
from jrun.config_manager import ConfigManager


def command_submit(args):
    """Handle 'submit' command."""
    # Initialize components
    tracker = JobTracker(args.db)
    submitter = JobSubmitter(tracker)

    # Handle different submission methods
    if args.config:
        # Load configuration from YAML
        config = ConfigManager(args.config)

        if args.group:
            # Submit specific group
            for group in config.get_job_groups():
                if group.name == args.group:
                    print(f"Submitting group: {group.name}")
                    job_ids = submitter.submit_group(group)
                    print(f"Submitted {len(job_ids)} jobs in group {group.name}")

                    # Print summary
                    if job_ids:
                        print("\nJob IDs:")
                        for name, job_id in job_ids.items():
                            print(f"  {name}: {job_id}")
                    break
            else:
                print(f"Group '{args.group}' not found in configuration.")
        else:
            # Submit all groups
            job_ids = {}
            for group in config.get_job_groups():
                print(f"Submitting group: {group.name}")
                group_ids = submitter.submit_group(group)
                job_ids.update(group_ids)

            print(f"Submitted {len(job_ids)} jobs in total")

            # Print summary
            if job_ids:
                print("\nJob IDs:")
                for name, job_id in list(job_ids.items())[:10]:  # Show first 10 jobs
                    print(f"  {name}: {job_id}")
                if len(job_ids) > 10:
                    print(f"  ... and {len(job_ids) - 10} more")

    elif args.command:
        # Create a simple job from command line arguments
        job_spec = JobSpec(
            name=args.name or f"job_{uuid.uuid4().hex[:8]}",
            command=args.command,
            preamble="\n".join(args.preamble) if args.preamble else "#!/bin/bash",
            group_name=args.group_name or "cli",
            exp_id=args.exp_id or uuid.uuid4().hex[:8],
        )

        # Submit the job
        job_id = submitter.submit_job(job_spec)
        print(f"Submitted job with ID {job_id}")

    else:
        print("Error: Must provide either --config or --command")
        sys.exit(1)


def command_status(args):
    """Handle 'status' command."""
    tracker = JobTracker(args.db)

    # Update job statuses
    tracker.update_status()

    # Build filters
    filters = {}

    if args.group:
        filters["group_name"] = args.group

    if args.exp_id:
        filters["exp_id"] = f"{args.exp_id}%"

    if args.status:
        filters["status"] = args.status.upper()

    if args.filter:
        filters["command"] = f"%{args.filter}%"

    # Get jobs
    jobs_df = tracker.list_jobs(filters, args.sort_by)

    if len(jobs_df) == 0:
        print("No jobs found matching criteria.")
        return

    # Display jobs
    if args.verbose:
        # Show all columns
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", 50)  # Truncate long fields
        print(jobs_df)
    else:
        # Show simplified view
        columns = ["job_id", "name", "status", "exp_id", "created_at"]
        pd.set_option("display.max_rows", None)
        print(jobs_df[columns])

    # Show summary
    summary = tracker.get_status_summary(group_name=args.group, exp_id=args.exp_id)

    print("\nStatus Summary:")
    for status, count in summary.items():
        print(f"  {status}: {count}")


def command_cancel(args):
    """Handle 'cancel' command."""
    tracker = JobTracker(args.db)

    if args.group:
        # Cancel all jobs in a group
        jobs_df = tracker.get_jobs_by_group(args.group)
        job_ids = jobs_df["job_id"].tolist()
    elif args.exp_id:
        # Cancel all jobs with a specific experiment ID
        jobs_df = tracker.get_jobs_by_exp_id(args.exp_id)
        job_ids = jobs_df["job_id"].tolist()
    else:
        # Cancel specific job IDs
        job_ids = args.job_ids

    if not job_ids:
        print("No jobs to cancel.")
        return

    # Confirm cancellation
    if not args.force:
        print(f"About to cancel {len(job_ids)} jobs:")
        for job_id in job_ids[:10]:
            status = tracker.get_job_status(job_id)
            print(f"  {job_id} ({status})")

        if len(job_ids) > 10:
            print(f"  ... and {len(job_ids) - 10} more")

        confirm = input("Are you sure you want to cancel these jobs? [y/N] ")
        if confirm.lower() not in ["y", "yes"]:
            print("Cancellation aborted.")
            return

    # Cancel jobs
    for job_id in job_ids:
        print(f"Cancelling job {job_id}...")
        os.system(f"scancel {job_id}")

    # Update job status in database
    tracker.update_status()
    print(f"Cancelled {len(job_ids)} jobs.")


def command_resubmit(args):
    """Handle 'resubmit' command."""
    tracker = JobTracker(args.db)
    submitter = JobSubmitter(tracker)

    # Get jobs to resubmit
    if args.group:
        # Resubmit all jobs in a group
        jobs_df = tracker.get_jobs_by_group(args.group)
        job_ids = jobs_df["job_id"].tolist()
    elif args.exp_id:
        # Resubmit all jobs with a specific experiment ID
        jobs_df = tracker.get_jobs_by_exp_id(args.exp_id)
        job_ids = jobs_df["job_id"].tolist()
    else:
        # Resubmit specific job IDs
        job_ids = args.job_ids

    if not job_ids:
        print("No jobs to resubmit.")
        return

    # Filter jobs based on status
    if args.failed_only:
        filtered_ids = []
        for job_id in job_ids:
            status = tracker.get_job_status(job_id)
            if status in ["FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED"]:
                filtered_ids.append(job_id)
        job_ids = filtered_ids

    if not job_ids:
        print("No jobs to resubmit after filtering.")
        return

    # Confirm resubmission
    if not args.force:
        print(f"About to resubmit {len(job_ids)} jobs:")
        for job_id in job_ids[:10]:
            status = tracker.get_job_status(job_id)
            print(f"  {job_id} ({status})")

        if len(job_ids) > 10:
            print(f"  ... and {len(job_ids) - 10} more")

        confirm = input("Are you sure you want to resubmit these jobs? [y/N] ")
        if confirm.lower() not in ["y", "yes"]:
            print("Resubmission aborted.")
            return

    # Resubmit jobs
    new_job_ids = {}

    for job_id in job_ids:
        # Get job details
        jobs_df = tracker.list_jobs({"job_id": job_id})

        if len(jobs_df) == 0:
            print(f"Job {job_id} not found.")
            continue

        # Create a new job spec from the existing job
        job_row = jobs_df.iloc[0]

        job_spec = JobSpec(
            name=job_row["name"],
            command=job_row["command"],
            preamble=job_row["preamble"],
            group_name=job_row["group_name"],
            exp_id=job_row["exp_id"],
            depends_on=[],  # Clear dependencies for resubmission
            params=json.loads(job_row["params"]) if job_row["params"] else {},
        )

        # Submit the job
        new_job_id = submitter.submit_job(job_spec)
        new_job_ids[job_id] = new_job_id
        print(f"Resubmitted job {job_id} as {new_job_id}")

    print(f"Resubmitted {len(new_job_ids)} jobs.")


def command_visualize(args):
    """Handle 'visualize' command to generate dependency graphs."""
    import networkx as nx
    import matplotlib.pyplot as plt

    tracker = JobTracker(args.db)

    # Build graph of job dependencies
    if args.group:
        # Visualize a specific group
        jobs_df = tracker.get_jobs_by_group(args.group)
    elif args.exp_id:
        # Visualize a specific experiment
        jobs_df = tracker.get_jobs_by_exp_id(args.exp_id)
    else:
        # Visualize all jobs
        jobs_df = tracker.list_jobs()

    if len(jobs_df) == 0:
        print("No jobs found matching criteria.")
        return

    # Build dependency graph
    G = nx.DiGraph()

    # Add nodes (jobs)
    for _, job in jobs_df.iterrows():
        job_id = job["job_id"]
        G.add_node(job_id, name=job["name"], status=job["status"], exp_id=job["exp_id"])

    # Add edges (dependencies)
    for _, job in jobs_df.iterrows():
        job_id = job["job_id"]
        if job["dependencies"]:
            for dep_id in job["dependencies"].split(","):
                if dep_id in G:
                    G.add_edge(dep_id, job_id)

    # Print graph summary
    print(
        f"Dependency graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    # Save visualization if requested
    if args.output:
        try:
            # Create node colors based on status
            status_colors = {
                "SUBMITTED": "gray",
                "PENDING": "yellow",
                "RUNNING": "blue",
                "COMPLETED": "green",
                "FAILED": "red",
                "CANCELLED": "orange",
                "TIMEOUT": "purple",
            }

            node_colors = [
                status_colors.get(G.nodes[n]["status"], "gray") for n in G.nodes
            ]

            # Create layout
            pos = nx.spring_layout(G)

            # Draw graph
            plt.figure(figsize=(12, 8))
            nx.draw(
                G,
                pos,
                node_color=node_colors,
                with_labels=True,
                labels={n: G.nodes[n]["name"] for n in G.nodes},
                font_size=8,
                node_size=500,
                alpha=0.8,
            )

            # Add legend
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=status,
                )
                for status, color in status_colors.items()
            ]
            plt.legend(handles=legend_elements, loc="upper right")

            # Add title
            if args.group:
                plt.title(f"Job Dependencies for Group: {args.group}")
            elif args.exp_id:
                plt.title(f"Job Dependencies for Experiment: {args.exp_id}")
            else:
                plt.title("Job Dependencies")

            # Save figure
            plt.savefig(args.output)
            print(f"Saved dependency graph to {args.output}")
        except Exception as e:
            print(f"Error creating visualization: {e}")


def main():
    """Main entry point for JobRunner CLI."""
    # Create main parser
    parser = argparse.ArgumentParser(
        description="Enhanced SLURM Job Runner with dependency management"
    )
    parser.add_argument(
        "--db", type=str, default="~/.jobrunner/jobs.db", help="Path to job database"
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit jobs to SLURM")
    submit_group = submit_parser.add_mutually_exclusive_group(required=True)
    submit_group.add_argument(
        "--config", type=str, help="Path to job configuration YAML file"
    )
    submit_group.add_argument("--command", type=str, help="Single command to run")
    submit_parser.add_argument(
        "--group", type=str, help="Submit only a specific group from config"
    )
    submit_parser.add_argument("--name", type=str, help="Name for single job")
    submit_parser.add_argument(
        "--preamble",
        type=str,
        action="append",
        help="SLURM preamble line(s) for single job",
    )
    submit_parser.add_argument(
        "--group-name", type=str, dest="group_name", help="Group name for single job"
    )
    submit_parser.add_argument(
        "--exp-id", type=str, dest="exp_id", help="Experiment ID for single job"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--group", type=str, help="Filter jobs by group name")
    status_parser.add_argument(
        "--exp-id", type=str, dest="exp_id", help="Filter jobs by experiment ID"
    )
    status_parser.add_argument(
        "--status", type=str, help="Filter jobs by status (PENDING, RUNNING, etc.)"
    )
    status_parser.add_argument(
        "--filter", type=str, help="Filter jobs by command substring"
    )
    status_parser.add_argument(
        "--sort-by", type=str, dest="sort_by", help="Sort jobs by column"
    )
    status_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all job details"
    )

    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel jobs")
    cancel_group = cancel_parser.add_mutually_exclusive_group(required=True)
    cancel_group.add_argument(
        "job_ids", nargs="*", default=[], help="Job IDs to cancel"
    )
    cancel_group.add_argument("--group", type=str, help="Cancel all jobs in a group")
    cancel_group.add_argument(
        "--exp-id", type=str, dest="exp_id", help="Cancel all jobs with experiment ID"
    )
    cancel_parser.add_argument(
        "--force", "-f", action="store_true", help="Skip confirmation prompt"
    )

    # Resubmit command
    resubmit_parser = subparsers.add_parser("resubmit", help="Resubmit jobs")
    resubmit_group = resubmit_parser.add_mutually_exclusive_group(required=True)
    resubmit_group.add_argument(
        "job_ids", nargs="*", default=[], help="Job IDs to resubmit"
    )
    resubmit_group.add_argument(
        "--group", type=str, help="Resubmit all jobs in a group"
    )
    resubmit_group.add_argument(
        "--exp-id", type=str, dest="exp_id", help="Resubmit all jobs with experiment ID"
    )
    resubmit_parser.add_argument(
        "--failed-only", action="store_true", help="Only resubmit failed jobs"
    )
    resubmit_parser.add_argument(
        "--force", "-f", action="store_true", help="Skip confirmation prompt"
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize job dependencies"
    )
    visualize_group = visualize_parser.add_mutually_exclusive_group()
    visualize_group.add_argument("--group", type=str, help="Visualize jobs in a group")
    visualize_group.add_argument(
        "--exp-id", type=str, dest="exp_id", help="Visualize jobs with experiment ID"
    )
    visualize_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (e.g., deps.png)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == "submit":
        command_submit(args)
    elif args.command == "status":
        command_status(args)
    elif args.command == "cancel":
        command_cancel(args)
    elif args.command == "resubmit":
        command_resubmit(args)
    elif args.command == "visualize":
        command_visualize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
