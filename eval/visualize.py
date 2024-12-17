import json
import argparse
from rich.console import Console
from rich.syntax import Syntax


def visualize_sample(file_path, index):
    # Load the specified item from samples.jsonl
    with open(file_path, "r") as f:
        lines = f.readlines()
        if index < 0 or index >= len(lines):
            print(
                f"Error: Index {index} is out of range. File contains {len(lines)} samples."
            )
            return
        selected_sample = json.loads(lines[index])

    # Extract the task_id, prompt, and completion
    task_id = selected_sample.get("task_id", "N/A")
    prompt = selected_sample.get("prompt", "")
    completion = selected_sample.get("completion", "")

    # Initialize Rich Console
    console = Console()

    # Print Task ID
    console.print(f"[bold green]Task ID:[/bold green] {task_id}\n")

    # Print Prompt with syntax highlighting
    console.print("[bold cyan]Prompt:[/bold cyan]")
    prompt_syntax = Syntax(prompt, "python", theme="monokai", line_numbers=True)
    console.print(prompt_syntax)

    # Print Completion with syntax highlighting
    console.print("\n[bold yellow]Completion:[/bold yellow]")
    completion_syntax = Syntax(completion, "python", theme="monokai", line_numbers=True)
    console.print(completion_syntax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a specific sample from samples.jsonl."
    )
    parser.add_argument("file_path", type=str, help="Path to samples.jsonl file")
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="Index of the sample to visualize (default: 0)",
    )
    args = parser.parse_args()

    visualize_sample(args.file_path, args.index)
