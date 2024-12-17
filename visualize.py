import json
import argparse
from rich.console import Console
from rich.syntax import Syntax


def visualize_first_sample(file_path):
    # Load the first item from samples.jsonl
    with open(file_path, "r") as f:
        first_line = f.readline()
        first_sample = json.loads(first_line)

    # Extract the task_id, prompt, and completion
    task_id = first_sample.get("task_id", "N/A")
    prompt = first_sample.get("prompt", "")
    completion = first_sample.get("completion", "")

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
        description="Visualize prompt and completion from samples.jsonl."
    )
    parser.add_argument("file_path", type=str, help="Path to samples.jsonl file")
    args = parser.parse_args()

    visualize_first_sample(args.file_path)
