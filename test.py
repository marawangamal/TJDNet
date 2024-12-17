from human_eval.data import read_problems, write_jsonl


def generate_one_completion(prompt):
    return "print('hello world')"


def main():
    problems = read_problems()

    num_samples_per_task = 200
    samples = [
        dict(
            task_id=task_id,
            completion=generate_one_completion(problems[task_id]["prompt"]),
        )
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    write_jsonl("samples.jsonl", samples)


if __name__ == "__main__":
    main()
