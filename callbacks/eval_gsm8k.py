import torch
import wandb

from transformers import TrainerCallback
from data.gsm8k import ChatTemplateGSM8k
from helpers import get_test_samples


class EvalGSM8KCallback(TrainerCallback):
    def __init__(
        self,
        eos_token,
        max_new_tokens=500,
        top_k=50,
        horizon=1,
        num_beams=1,
        tokenizer=None,
    ):
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.horizon = horizon
        self.num_beams = num_beams
        self.tokenizer = tokenizer
        self.eos_token = eos_token

    def on_step_end(
        self,
        args,
        state,
        control,
        model=None,
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs,
    ):
        # Only compute on single GPU
        if not args.local_rank == 0:
            return

        steps_per_epoch = state.max_steps // state.num_train_epochs
        should_eval = state.global_step % steps_per_epoch == 0
        if not should_eval:
            return

        accuracy = compute_accuracy(
            model,
            self.tokenizer,
            eval_dataloader,
            self.eos_token,
            max_new_tokens=self.max_new_tokens,
            horizon=self.horizon,
            top_k=self.top_k,
            num_beams=self.num_beams,
        )

        print("\n=== Evaluation at step", state.global_step, "===")
        print(f"Accuracy: {accuracy}")

        wandb.log(
            {
                "eval/accuracy": accuracy,
            },
            step=state.global_step,
        )


def compute_accuracy(
    model,
    tokenizer,
    eval_dataloader,
    eos_token,
    max_new_tokens=500,
    horizon=1,
    top_k=50,
    num_beams=1,
    max_num_batches=5,
):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            for i in range(batch["input_ids"].size(0)):
                input_ids = batch["input_ids"][i].unsqueeze(0).to(model.device)
                labels = batch["labels"][i].unsqueeze(0).to(model.device)
                attention_mask = (
                    batch["attention_mask"][i].unsqueeze(0).to(model.device)
                )
                input_ids = input_ids[:, : attention_mask.sum()]
                labels = labels[:, : attention_mask.sum()]
                inputs_decoded = tokenizer.decode(input_ids[0])
                labels_decoded = tokenizer.decode(labels[0])
                pred = get_test_samples(
                    model,
                    tokenizer,
                    prompt=inputs_decoded,
                    max_new_tokens=max_new_tokens,
                    horizon=horizon,
                    top_k=top_k,
                    num_beams=num_beams,
                )

                # Parse the sample
                ground_truth = ChatTemplateGSM8k.safe_parse(labels_decoded, eos_token)
                pred = ChatTemplateGSM8k.safe_parse(pred, eos_token)
                correct += ground_truth == pred and ground_truth is not None
                total += 1

            if total >= max_num_batches:
                break
    return correct / total


def get_int_answer(tokenizer, ids):
    labels_decoded = tokenizer.decode(ids)
    return int(labels_decoded.split("####")[1].split(tokenizer.sep_token)[0].strip())
