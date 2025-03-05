from typing import Optional
import torch
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import wandb
from tqdm import tqdm


from transformers import TrainerCallback
from utils.train_helpers import get_test_samples


# TODO: make generic, just needs a chat template with safe_parse
# class EvalGSM8KCallback(TrainerCallback):
# def __init__(
#     self,
#     eos_token,
#     test_dataset,
#     chat_template: BaseClassifierChatTemplate,
#     max_new_tokens=500,
#     top_k=50,
#     horizon=1,
#     num_beams=1,
#     tokenizer=None,
# ):
#     self.max_new_tokens = max_new_tokens
#     self.top_k = top_k
#     self.horizon = horizon
#     self.num_beams = num_beams
#     self.tokenizer = tokenizer
#     self.eos_token = eos_token
#     self.chat_template = chat_template
#     self.test_dataset = test_dataset

#     wandb.define_metric("eval/accuracy-v2", step_metric="global_step")

# def on_step_end(
#     self,
#     args,
#     state,
#     control,
#     model=None,
#     **kwargs,
# ):
#     # Only compute on single GPU
#     if not args.local_rank == 0:
#         return

#     steps_per_epoch = state.max_steps // state.num_train_epochs
#     should_eval = state.global_step % steps_per_epoch == 0
#     if not should_eval:
#         return

#     accuracy = compute_accuracy(
#         model,
#         self.tokenizer,
#         self.test_dataset,
#         self.eos_token,
#         self.chat_template,
#         max_new_tokens=self.max_new_tokens,
#         horizon=self.horizon,
#         top_k=self.top_k,
#         num_beams=self.num_beams,
#     )

#     print("\n=== Evaluation at step", state.global_step, "===")
#     print(f"Accuracy: {accuracy}")

#     wandb.log(
#         {
#             "eval/accuracy": accuracy,
#         },
#         step=state.global_step,
#     )

# def on_evaluate(
#     self,
#     args: TrainingArguments,
#     state: TrainerState,
#     control: TrainerControl,
#     **kwargs,
# ):
#     print("++++ EVALUATE ++++")
#     wandb.log(
#         {
#             "eval/accuracy-v2": torch.rand(1).item(),
#         },
#         step=state.global_step,
#         commit=True,
#     )


def compute_accuracy(
    model,
    tokenizer,
    test_dataset,
    eos_token,
    chat_template,
    max_new_tokens=125,
    horizon=1,
    top_k=50,
    num_beams=1,
    max_num_samples: Optional[int] = 50,
    prompt="",
):
    model.eval()
    correct = 0
    total = 0

    # Create tqdm progress bar
    pbar = tqdm(
        enumerate(test_dataset),
        total=(
            min(len(test_dataset), max_num_samples)
            if max_num_samples
            else len(test_dataset)
        ),
        desc="Computing accuracy",
        leave=False,
    )

    with torch.no_grad():
        for i, batch in pbar:
            # TODO: issue is that the input_ids contain the answer, so the model is cheating
            inputs_decoded = tokenizer.decode(batch["prompt_ids"])
            labels_decoded = tokenizer.decode(batch["input_ids"])
            pred = get_test_samples(
                model,
                tokenizer,
                prompt=prompt + inputs_decoded,
                max_new_tokens=max_new_tokens,
                horizon=horizon,
                top_k=top_k,
                num_beams=num_beams,
            )

            # Parse the sample
            # ground_truth = chat_template.safe_parse(labels_decoded, eos_token)
            # pred = chat_template.safe_parse(pred, eos_token)
            # correct += ground_truth == pred and ground_truth is not None
            correct += chat_template.check_answer(pred, labels_decoded, eos_token)
            total += 1
            pbar.set_postfix({"accuracy": f"{correct / total:.4f}"})
            if max_num_samples and total >= max_num_samples:
                break
    return correct / total


def get_int_answer(tokenizer, ids):
    labels_decoded = tokenizer.decode(ids)
    return int(labels_decoded.split("####")[1].split(tokenizer.sep_token)[0].strip())
