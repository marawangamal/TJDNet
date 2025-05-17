import json
import os.path as osp
import textwrap
from typing import Optional, Union
import torch
from tqdm import tqdm

from dataloaders._base import BaseChatTemplate
from tjdnet.models.tjd import TJD, TJDGenerationConfig
from utils.average_meter import AverageMeter

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import DatasetDict


# def collate_fn(batch, tokenizer):
#     batch_dict = {"input_ids": [item["prompt_ids"] for item in batch]}
#     padded_batch = tokenizer.pad(batch_dict, return_tensors="pt")

#     batch_dict_labels = {"input_ids": [item["input_ids"] for item in batch]}
#     padded_batch_labels = tokenizer.pad(batch_dict_labels, return_tensors="pt")

#     return {
#         **padded_batch,
#         "labels": padded_batch_labels["input_ids"],
#         "attention_mask_labels": padded_batch_labels["attention_mask"],
#     }


def collate_fn(batch, tokenizer):
    # return batch[0]
    # stack all tensors across keys
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = torch.stack([torch.tensor(b[key]) for b in batch])
    return collated_batch


def custom_shorten(text, begin_width=70, end_width=50, placeholder=" … "):
    if len(text) <= begin_width + end_width:
        return text
    return text[:begin_width] + placeholder + text[-end_width:]


def compute_accuracy(
    model: TJD,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    dataset: DatasetDict,
    chat_template: BaseChatTemplate,
    generation_config: TJDGenerationConfig,
    batch_size: int = 1,
    max_iters: Optional[int] = None,
    ckpt_dir: Optional[str] = None,
    verbose: bool = True,
    return_avg_meters: bool = False,
):
    """Compute accuracy of the model on the given dataset.

    Args:
        model (TJD): _description_
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): _description_
        dataset (DatasetDict): _description_
        chat_template (BaseChatTemplate): _description_
        generation_config (TJDGenerationConfig): _description_
        batch_size (int, optional): _description_. Defaults to 1.
        max_iters (Optional[int], optional): _description_. Defaults to None.
        ckpt_dir (Optional[str], optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to True.
        return_avg_meters (bool, optional): _description_. Defaults to False.

    Returns:
        dict: Dictionary containing the following keys:
            - accuracy (float): Accuracy of the model on the dataset.
            - acceptance_rate (float): Acceptance rate of the model on the dataset.
            - accuracy_avg_meter (dict): AverageMeter object containing the following
            - acceptance_rate_avg_meter (dict): AverageMeter object containing the following
            - total_samples (int): Total number of samples in the dataset.
    """

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )

    # Load eval results json if exists
    ckpt_path = osp.join(ckpt_dir, "eval_performance.json") if ckpt_dir else None
    results = {}
    if ckpt_path and osp.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            results = json.load(f)

    # Setup meters
    accuracy_meter = AverageMeter(**results.get("accuracy_avg_meter_kwargs", {}))
    accept_rate_meter = AverageMeter(**results.get("accept_rate_avg_meter_kwargs", {}))

    total_samples = len(dataset)
    pbar = tqdm(
        dataloader,
        total=(
            (total_samples + batch_size - 1) // batch_size
            if max_iters is None
            else max_iters
        ),
        desc="Evaluating",
        leave=True,
    )

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            if i < accuracy_meter.count // batch_size:
                continue
            batch = {k: v.to(model.device) for k, v in batch.items()}
            # input_ids, attention_mask = chat_template.format_batch(
            #     input_ids=batch["input_ids"],
            #     attention_mask=batch["attention_mask"],
            #     tokenizer=tokenizer,
            # )
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            outputs, ardict = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

            # Batched decoding
            y_pred_str = tokenizer.batch_decode(outputs)
            y_true = batch["labels"]
            y_pred = torch.tensor(
                [
                    chat_template.parse_answer(y, tokenizer.eos_token)  # type: ignore
                    for y in y_pred_str
                ],
                device=outputs.device,
            )
            corr = (y_pred == y_true).float().sum()
            accuracy_meter.update(
                corr / len(batch["input_ids"]), len(batch["input_ids"])
            )

            # # Compute accuracy
            # correct_mask = [
            #     chat_template.check_answer(y_pred[b], y_true[b], tokenizer.eos_token)  # type: ignore
            #     for b in range(len(y_pred))
            # ]

            tokens_accepted = ardict["tokens_accepted"]
            tokens_generated = ardict["tokens_generated"]

            accept_rate_meter.update(
                tokens_accepted / tokens_generated if tokens_generated > 0 else 0,
                ardict["tokens_generated"],
            )

            # Update progress bar
            pbar.set_postfix(
                {
                    "acc": f"{accuracy_meter.avg:.4f}",
                    "ar": f"{accept_rate_meter.avg:.4f}",
                }
            )

            if max_iters and i >= max_iters:
                print("Max number of samples reached, stopping evaluation.")
                break

    # ── Evaluation summary ──────────────────────────────────────────────
    if verbose and y_pred:
        line = "─" * 60
        prompt = tokenizer.decode(batch["input_ids"][0])
        summary = (
            f"\n{line}\n"
            f"📊  EVALUATION SUMMARY\n"
            f"{line}\n"
            f"Accuracy        : {accuracy_meter.avg:.2%} "
            f"({accuracy_meter.sum}/{accuracy_meter.count} correct)\n"
            f"Acceptance rate : {accept_rate_meter.avg:.2%} "
            f"({accept_rate_meter.sum}/{accept_rate_meter.count} tokens accepted)\n\n"
            f"Prompt         : {custom_shorten(prompt, begin_width=120)}\n"
            f"▶ Ground truth  : {y_true[0].item()}\n"
            f"▶ Model output  : {custom_shorten(y_pred_str[0], begin_width=120)}\n"
            f"{line}\n"
        )
        print(summary)

    if return_avg_meters:
        return {
            "accuracy": accuracy_meter.avg,
            "accuracy_avg_meter": accuracy_meter.dump(),
            "acceptance_rate": accept_rate_meter.avg,
            "acceptance_rate_avg_meter": accept_rate_meter.dump(),
            "total_samples": total_samples,
        }

    else:
        return {
            "accuracy": accuracy_meter.avg,
            "acceptance_rate": accept_rate_meter.avg,
        }
