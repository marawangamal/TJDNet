from typing import Optional, Union
import torch
from tqdm import tqdm

from dataloaders._base import BaseChatTemplate
from tjdnet.models.tjd import TJD, TJDGenerationConfig
from utils.utils import AverageMeter

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import DatasetDict


# TODO: rename prompt_ids
def collate_fn(batch, tokenizer):
    batch_dict = {"input_ids": [item["prompt_ids"] for item in batch]}
    padded_batch = tokenizer.pad(batch_dict, return_tensors="pt")

    batch_dict_labels = {"input_ids": [item["input_ids"] for item in batch]}
    padded_batch_labels = tokenizer.pad(batch_dict_labels, return_tensors="pt")

    return {
        **padded_batch,
        "labels": padded_batch_labels["input_ids"],
        "attention_mask_labels": padded_batch_labels["attention_mask"],
    }


def compute_accuracy(
    model: TJD,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    test_dataset: DatasetDict,
    chat_template: BaseChatTemplate,
    generation_config: TJDGenerationConfig,
    # top_k: int = 50,
    # do_sample: bool = True,
    batch_size: int = 1,
    on_batch_end=None,
    log_samples=False,
    log_samples_count=10,
    # replace with generate_kwargs
    # max_new_tokens: int = 128,
    # horizon: int = 1,
    avg_meter_kwargs={},
    verbose=True,
    max_num_samples: Optional[int] = None,
    # **kwargs,
):
    dataloader = torch.utils.data.DataLoader(
        test_dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )
    model.eval()
    acc_meter = AverageMeter(**avg_meter_kwargs)
    total_samples = len(test_dataset)
    pbar = tqdm(
        dataloader,
        total=(total_samples + batch_size - 1) // batch_size,  # Ceiling division
        desc="Computing accuracy",
        leave=True,
    )
    batches_to_skip = acc_meter.count // batch_size

    y_pred = []
    y_true = []
    failures = []
    successes = []

    printv = print if verbose else lambda *args, **kwargs: None

    tokens_proposed = 0
    tokens_accepted = 0

    printv("Total number of samples:", total_samples)
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            if i < batches_to_skip:
                continue
            batch = {k: v.to(model.device) for k, v in batch.items()}
            input_ids, attention_mask = chat_template.format_batch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                tokenizer=tokenizer,
            )
            outputs, acceptance_metrics = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                # x=input_ids,
                # attn_mask=attention_mask,
                # **generate_kwargs,
            )  # (batch_size, max_seq_len') max_seq_len' might be less than max_seq_len if all sequences stopped early
            tokens_proposed += acceptance_metrics["tokens_proposed"]
            tokens_accepted += acceptance_metrics["tokens_accepted"]

            # Batched decoding
            y_pred = tokenizer.batch_decode(outputs)
            y_true = tokenizer.batch_decode(batch["labels"])
            # Compute accuracy
            correct_mask = [
                chat_template.check_answer(y_pred[b], y_true[b], tokenizer.eos_token)  # type: ignore
                for b in range(len(y_pred))
            ]
            batch_correct = sum(correct_mask)
            acc_meter.update(
                batch_correct / len(batch["input_ids"]), len(batch["input_ids"])
            )

            # Update progress bar
            pbar.set_postfix({"acc": f"{acc_meter.avg:.4f}"})

            if on_batch_end:
                on_batch_end({**acc_meter.dump(), "total_samples": total_samples})

            if log_samples:
                # Add failures to failures list and successes to successes list
                if len(failures) < log_samples_count:
                    failures.extend(
                        [
                            (y_pred[b], y_true[b])
                            for b in range(len(y_pred))
                            if not correct_mask[b]
                        ]
                    )
                if len(successes) < log_samples_count:
                    successes.extend(
                        [
                            (y_pred[b], y_true[b])
                            for b in range(len(y_pred))
                            if correct_mask[b]
                        ]
                    )
                    printv(f"Failures:\n{failures}")

            if max_num_samples and i * batch_size >= max_num_samples:
                print("Max number of samples reached, stopping evaluation.")
                break

    # Print example
    if len(y_pred) > 0 and len(y_true) > 0:
        printv("Example:")
        printv(f"y_true:\n {y_true[0]}")
        printv(f"y_pred:\n {y_pred[0]}")
    print("Acceptance rate:", tokens_accepted / tokens_proposed)
    return acc_meter.avg, {**acc_meter.dump(), "total_samples": total_samples}
