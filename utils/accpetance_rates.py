from typing import Union
import torch
from tqdm import tqdm

from dataloaders._base import BaseChatTemplate
from utils.utils import AverageMeter

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import DatasetDict


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


def compute_acceptance_rate(
    model: torch.nn.Module,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    test_dataset: DatasetDict,
    chat_template: BaseChatTemplate,
    # top_k: int = 50,
    # do_sample: bool = True,
    batch_size: int = 1,
    on_batch_end=None,
    log_samples=False,
    log_samples_count=10,
    avg_meter_kwargs={},
    generate_kwargs={},
    **kwargs,
):

    ar_meter = AverageMeter(**avg_meter_kwargs)

    dataloader = torch.utils.data.DataLoader(
        test_dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )
    model.eval()
    total_samples = len(test_dataset)
    pbar = tqdm(
        dataloader,
        total=(total_samples + batch_size - 1) // batch_size,  # Ceiling division
        desc="Computing acceptance rate",
        leave=True,
    )
    batches_to_skip = ar_meter.count // batch_size

    print("Total number of samples:", total_samples)
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
            _, acceptance_metrics = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )  # (batch_size, max_seq_len') max_seq_len' might be less than max_seq_len if all sequences stopped early

            ar_meter.update(
                val=acceptance_metrics["tokens_accepted"]
                / acceptance_metrics["tokens_proposed"],
                n=acceptance_metrics["tokens_accepted"],
            )

            pbar.set_postfix({"acc": f"{ar_meter.avg:.4f}"})

            if on_batch_end:
                on_batch_end({**ar_meter.dump(), "total_samples": total_samples})

    return ar_meter.avg, {**ar_meter.dump(), "total_samples": total_samples}
