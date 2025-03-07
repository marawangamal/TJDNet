from typing import Optional
import torch
from tqdm import tqdm

from utils.average_meter import AverageMeter


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
    model,
    tokenizer,
    test_dataset,
    chat_template,
    max_new_tokens=125,
    horizon=1,
    top_k=50,
    do_sample=True,
    batch_size=1,
    max_num_samples: Optional[int] = 50,
    **kwargs,
):
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )
    model.eval()
    # correct = 0
    # total = 0
    acc_meter = AverageMeter()
    acc_meter_lst = AverageMeter()

    # Create tqdm progress bar
    total_samples = len(test_dataset)
    if max_num_samples:
        total_samples = min(total_samples, max_num_samples)

    pbar = tqdm(
        dataloader,
        total=(total_samples + batch_size - 1) // batch_size,  # Ceiling division
        desc="Computing accuracy",
        leave=False,
    )

    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                do_sample=do_sample,
                horizon=horizon,
                stop_token=tokenizer.eos_token_id,  # For tjd models this causes stopping when end token is reached
            )  # (batch_size, max_seq_len') max_seq_len' might be less than max_seq_len if all sequences stopped early

            outputs_lst = [
                model.generate(
                    input_ids=batch["input_ids"][i : i + 1],
                    attention_mask=batch["attention_mask"][i : i + 1],
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    do_sample=do_sample,
                    horizon=horizon,
                    stop_token=tokenizer.eos_token_id,  # For tjd models this causes stopping when end token is reached
                    return_tensors="pt",
                )
                for i in range(len(batch["input_ids"]))
            ]

            # Batched decoding
            y_pred = tokenizer.batch_decode(outputs)
            y_true = tokenizer.batch_decode(batch["labels"])

            y_pred_lst = [
                tokenizer.decode(outputs_lst[i][0])
                for i in range(len(batch["input_ids"]))
            ]
            y_true_lst = [
                tokenizer.decode(batch["labels"][i])
                for i in range(len(batch["input_ids"]))
            ]

            # Compute accuracy
            batch_correct = sum(
                [
                    chat_template.check_answer(
                        y_pred[b], y_true[b], tokenizer.eos_token
                    )
                    for b in range(len(y_pred))
                ]
            )

            batch_correct_lst = sum(
                [
                    chat_template.check_answer(
                        y_pred_lst[b], y_true_lst[b], tokenizer.eos_token
                    )
                    for b in range(len(y_pred_lst))
                ]
            )

            # correct += batch_correct
            # total += batch_size_actual
            acc_meter.update(
                batch_correct / len(batch["input_ids"]), len(batch["input_ids"])
            )
            acc_meter_lst.update(
                batch_correct_lst / len(batch["input_ids"]), len(batch["input_ids"])
            )

            # Update progress bar
            pbar.set_postfix(
                {"acc": f"{acc_meter.avg:.4f}", "acc_lst": f"{acc_meter_lst.avg:.4f}"}
            )

            if acc_meter.avg != acc_meter_lst.avg:
                print("Error: acc_meter and acc_meter_lst are not equal")

            if max_num_samples and acc_meter.count >= max_num_samples:
                break

    # Print example
    print("Example:")
    print("y_true:", y_true[0])
    print("y_pred:", y_pred[0])

    return acc_meter_lst.avg
