from typing import Optional
import torch
from tqdm import tqdm


from utils.train_helpers import get_test_samples


def compute_accuracy_v1(
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
    batchsize=1,
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
    batch_size=32,
    max_num_samples: Optional[int] = None,
    **kwargs,
):
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )
    model.eval()
    correct = 0
    total = 0

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

            # Batched decoding
            y_pred = tokenizer.batch_decode(outputs)
            y_true = tokenizer.batch_decode(batch["labels"])

            # Compute accuracy
            batch_correct = sum(
                [chat_template.check_answer(y_pred, y_true, tokenizer.eos_token)]
            )
            correct += batch_correct
            batch_size_actual = len(batch["input_ids"])
            total += batch_size_actual

            # Update progress bar
            pbar.set_postfix({"accuracy": f"{correct / total:.4f}"})

            if max_num_samples and total >= max_num_samples:
                break

    return correct / total


def get_int_answer(tokenizer, ids):
    labels_decoded = tokenizer.decode(ids)
    return int(labels_decoded.split("####")[1].split(tokenizer.sep_token)[0].strip())
