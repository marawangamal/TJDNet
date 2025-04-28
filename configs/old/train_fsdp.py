"""

Example
    salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:a100l:4 --time=24:00:00 --mem=256G
    python train_fsdp.py --batch-size 1 --seq-len 8

Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
Run these to get the data in ../data folder
    !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    !tar -zxvf MNIST.tar.gz

Getting the FSDP `transformer_layer_cls`
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True)

"""

import enum
from math import e
import os
import argparse
import functools
from tqdm import tqdm

from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)

# from torch.distributed.fsdp import FullStateDictConfig, StateDictType
# from torch.distributed.fsdp.fully_sharded_data_parallel import (
#     FullyShardedDataParallel as FSDP,
# )

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    _or_policy,
)
import pdb
import sys

from utils.accuracy import compute_accuracy
from tjdnet.distributions.base import BaseDist
from utils.helpers import get_chat_template, get_model_and_tokenizer, parse_args


def create_tjd_auto_wrap_policy(
    transformer_layer_cls,  # LlamaDecoderLayer
    base_dist_cls,  # Your BaseDist class
):
    """
    Creates a policy that wraps both transformer layers and BaseDist modules in FSDP.
    """

    def policy_fn(module, recurse, unwrapped_params, **kwargs):
        # Check if the module is a transformer layer
        if isinstance(module, tuple(transformer_layer_cls)):
            return True

        # Check if the module is a BaseDist
        if isinstance(module, base_dist_cls):
            return True

        # Use default recursive wrapping for other modules
        return False

    return policy_fn


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


# from torch.distributed.fsdp.fully_sharded_data_parallel import (
#     CPUOffload,
#     BackwardPrefetch,
# )
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from dataloaders.gsm8k import load_gsm8k_data


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    printr(f"Training started on rank {rank}", rank)
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    # Create tqdm progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")

    for batch in train_loader:
        # Prepare batch for GPU
        batch_dv = {
            k: v.to(rank) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }

        # Forward and backward pass
        optimizer.zero_grad()
        output = model(**batch_dv)
        # loss = output.loss # for hf model
        loss = output["loss"]  # for tjd model
        loss.backward()
        optimizer.step()

        # Track loss
        ddp_loss[0] += loss.item()
        first_key = next(iter(batch_dv.keys()))
        ddp_loss[1] += len(batch_dv[first_key])

        # Update progress bar on rank 0
        if rank == 0:
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})

    # Close progress bar
    if rank == 0:
        pbar.close()

    printr(f"Rank {rank} done with training, about to reduce", rank)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_loss = ddp_loss[0] / ddp_loss[1]
        print(f"Train Epoch: {epoch} \tAverage Loss: {avg_loss:.6f}")


def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                int(ddp_loss[1]),
                int(ddp_loss[2]),
                100.0 * ddp_loss[1] / ddp_loss[2],
            )
        )


def test_v2(
    model,
    tokenizer,
    test_dataset,
    chat_template,
    horizon,
    top_k,
    num_beams,
    eos_token,
    rank,
    world_size,
):

    # Only process on rank 0 to avoid duplicate computations
    if rank == 0:
        with torch.no_grad():
            printr(f"Unsharding model for evaluation...", rank)

            # Configure FSDP to gather full parameters on rank 0
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )

            # Temporarily consolidate the model
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, full_state_dict_config
            ):
                accuracy = compute_accuracy(
                    model=model,
                    tokenizer=tokenizer,
                    test_dataset=test_dataset,
                    eos_token=eos_token,
                    chat_template=chat_template,
                    horizon=horizon,
                    top_k=top_k,
                    num_beams=num_beams,
                )
                printr(f"Test accuracy: {accuracy:.4f}", rank)

    # Make sure all processes wait for the evaluation to complete
    dist.barrier()


def printr(msg: str, rank: int):
    if rank == 0:
        print(msg)


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    model, tokenizer = get_model_and_tokenizer(args)
    transformer_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={type(model.model.layers[0])},
    )

    # 2. Create a direct function for BaseDist wrapping
    def base_dist_policy(module, recurse=True, **kwargs):
        return isinstance(module, BaseDist)

    # 3. Combine the policies
    combined_policy = functools.partial(
        _or_policy, policies=[transformer_policy, base_dist_policy]
    )
    model = FSDP(
        model,
        auto_wrap_policy=combined_policy,
        device_id=rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )
    chat_template = get_chat_template(args)
    printr(f"Model created on rank {rank}", rank)

    lm_dataset = load_gsm8k_data(tokenizer, input_seq_len=128, print_stats=rank == 0)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Create proper distributed sampler
    train_sampler = DistributedSampler(
        lm_dataset["train"],  # pyright: ignore[reportArgumentType]
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        lm_dataset["train"],  # pyright: ignore[reportArgumentType]
        collate_fn=data_collator,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # my_auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=100
    # )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # meta-llama/Llama-2-7b-chat-hf
    # model = Net().to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    init_start_event.record()  # pyright: ignore[reportCallIssue]
    for epoch in range(1, args.epochs + 1):
        # train(
        #     args,
        #     model,
        #     rank,
        #     world_size,
        #     train_loader,
        #     optimizer,
        #     epoch,
        # )

        test_v2(
            model=model,
            rank=rank,
            world_size=world_size,
            tokenizer=tokenizer,
            test_dataset=lm_dataset["test"],
            chat_template=chat_template,
            horizon=args.horizon_eval,
            top_k=args.top_k,
            num_beams=args.num_beams,
            eos_token=tokenizer.eos_token_id,
        )

    init_end_event.record()  # pyright: ignore[reportCallIssue]

    if rank == 0:
        init_end_event.synchronize()
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{model}")

    # if args.save_model:
    #     # use a barrier to make sure training is done on all ranks
    #     dist.barrier()
    #     states = model.state_dict()
    #     if rank == 0:
    #         torch.save(states, "mnist_cnn.pt")

    cleanup()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(42)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True
    )
