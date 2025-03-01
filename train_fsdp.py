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

import os
import argparse
import functools
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import pdb
import sys


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

from data.gsm8k import load_gsm8k_data


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    printr(f"Taining started on rank {rank}", rank)
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(train_loader):
        # data, target = data.to(rank), target.to(rank)
        # batch is not a dict
        batch_dv = {
            k: v.to(rank) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        output = model(**batch_dv)
        loss = output.loss
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        printr(f"Loss (rank {rank}): {loss.item()}", rank)
        first_key = next(iter(batch_dv.keys()))
        ddp_loss[1] += len(batch_dv[first_key])

    print(f"Rank {rank} done with training, about to reduce")
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


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


def printr(msg: str, rank: int):
    if rank == 0:
        print(msg)


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    # )

    # dataset1 = datasets.MNIST(
    #     "../data", train=True, download=False, transform=transform
    # )
    # dataset2 = datasets.MNIST(
    #     "../data", train=False, download=False, transform=transform
    # )

    # sampler1 = DistributedSampler(
    #     dataset1, rank=rank, num_replicas=world_size, shuffle=True
    # )
    # sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    # train_kwargs = {"batch_size": args.batch_size, "sampler": sampler1}
    # test_kwargs = {"batch_size": args.test_batch_size, "sampler": sampler2}
    # cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    # train_kwargs.update(cuda_kwargs)
    # test_kwargs.update(cuda_kwargs)

    # train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    lm_dataset = load_gsm8k_data(tokenizer, input_seq_len=128)
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

    # if rank == 0:
    #     ForkedPdb().set_trace()

    # meta-llama/Llama-2-7b-chat-hf
    # model = Net().to(rank)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True
    )
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={type(model.model.layers[0])},
    )
    model = FSDP(
        model,
        auto_wrap_policy=llama_auto_wrap_policy,
        device_id=rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )

    printr(f"Model created on rank {rank}", rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    init_start_event.record()  # pyright: ignore[reportCallIssue]
    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
        )
        # test(model, rank, world_size, test_loader)

    init_end_event.record()  # pyright: ignore[reportCallIssue]

    if rank == 0:
        init_end_event.synchronize()
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128, help="input sequence length"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True
    )
