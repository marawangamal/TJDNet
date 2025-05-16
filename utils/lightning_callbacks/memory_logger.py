import torch
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class CUDAMemoryLogger(Callback):
    @rank_zero_only  # ensure one-time print in DDP/FSDP
    def _print_memory(self, msg: str):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"[{msg}] allocated: {alloc:>7.0f} GB │ reserved: {reserved:>7.0f} GB │ peak: {peak:>7.0f} GB"
            )
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx in [0, 10, 20, 30, 40]:
            self._print_memory(f"batch {batch_idx} (before forward)")
