# import torch
# from lightning.pytorch.callbacks import Callback
# from pytorch_lightning.utilities import rank_zero_only


# class GenerateCallback(Callback):
#     def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#         if batch_idx in [0, 10, 20, 30, 40]:
#             self._print_memory(f"batch {batch_idx} (before forward)")
