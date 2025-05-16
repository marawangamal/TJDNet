from typing import Union
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import DatasetDict
from dataloaders._base import BaseChatTemplate
from tjdnet.models.tjd import TJDGenerationConfig
from utils.accuracy import compute_accuracy


class TestAccuracyCallback(pl.Callback):
    """PyTorch Lightning callback to evaluate accuracy on test dataset at the end of each epoch."""

    def __init__(
        self,
        test_dataset: DatasetDict,
        generation_config: TJDGenerationConfig,
        chat_template: BaseChatTemplate,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        super().__init__()
        self.test_dataset = test_dataset
        self.generation_config = generation_config
        self.chat_template = chat_template
        self.accuracy = Accuracy(task="multiclass")
        self.tokenizer = tokenizer

    def setup(self, trainer, pl_module, stage=None):
        """Set up the test dataloader if not provided."""
        self.accuracy = self.accuracy.to(pl_module.device)

    def test_step(self, batch, batch_idx):

        input_ids, attention_mask = self.chat_template.format_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            tokenizer=self.tokenizer,
        )
        outputs, ardict = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        # Batched decoding
        y_pred = tokenizer.batch_decode(outputs)
        y_true = tokenizer.batch_decode(batch["labels"])
        # Compute accuracy
        correct_mask = [
            chat_template.check_answer(y_pred[b], y_true[b], tokenizer.eos_token)  # type: ignore
            for b in range(len(y_pred))
        ]

    # Log eval_acc

    # def on_epoch_start(self, trainer, pl_module):
    #     """Compute accuracy on test dataset at the end of each epoch."""
    #     # Skip if no test dataloader available
    #     if self.test_dataloader is None:
    #         return

    #     # Enable evaluation mode
    #     pl_module.eval()

    #     # Initialize variables for accuracy computation
    #     total_acc = 0.0

    #     # Disable gradient computation for evaluation
    #     with torch.no_grad():
    #         for batch in self.test_dataloader:
    #             # Handle different batch formats
    #             if isinstance(batch, list) or isinstance(batch, tuple):
    #                 x, y = batch
    #             else:
    #                 x, y = batch["input"], batch["target"]

    #             # Move to device if not already there
    #             if x.device != pl_module.device:
    #                 x = x.to(pl_module.device)
    #             if y.device != pl_module.device:
    #                 y = y.to(pl_module.device)

    #             # Get model predictions
    #             y_hat = pl_module(x)

    #             # Update accuracy
    #             self.accuracy.update(y_hat, y)

    #     # Compute final accuracy
    #     test_acc = self.accuracy.compute().item()

    #     # Log accuracy
    #     trainer.logger.log_metrics({"test_acc": test_acc}, step=trainer.global_step)

    #     # Print accuracy
    #     print(f"Epoch {trainer.current_epoch}: Test Accuracy = {test_acc:.4f}")

    #     # Reset accuracy for next epoch
    #     self.accuracy.reset()

    #     # Return to training mode
    #     pl_module.train()
