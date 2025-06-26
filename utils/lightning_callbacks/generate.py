from typing import Any
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from tjdnet.models.tjd import TJD


class GenerateCallback(Callback):
    def __init__(self, prompt: str = "What is 20°C in Fahrenheit?"):
        self.prompt = prompt
        self.generate_called = False

    @rank_zero_only
    def _generate(self, pl_module: LightningModule) -> None:
        device = next(pl_module.parameters()).device
        # Generate sample text
        model: TJD = pl_module.model
        output = model.generate(
            input_ids=pl_module.tokenizer.encode(self.prompt, return_tensors="pt").to(
                device
            ),
            max_new_tokens=128,
            do_sample=True,
            top_k=200,
            eos_token_id=pl_module.tokenizer.eos_token_id,
            pad_token_id=pl_module.tokenizer.eos_token_id,
        )
        generated_text = pl_module.tokenizer.decode(output[0])
        line = "─" * 80
        summary = (
            f"\n{line}\n"
            f"📊  SAMPLE \n"
            f"{line}\n"
            f"▶ Prompt      : {self.prompt}\n"
            f"▶ Generation  : {generated_text}\n"
            f"{line}\n"
        )
        print(summary)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._generate(pl_module)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self.generate_called:
            self.generate_called = True
            self._generate(pl_module)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._generate(pl_module)
