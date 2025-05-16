from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from tjdnet.models.tjd import TJD, TJDGenerationConfig


class GenerateCallback(Callback):
    def __init__(self, prompt: str = "What is 20°C in Fahrenheit?"):
        self.prompt = prompt

    @rank_zero_only
    def _generate(self, pl_module: LightningModule) -> None:
        # Generate sample text
        model: TJD = pl_module.model
        output, _ = model.generate(
            input_ids=pl_module.tokenizer.encode(self.prompt, return_tensors="pt").to(
                model.device
            ),
            generation_config=TJDGenerationConfig(
                max_new_tokens=128,
                do_sample=True,
                top_k=200,
            ),
        )
        generated_text = pl_module.tokenizer.decode(output[0])
        # linesep = "\n" + "-" * 80 + "\n"
        # print(f"{linesep}\nGenerated text:\n{generated_text}\n{linesep}")
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

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._generate(pl_module)
