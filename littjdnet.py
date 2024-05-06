import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorch_lightning import LightningModule

from TJDNet import TJDNet


logger = logging.getLogger(__name__)

# Configure logging
fmt = "[%(levelname)s] - %(message)s"
logging.basicConfig(
    format=fmt,
    level=logging.INFO,
)


class LitTJDNet(LightningModule):
    def __init__(
        self,
        model_name: str,
        tjd_config: dict,
        lr: float = 5e-5,
    ):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = TJDNet(
            model,
            condition_func=tjd_config["condition_func"],
            replacement_func=tjd_config["replacement_func"],
        )
        model.replace_base_model_layers()
        self.model = model
        self.lr = lr

        # Ensure the tokenizer has a pad token
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            logger.info(
                "Tokenizer does not have a pad token set. Setting pad_token to eos_token."
            )
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        # self.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 5 == 0:  # don't make the tf file huge

            for k, v in self.named_parameters():
                self.logger.experiment.add_histogram(  # type: ignore
                    f"{k}_grad", values=v.grad, global_step=self.trainer.global_step
                )

            for k, v in self.named_parameters():
                self.logger.experiment.add_histogram(  # type: ignore
                    k, values=v.data, global_step=self.trainer.global_step
                )

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, prog_bar=False)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_end(self) -> None:
        logger.info("Generating text sample...")
        prompt = "The meaning of life is"
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)  # type: ignore
        sample_outputs = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=50,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        generated_text = self.tokenizer.decode(
            sample_outputs[0], skip_special_tokens=True
        )
        logger.info(f"Generated text:\n{generated_text}\n")
