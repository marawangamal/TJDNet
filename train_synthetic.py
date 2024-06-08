import os.path as osp
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer
from TJDNet import TTDist, sample_from_tensor_dist
from utils.utils import get_experiment_name


class BasicTJDLayerOutput:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss


class BasicTJDLayer(torch.nn.Module):
    def __init__(
        self, vocab_size: int, model_rank: int, norm_method: str, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = torch.nn.Parameter(torch.randn(1, model_rank))
        self.beta = torch.nn.Parameter(torch.randn(1, model_rank))
        self.core = torch.nn.Parameter(
            torch.randn(1, model_rank, vocab_size, model_rank)
        )
        self.norm_method = norm_method

    def forward(self, target: torch.Tensor):
        ttdist = TTDist(
            self.alpha,
            self.beta,
            self.core,
            target.size(1),
            repeat_batch_size=target.size(0),
            norm_method=self.norm_method,
        )
        probs_tilde, norm_constant = ttdist.get_prob_and_norm(target)
        loss = (-torch.log(probs_tilde) + torch.log(norm_constant)).mean()
        return BasicTJDLayerOutput(loss)


class Litmodel(LightningModule):
    def __init__(self, model_rank, vocab_size, lr=1e-3, norm_method="relu"):
        super().__init__()
        self.model = BasicTJDLayer(
            model_rank=model_rank, vocab_size=vocab_size, norm_method=norm_method
        )
        self.lr = lr

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.trainer.global_step % 5 == 0:  # don't make the tf file huge
            for k, v in self.named_parameters():
                if v.grad is not None:  # Check if the gradient exists
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


def collate_fn(batch):
    target = [item[0] for item in batch]
    return {"target": torch.stack(target)}


def normalize_matrix(matrix):
    """Placeholder normalization function, replace with the actual implementation."""
    norm_factor = torch.norm(matrix, dim=0, keepdim=True)
    return matrix / norm_factor


def make_batched_alpha_beta_core(alpha, beta, core, batch_size):
    alpha = alpha.repeat(batch_size, 1)
    beta = beta.repeat(batch_size, 1)
    core = core.repeat(batch_size, 1, 1, 1)
    return alpha, beta, core


def main():
    n_epochs = 10
    batch_size = 8
    n_train_samples = 8 * 100
    n_test_samples = 8 * 10
    true_rank = 4
    model_rank = 2
    vocab_size = 4
    seq_len = 1
    lr = 1e-4
    checkpoint_dir = "checkpoints"
    norm_method = "softmax"  # relu, abs, sigmoid, softmax

    experiment_conf = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "true_rank": true_rank,
        "model_rank": model_rank,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "norm_method": norm_method,
    }
    experiment_name = get_experiment_name(experiment_conf)

    true_alpha = torch.randn(1, true_rank)
    true_beta = torch.randn(1, true_rank)
    true_core = torch.randn(1, true_rank, vocab_size, true_rank)
    true_ttdist = TTDist(
        true_alpha,
        true_beta,
        true_core,
        seq_len,
        repeat_batch_size=batch_size,
        norm_method=norm_method,
    )
    true_dist = true_ttdist.materialize().squeeze()
    true_dist = true_dist / true_dist.sum()  # P(d1, d2, ..., dN)

    # Sample `batch_size` random samples from the true distribution
    train_samples = sample_from_tensor_dist(true_dist[0], n_train_samples)
    train_dataset = torch.utils.data.TensorDataset(train_samples)  # type: ignore
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  # type: ignore

    # Sample `batch_size` random samples from the true distribution
    test_samples = sample_from_tensor_dist(true_dist[0], n_test_samples)
    test_dataset = torch.utils.data.TensorDataset(test_samples)  # type: ignore
    test_dataloader = torch.utils.data.DataLoader(  # type: ignore
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    lit_model = Litmodel(
        vocab_size=vocab_size, model_rank=model_rank, lr=lr, norm_method=norm_method
    )

    tb_logger = TensorBoardLogger(
        osp.join(checkpoint_dir, experiment_name), name="", version=""
    )
    trainer = Trainer(
        max_epochs=n_epochs,
        log_every_n_steps=1,
        logger=tb_logger,
        gradient_clip_val=1.0,
    )
    trainer.fit(lit_model, train_dataloader, test_dataloader, ckpt_path="last")


if __name__ == "__main__":
    main()
