import os.path as osp
import argparse
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer
from TJDNet import TTDist, sample_from_tensor_dist, batched_index_select
from utils.utils import get_experiment_name


class BasicTJDLayerOutput:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss


class BasicTJDLayer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_rank: int,
        norm_method: str,
        seq_len: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = torch.nn.Parameter(torch.randn(1, model_rank))
        self.beta = torch.nn.Parameter(torch.randn(1, model_rank))
        self.core = torch.nn.Parameter(
            torch.randn(1, model_rank, vocab_size, model_rank)
        )
        self.norm_method = norm_method
        self.seq_len = seq_len

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

    def materialize(self):
        prob_unnorm = (
            TTDist(
                self.alpha,
                self.beta,
                self.core,
                repeat_batch_size=1,
                norm_method=self.norm_method,
                n_core_repititions=self.seq_len,
            )
            .materialize()
            .squeeze()
        )
        return prob_unnorm / prob_unnorm.sum()


class FJDLayer(torch.nn.Module):
    def __init__(
        self, vocab_size: int, seq_len: int, norm_method: str, *args, **kwargs
    ):
        super().__init__()
        self.norm_method = norm_method
        self.prob_dist_unnorm = torch.nn.Parameter(torch.randn([vocab_size] * seq_len))
        self.precond_func = {
            "relu": torch.relu,
            "abs": torch.abs,
            "sigmoid": torch.sigmoid,
            "softmax": torch.exp,
        }[self.norm_method]

    def forward(self, target: torch.Tensor):
        """Compute the loss for the model.

        Args:
            target (torch.Tensor): Shape: (batch_size, seq_len)

        Returns:
            BasicTJDLayerOutput: Object containing the loss value
        """
        precond_applied = self.precond_func(self.prob_dist_unnorm)
        batch_unnorm_probs = batched_index_select(
            precond_applied, target
        )  # (batch_size,)
        norm_consts = precond_applied.sum()  # Shape: (1,)
        loss = (-torch.log(batch_unnorm_probs) + torch.log(norm_consts)).mean()
        return BasicTJDLayerOutput(loss)

    def materialize(self):
        precond_applied = self.precond_func(self.prob_dist_unnorm)
        return precond_applied / precond_applied.sum()


class Litmodel(LightningModule):
    def __init__(
        self,
        model_rank,
        vocab_size,
        true_dist: torch.Tensor,
        lr=1e-3,
        norm_method="relu",
        model_name="tjd",
        seq_len=1,
    ):
        super().__init__()
        self.true_dist = true_dist
        self.model_cls = {"tjd": BasicTJDLayer, "fjd": FJDLayer}[model_name]
        self.model = self.model_cls(
            model_rank=model_rank,
            vocab_size=vocab_size,
            norm_method=norm_method,
            seq_len=seq_len,
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

        # Compute the KL divergence between the true and estimated distributions
        estimated_dist = self.model.materialize()  # P(d1, d2, ..., dN)
        self.true_dist = self.true_dist.to(estimated_dist.device)

        assert (
            abs(estimated_dist.sum() - 1) < 1e-6
        ), "Estimated distribution is not normalized"
        assert (
            abs(self.true_dist.sum() - 1) < 1e-6
        ), "True distribution is not normalized"

        kl_div_abs = torch.abs(
            torch.nn.functional.kl_div(
                torch.log(estimated_dist), self.true_dist, reduction="batchmean"
            )
        )
        p_hat_x = batched_index_select(estimated_dist, batch["target"])
        p_x = batched_index_select(self.true_dist, batch["target"])
        loss_diff = -torch.log(p_hat_x).mean() + torch.log(p_x).mean()
        loss_diff = loss_diff if loss_diff.item() > 0 else 0
        self.log("loss_diff", loss_diff, on_step=False, prog_bar=False)
        self.log("kl_div_abs", kl_div_abs, on_step=False, prog_bar=False)
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


def main(
    n_epochs: int = 100,
    batch_size: int = 8,
    n_train_samples: int = 8 * 100,
    n_test_samples: int = 8 * 10,
    true_rank: int = 4,
    model_rank: int = 2,
    vocab_size: int = 4,
    seq_len: int = 1,
    lr: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    norm_method: str = "relu",  # relu, abs, sigmoid, softmax
    model_name: str = "fjd",  # tjd, fjd
):

    experiment_conf = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_train_samples": n_train_samples,
        "n_test_samples": n_test_samples,
        "true_rank": true_rank,
        "model_rank": model_rank,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "norm_method": norm_method,
        "model_name": model_name,
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
    true_dist_repeated = true_ttdist.materialize().squeeze()
    true_dist = true_dist_repeated[0]
    true_dist = true_dist / true_dist.sum()  # P(d1, d2, ..., dN)

    # Sample `batch_size` random samples from the true distribution
    train_samples = sample_from_tensor_dist(true_dist, n_train_samples)
    train_dataset = torch.utils.data.TensorDataset(train_samples)  # type: ignore
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  # type: ignore

    # Sample `batch_size` random samples from the true distribution
    test_samples = sample_from_tensor_dist(true_dist, n_test_samples)
    test_dataset = torch.utils.data.TensorDataset(test_samples)  # type: ignore
    test_dataloader = torch.utils.data.DataLoader(  # type: ignore
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    lit_model = Litmodel(
        vocab_size=vocab_size,
        model_rank=model_rank,
        lr=lr,
        norm_method=norm_method,
        true_dist=true_dist,
        seq_len=seq_len,
        model_name=model_name,
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--norm_method", type=str, default="relu", help="Normalization method"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=2, help="Sequence length")
    parser.add_argument("--model_name", type=str, default="fjd", help="Model name")

    args = parser.parse_args()

    main(
        norm_method=args.norm_method,
        lr=args.lr,
        seq_len=args.seq_len,
        model_name=args.model_name,
    )
