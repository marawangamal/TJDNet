import argparse
import lightning as L

from utils.lmodules_v2 import LModel, LDataModule
from utils.lightning_callbacks.generate import GenerateCallback


def train(args: argparse.Namespace) -> None:
    L.seed_everything(42)
    model = LModel(
        model=args.model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        epochs=args.epochs,
        horizon=args.horizon,
        rank=args.rank,
        hidden_dim=args.hidden_dim,
        experiment_name=args.experiment_name,
    )
    data = LDataModule(
        model=args.model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_num_samples=None,
        dataset=args.dataset,
    )
    trainer = L.Trainer(max_epochs=args.epochs, callbacks=[GenerateCallback()])
    trainer.fit(model, datamodule=data)
    trainer.save_checkpoint(args.checkpoint)


def test(args: argparse.Namespace) -> None:
    model = LModel.load_from_checkpoint(args.ckpt)
    data = LDataModule(
        model=model.hparams["model"],
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dataset=args.dataset or model.hparams.get("dataset", "stemp"),
    )
    model.hparams.update(vars(args))
    trainer = L.Trainer(callbacks=[GenerateCallback()])
    trainer.test(model, datamodule=data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal train/test script")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("--model", default="gpt2")
    train_p.add_argument("--dataset", default="stemp")
    train_p.add_argument("--batch_size", type=int, default=32)
    train_p.add_argument("--seq_len", type=int, default=128)
    train_p.add_argument("--epochs", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--warmup_steps", type=int, default=100)
    train_p.add_argument("--horizon", type=int, default=2)
    train_p.add_argument("--rank", type=int, default=2)
    train_p.add_argument("--hidden_dim", type=int, default=128)
    train_p.add_argument("--experiment_name", default="main_v2")
    train_p.add_argument("--checkpoint", default="model.ckpt")

    test_p = sub.add_parser("test", help="Evaluate a checkpoint")
    test_p.add_argument("ckpt")
    test_p.add_argument("--dataset", default=None)
    test_p.add_argument("--batch_size", type=int, default=32)
    test_p.add_argument("--seq_len", type=int, default=128)
    test_p.add_argument("--max_new_tokens", type=int, default=32)
    test_p.add_argument("--top_k", type=int, default=1)
    test_p.add_argument("--do_sample", action="store_true")
    test_p.add_argument(
        "--gen_mode",
        default="draft",
        choices=["draft", "base", "speculative"],
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
