import argparse
import os
import os.path as osp

import torch


def main(args):
    # Load the experiment directory
    experiments = {}
    for exp in os.listdir(args.experiments):
        ckpt_path = osp.join(args.experiments, exp, "best.ckpt")
        if osp.isdir(ckpt_path):
            ckpt = torch.load(osp.join(ckpt_path, "meta.pt"), map_location="cpu")

            # Apply filter
            if args.filter is not None:
                hparams = ckpt["hyper_parameters"]
                if (
                    not args.filter_by in hparams
                    # or hparams[args.filter_by] != args.filter
                    # check if args.filter is substring of hparams[args.filter_by]
                    or args.filter not in hparams[args.filter_by]
                ):
                    continue

            # Add eval loss
            eval_loss = ckpt["callbacks"][
                "ModelCheckpoint{'monitor': 'eval_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"
            ]["best_model_score"]
            experiments[exp] = eval_loss

    # Find best model
    best_model = min(experiments, key=lambda k: experiments[k])

    # Tag the best model (add  empty file named '.best' in the best model directory)
    best_file_path = osp.join(args.experiments, best_model, ".best")
    with open(best_file_path, "w") as f:
        f.write(f"Best model: {best_model}\n")
        f.write(f"Eval loss: {experiments[best_model]}\n")
        f.write(f"Filter: {args.filter}\n")
        f.write(f"Filter by: {args.filter_by}\n")
        f.write(f"Tag: {args.tag}\n")

    # Remove .best from all other models in experiments (idempotentency)
    for exp in experiments:
        if exp != best_model:
            best_file_path = osp.join(args.experiments, exp, ".best")
            if osp.exists(best_file_path):
                os.remove(best_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag the best model")
    parser.add_argument(
        "--experiments",
        type=str,
        default="experiments",
        help="Path to the experiments directory",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter value",
    )
    parser.add_argument(
        "--filter_by",
        type=str,
        default="group_id",
        help="Filter attribute",
    )
    args = parser.parse_args()
    main(args)
