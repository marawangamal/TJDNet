import os
import json
import glob
from collections import defaultdict
import pandas as pd

CKPT_DIR = "checkpoints"


def main():
    """Loop through checkpoints and aggregate results from checkponts/*/eval_converged_metrics.json

    Example:
    cat checkpoints/e20_bs64_sl128_l0_001_ws100_gcv1_0_mgpt2_mhcp_hd128_r1_h2_pfexp_imrandom_tmlora_lr32_umelTrue_he2_mnt128_tk200_ussFalse_dstemp_lsepoch_lo1_esepoch_ev1_gsepoch_ge1000_mns10000_eoFalse_caTrue_abs1_wie87841a4/eval_converged_metrics.json
    >> {"eval_loss": 12.552762985229492, "eval_nll": 12.552765846252441, "eval_acc": 1.0}

    """
    # Get all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(CKPT_DIR, "*"))

    # Dictionary to store results
    results = []

    # Loop through each checkpoint directory
    for ckpt_dir in checkpoint_dirs:
        if not os.path.isdir(ckpt_dir):
            continue

        # Path to the metrics file
        metrics_file = os.path.join(ckpt_dir, "eval_converged_metrics.json")

        # Check if metrics file exists
        if not os.path.exists(metrics_file):
            print(f"Metrics file not found in {ckpt_dir}")
            continue

        # Read the metrics file
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            # Add the checkpoint name to the metrics
            ckpt_name = os.path.basename(ckpt_dir)
            metrics["checkpoint"] = ckpt_name

            # Add to results
            results.append(metrics)

        except json.JSONDecodeError:
            print(f"Error parsing JSON in {metrics_file}")
        except Exception as e:
            print(f"Error processing {metrics_file}: {str(e)}")

    # Convert results to DataFrame for easier analysis
    if results:
        df = pd.DataFrame(results)

        # Print summary statistics
        print(f"Found metrics for {len(results)} checkpoints")
        print("\nSummary statistics:")
        print(df.describe())

        # Save results to CSV for further analysis
        output_dir = "results/gpt2_cp_grid"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "summary.csv")
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")

        return df
    else:
        print("No metrics found in any checkpoint directory")
        return None


if __name__ == "__main__":
    main()
