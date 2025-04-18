def migrate_status_table(cache_file):
    """Migrate old status table to add created_at column if it doesn't exist."""

    outpath = osp.expanduser(cache_file)
    if not osp.exists(outpath):
        print("No status table found. Nothing to migrate.")
        return

    try:
        # Load the existing status table
        status_table = pd.read_csv(outpath)

        # Check if created_at column already exists
        if "created_at" not in status_table.columns:
            print("Migrating status table to add created_at column...")

            # Add an empty created_at column
            status_table["created_at"] = ""

            # Save the updated table
            status_table.to_csv(outpath, index=False)
            print(f"Migration completed. Added created_at column to {outpath}")
        else:
            print("Status table already has created_at column. No migration needed.")

    except Exception as e:
        print(f"Error migrating status table: {e}")


if __name__ == "__main__":

    import argparse
    import os.path as osp
    import pandas as pd

    parser = argparse.ArgumentParser(description="Migrate status table.")
    parser.add_argument(
        "--dev", action="store_true", help="Use dev cache file instead of prod."
    )

    args = parser.parse_args()

    cache_file = (
        "~/.jobrunner/jobrunner_status_table.csv"
        if not args.dev
        else "~/.jobrunner/jobrunner_status_table_dev.csv"
    )

    # Migrate status table if needed
    migrate_status_table(cache_file)
