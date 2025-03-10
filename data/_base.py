import os
import os.path as osp
import tempfile

ROOT_DIR = "/home/mila/m/marawan.gamal/scratch/hf_cache"


def setup():
    os.environ["HF_HOME"] = ROOT_DIR
    os.environ["HF_DATASETS_CACHE"] = osp.join(ROOT_DIR, "datasets")
    os.environ["HF_MODULES_CACHE"] = osp.join(ROOT_DIR, "modules")
    os.environ["TRANSFORMERS_CACHE"] = osp.join(ROOT_DIR, "transformers")

    # Set TMPDIR to a directory in the scratch space
    os.environ["TMPDIR"] = osp.join(ROOT_DIR, "tmpdir")
    tempfile.tempdir = os.environ["TMPDIR"]
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
