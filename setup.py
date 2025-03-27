from setuptools import setup

setup(
    name="tjdnet",
    version="1.0",
    packages=["tjdnet", "data", "utils"],
    install_requires=[
        "torch>=2.0.0",
        "tqdm>=4.66.4",
        "transformers>=4.40.1",
        "hydra-core>=1.3.2",
        "pytorch-lightning>=2.2.4",
        "tensorboardX>=2.6.2.2",
        "pydantic>=2.7.1",
        "datasets>=1.14.0",
        "tensorboard>=2.6.0",
    ],
)
