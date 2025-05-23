from setuptools import setup, find_packages

setup(
    name="tjdnet",  # or whatever your project is called
    version="1.0.0",
    packages=find_packages(
        include=[
            "tjdnet*",  # your main library
            "utils*",  # etc.
            "dataloaders*",
        ]
    ),
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
        "pyyaml>=6.0",  # For YAML configuration
        "pandas>=1.5.0",  # For data handling
        "matplotlib>=3.5.0",  # For visualization
    ],
)
