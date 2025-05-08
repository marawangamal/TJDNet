from setuptools import setup, find_packages

setup(
    name="tjdnet",  # or whatever your project is called
    version="0.1.0",
    packages=find_packages(
        include=[
            "tjdnet*",  # your main library
            "jobrunner*",  # CLI package
            "utils*",  # etc.
            "dataloaders*",
            "jrun*",  # Add the new jrun package
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
        "networkx>=2.8.0",  # For dependency graphs
        "sqlite3>=2.6.0",  # For database (if not in standard library)
    ],
    entry_points={
        "console_scripts": [
            "jobrunner = jobrunner.main:main",  # Original entry point
            "jrun = jrun.cli:main",  # New entry point for the renamed package
        ],
    },
)
