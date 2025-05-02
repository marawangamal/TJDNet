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
    ],
    entry_points={
        "console_scripts": [
            "jobrunner = jobrunner.main:main",  # command → package.module:function
        ],
    },
)
