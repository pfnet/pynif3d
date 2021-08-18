import runpy
from distutils.core import setup

from setuptools import find_packages

__version__ = runpy.run_path("pynif3d/__init__.py")["__version__"]

minimum_requirements = [
    "torch>=1.6.0",
    "scikit-build",
    "opencv-python>=3.4.0",
    "setuptools>=51.0.0",
    "numpy>=1.18.5",
    "PyYAML>=5.3.1",
    "Cython>=0.29",
    "gdown>=3.10.0",
    "m2r2==0.2.7",
    "mistune==0.8.4",
    "torchvision",
]

develop_requirements = [
    "black==19.3.b0",
    "flake8-comprehensions==3.3.0",
    "flake8-bugbear==20.1.4",
    "flake8==3.8.4",
    "isort==4.3.21",
    "m2r2",
    "mccabe==0.6.1",
    "mock",
    "sphinx",
    "sphinx_markdown_tables",
    "sphinx_rtd_theme",
]

examples_requirements = [
    "argparse>=1.4.0",
    "tqdm>=4.42.0",
    "tensorboard>=2.0.0",
    "imageio",
]

setup(
    name="pynif3d",
    version=__version__,
    url="https://github.com/pfnet/pynif3d",
    author="Woven Core, Inc.",
    description="PyTorch-based library for NIF-based 3D geometry representation",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.6.0",
    install_requires=minimum_requirements,
    extras_require={
        "develop": develop_requirements,
        "examples": examples_requirements,
    },
)
