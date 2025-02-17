# for installing rwkv (inference) to a specific python environment
# it must be placed one level above the rwkv directory
# and run the following command:
# pip install -e .  --verbose

from setuptools import setup, find_packages

setup(
    name="rwkv",
    version="0.1",
    # packages=find_packages(),
    # only install the inference engine ("rwkv")
    packages=find_packages(include=["rwkv"]),
    install_requires=[],
    # py_modules=["__init__"],  # Add other top-level .py files here
)