import os
import pkg_resources
from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    install_requires = f.read().split("\n")

setup(
    name="clip_onnx",
    version="1.2",
    py_modules=["clip_onnx, clip"],
    description="",
    author="Maxim Gerasimov",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True
)
