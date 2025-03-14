#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="disco-musica",
    version="0.1.0",
    description="An open-source multimodal AI music generation application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Disco Musica Team",
    author_email="info@discomusica.ai",
    url="https://github.com/disco-musica/disco-musica",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Conversion",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "disco-musica=main:main",
        ],
    },
    keywords="music generation, AI, deep learning, audio processing, midi",
)