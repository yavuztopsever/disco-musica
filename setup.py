from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="disco-musica",
    version="0.1.0",
    author="Disco Musica Team",
    author_email="your.email@example.com",
    description="Open-source multimodal AI music generation application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/disco-musica",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)