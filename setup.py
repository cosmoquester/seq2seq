from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="seq2seq",
    version="0.0.1",
    description="This is seq2seq model structures with Tensorflow 2.X",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=["tensorflow>=2"],
    url="https://github.com/psj8252/seq2seq.git",
    author="Park Sangjun",
    keywords=["seq2seq", "rnn", "attention", "transformer"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
    ],
    packages=find_packages(exclude=["tests"]),
)
