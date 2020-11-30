from setuptools import find_packages, setup

setup(
    name="seq2seq",
    version="0.0.1",
    description="This is seq2seq model structures with Tensorflow 2.",
    python_requires=">=3.6",
    install_requires=["tensorflow>=2"],
    url="https://github.com/psj8252/seq2seq.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
)
