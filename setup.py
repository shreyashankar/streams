import pathlib

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="streams",
    version="0.1",
    description="Streaming Protocols for ML Benchmarks",
    long_description=README,
    long_description_content_type="text/markdown",
    author="shreyashankar",
    author_email="shreya@cs.stanford.edu",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "avalanche-lib",
        "cvxpy",
        "folktables",
        "kaggle",
        "matplotlib",
        "numpy",
        "pandas",
        "pre-commit",
        "pytest",
        "python-dotenv",
        "seaborn",
        "scipy",
        "wilds",
    ],
    include_package_data=True,
)
