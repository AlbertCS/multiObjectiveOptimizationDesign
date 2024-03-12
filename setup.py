from setuptools import find_packages, setup

setup(
    name="MultiObjectiveOptimization",
    version="0.0.1",
    author="Albert",
    author_email="acanella@bsc.es",
    description="MultiObjective Optimization for protein design",
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package requires
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
