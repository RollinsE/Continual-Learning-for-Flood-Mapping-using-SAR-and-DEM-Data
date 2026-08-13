from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="flood-extent-mapping",
    version="0.15.19",
    description="Flood extent mapping with Sentinel-1 SAR and DEM data, including training, evaluation, continual learning and deployment",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Rollins Edeh",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    keywords=["flood mapping", "Sentinel-1", "semantic segmentation", "remote sensing", "continual learning"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    project_urls={
        "Original MMFlood project": "https://github.com/edornd/mmflood",
        "MMFlood paper": "https://doi.org/10.1109/ACCESS.2022.3205419",
    },
    entry_points={
        "console_scripts": [
            "floodmap=floods.cli:main",
        ]
    },
)
