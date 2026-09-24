"""
Setup configuration for the AI-Powered 3D Medical Image Reconstruction
and Disease Analysis project.

Project ID: SKIT/DS/2023-2027/CSE-F-07
"""
from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ai-3d-medical-imaging",
    version="0.1.0",
    description=(
        "AI-Powered 3D Medical Image Reconstruction and Disease Analysis — "
        "SKIT/DS/2023-2027/CSE-F-07"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mohd Rafey, Naman Verma, Mayuri Agarwal, Mohit Chaudhary",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "tensorflow>=2.13.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "nibabel>=5.1.0",
        "medmnist>=2.2.3",
        "PyYAML>=6.0.1",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
