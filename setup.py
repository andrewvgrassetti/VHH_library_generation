from setuptools import setup, find_packages

setup(
    name="vhh_library_generation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "biopython>=1.81",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "fair-esm>=2.0.0",
        "nanomelt>=1.3.0",
    ],
    python_requires=">=3.8",
    description="VHH Biosimilar Library Generator",
    author="VHH Library Team",
)
