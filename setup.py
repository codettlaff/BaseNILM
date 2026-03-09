from setuptools import setup, find_packages

setup(
    name="dp-smart-meter",          # distribution name (can have dashes)
    version="0.1.0",
    description="Differentially Private Smart Meter project",
    package_dir={"": "src"},         # tell setuptools packages live in src/
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        # put your dependencies here later, e.g.:
        # "numpy",
        # "pandas",
        # "scipy",
    ],
)