from setuptools import setup, find_packages

setup(
    name="AlphaTools",
    version="0.1.0",
    packages=['AlphaTools'],
    python_requires=">=3.8",
    author="taint",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "statsmodels",
        "ta",
        "jupyter",
        "tqdm",
        "SQLAlchemy",
        "optuna",
    ],
    author_email="thetai2382002@gmail.com",
)