from setuptools import setup, find_packages

setup(
    name="mlopps",
    version="0.1.0",
    packages=find_packages(include=['mlopps', 'mlopps.*', 'pipelines', 'pipelines.*', 'api', 'api.*', 'monitoring', 'monitoring.*']),
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.2",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "mlflow==2.6.0",
        "dvc==3.15.0",
        "dvc-s3==2.23.0",
        "fastapi==0.98.0",
        "uvicorn==0.23.1",
        "pydantic==1.10.12",
        "evidently==0.4.5",
        "pandera==0.15.1",
        "pytest==7.4.0",
        "pytest-cov==4.1.0",
        "python-dotenv==1.0.0",
        "tqdm==4.65.0",
        "joblib==1.3.1"
    ],
    python_requires=">=3.8",
    package_data={
        "mlopps": ["*.json", "*.yml", "*.yaml", "config/*.json", "config/*.yml", "config/*.yaml"]
    },
    include_package_data=True
)