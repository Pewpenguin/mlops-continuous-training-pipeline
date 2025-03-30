from setuptools import setup, find_packages

# Define development dependencies
extras_require = {
    'dev': [
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'black>=23.3.0',
        'isort>=5.12.0',
        'flake8>=6.0.0',
        'mypy>=1.3.0',
        'wheel>=0.42.0',
        'setuptools>=69.0.0'
    ]
}

setup(
    name="mlopps",
    version="0.1.0",
    description="MLOps Pipeline Package for Machine Learning Operations",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['pipelines*', 'api*', 'monitoring*']),
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
    extras_require=extras_require,
    package_data={
        "mlopps": [
            "*.json", "*.yml", "*.yaml",
            "config/*.json", "config/*.yml", "config/*.yaml",
            "api/templates/*.html",
            "monitoring/templates/*.html",
            "examples/*.json"
        ]
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        'console_scripts': [
            'mlopps=mlopps.cli:main',
            'mlopps-api=mlopps.api.app:main',
            'mlopps-monitor=mlopps.monitoring.data_drift:main'
        ],
    }
)