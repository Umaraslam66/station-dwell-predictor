from setuptools import setup, find_packages

setup(
    name="station_dwell_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.2.3',
        'scikit-learn>=0.24.1',
        'plotly>=4.14.3',
    ],
)
