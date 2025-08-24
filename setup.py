from setuptools import setup, find_packages

setup(
    name="analytical_services",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "yfinance",  # for stock data
        "ccxt",      # for cryptocurrency data
        "requests",  # for API calls
        "scipy",     # for statistical analysis
        "matplotlib",# for plotting
        "scikit-learn", # for machine learning
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular analytics package for stocks, sports, forex, and cryptocurrency analysis",
    python_requires=">=3.8",
)
