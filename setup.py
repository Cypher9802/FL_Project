from setuptools import setup, find_packages

setup(
    name="federated-learning-mobile",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "flower>=1.5.0",
        "opacus>=1.4.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "ucimlrepo>=0.0.3",
    ],
    python_requires=">=3.8",
    author="ML Research Team",
    description="Federated Learning for Mobile Fitness Tracking",
    long_description="A complete federated learning implementation for personalized fitness tracking on mobile devices with privacy preservation and mobile optimization.",
)
