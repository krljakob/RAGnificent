from setuptools import find_packages, setup

setup(
    name="RAGnificent",
    version="0.1.0",
    packages=find_packages(),
    package_data={"RAGnificent": ["*.pyi"]},
    install_requires=[
        "requests>=2.32.3",
        "beautifulsoup4>=4.13.4",
        "pytest>=8.3.5",
        "markdownify>=1.1.0",
        "pytest-mock>=3.14.0",
        "numpy>=1.26.0",
    ],
)
