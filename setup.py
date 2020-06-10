
from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(

    name = "itembed",
    version = "0.4.2",
    packages = find_packages(),

    author = "Johan Berdat",
    author_email = "jojolebarjos@gmail.com",
    license = "MIT",

    url = "https://gitlab.com/jojolebarjos/itembed",
    project_urls = {
        "Documentation": "https://itembed.readthedocs.io/en/stable/",
        "Tracker": "https://gitlab.com/jojolebarjos/itembed/-/issues",
    },

    description = "word2vec for itemsets",
    long_description = long_description,
    long_description_content_type = "text/markdown",

    keywords = [
        "itemset",
        "word2vec",
        "embedding",
    ],

    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities",
    ],

    python_requires = ">=3.6",
    install_requires = [
        "numba>=0.34",
        "scipy>=0.16",
        "tqdm>=1.0",
    ],

)
