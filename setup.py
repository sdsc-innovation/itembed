
from setuptools import find_packages, setup


with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()

setup(

    name = 'itembed',
    version = '0.4.1',
    packages = find_packages(),

    author = 'Johan Berdat',
    author_email = 'jojolebarjos@gmail.com',
    license = 'MIT',

    url = 'https://gitlab.com/jojolebarjos/itembed',

    description = 'word2vec for itemsets',
    long_description = long_description,
    long_description_content_type = 'text/markdown',

    keywords = [
        'itemset',
        'word2vec',
        'embedding',
    ],

    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Utilities',
    ],

    install_requires = [
        'numba',
        'numpy',
        'tqdm',
    ]

)
