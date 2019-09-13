
from setuptools import find_packages, setup

setup(
    
    name = 'item2vec',
    version = '0.1.1',
    packages = find_packages(),
    
    author = 'Jojo le Barjos',
    author_email = 'jojolebarjos@gmail.com',
    license_file = 'LICENSE',
    
    description = 'word2vec for itemsets',
    
    keywords = [
        'itemset',
        'word2vec',
        'embedding'
    ],
    
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Freely Distributable',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Linguistic'
    ],
    
    install_requires = [
        'numba',
        'numpy',
        'tqdm'
    ]
    
)
