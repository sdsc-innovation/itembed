
from setuptools import setup

setup(
    
    name = 'item2vec',
    version = '0.1',
    packages = ['item2vec'],
    
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
        'Development Status :: 3 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
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
