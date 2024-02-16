# Installation

Python 3.7+ is required, as well as [NumPy](https://numpy.org/) and [Numba](https://numba.pydata.org/).
To install the latest stable version of `itembed`, the recommended source is [PyPI](https://pypi.org/project/itembed/):

```
pip install itembed
```

To install directly from source:

```
pip install git+https://github.com/jojolebarjos/itembed.git
```

After installation, you can verify by checking the version:

```py
import itembed

print(itembed.__version__)
```

As `itembed` relies on Numba for fast code generation, some of the performance tips provided in their documentation may apply.
In particular, Intel's short vector math library can be [installed](https://numba.readthedocs.io/en/stable/user/performance-tips.html#intel-svml) to increase performances.
