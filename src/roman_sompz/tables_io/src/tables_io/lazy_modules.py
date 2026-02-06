""" Lazy loading modules """

import importlib


class LazyModule:
    """
    A partial implementation of a lazily imported module.

    This isn't a general solution, since it doesn't actually
    create a module-like object, but the LazyLoader solution
    in importlib seems not to cope with submodules like astropy.table,
    or pyarrow.parquet, and always imports the full module.
    """

    def __init__(self, name):
        self.name = name
        self._module = None

    @property
    def module(self):
        if self._module is not None:
            return self._module
        try:
            self._module = importlib.import_module(self.name)
        except ImportError as err:
            raise ImportError(
                f"Cannot use selected data format, {self.name} not available"
            ) from err

        return self._module

    def __dir__(self):
        """
        Get the attributes of the module, to support tab-autocomplete.
        """
        return dir(self.module)

    def __getattr__(self, item):
        """Get something from the module"""
        return getattr(self.module, item)


def lazyImport(modulename):
    """
    Lazily load a module
    """
    return LazyModule(modulename)


tables = lazyImport("tables")
"""The Tables Module"""
apTable = lazyImport("astropy.table")
"""The Astropy Tables Module"""
apDiffUtils = lazyImport("astropy.utils.diff")
"""The Astropy Utils Diff Module"""
fits = lazyImport("astropy.io.fits")
"""The Astropy FITS module"""
h5py = lazyImport("h5py")
"""The H5PY Module"""
pa = lazyImport("pyarrow")
"""The PyArrow Module"""
pd = lazyImport("pandas")
"""The Pandas Module"""
pq = lazyImport("pyarrow.parquet")
"""The PyArrow Parquet Module"""
ds = lazyImport("pyarrow.dataset")
"""The PyArrow Dataset Module"""
jnp = lazyImport("jax.numpy")
"""The JAX Numpy Module"""
