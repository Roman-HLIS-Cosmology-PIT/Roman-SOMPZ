import pytest
import sys

from tests.helpers.utilities import compare_table_dicts, check_deps
from tables_io.lazy_modules import (
    tables,
    apTable,
    apDiffUtils,
    fits,
    h5py,
    pd,
    pq,
    jnp,
    lazyImport,
)


@pytest.mark.parametrize(
    "mod",
    [tables, apTable, apDiffUtils, fits, h5py, pd, pq],
)
def test_deps(mod):
    """Testing that dependencies get loaded with Lazy Import"""
    assert check_deps([mod])


@pytest.mark.skipif(not check_deps([jnp]), reason="Failed to load jax.numpy")
def test_deps_jnp():
    """Testing Jax Numpy dependencies"""
    assert check_deps([jnp])


def test_check_deps():
    """Testing the Lazy Import functionality: fails on importing module that doesn't exist"""
    bad_module = lazyImport("this_does_not_exist")
    assert not check_deps([bad_module])


def test_lazy_load():
    """Test that the lazy import works"""
    noModule = lazyImport("thisModuleDoesnotExist")
    try:
        noModule.d
    except ImportError:
        pass
    else:
        raise ImportError("lazyImport failed")


@pytest.mark.skipif("wave" in sys.modules, reason="Wave module already imported")
def test_lazy_load2():
    """A second test that the lazy import works"""
    # I picked an obscure python module that is unlikely
    # to be loaded by anything else.

    wave = lazyImport("wave")
    # should not be loaded yet
    assert "wave" not in sys.modules

    # should trigger load
    assert "WAVE_FORMAT_PCM" in dir(wave)
    assert wave.sys == sys

    assert "wave" in sys.modules
