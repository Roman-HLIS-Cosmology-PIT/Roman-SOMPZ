"""
Utilities for testing
"""
import sys
import numpy as np
from tables_io.lazy_modules import LazyModule, tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp


def check_deps(deps=None):
    missing = False
    if deps is None:  # pragma: no cover
        deps = [tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp]
    for mod in deps:
        if not isinstance(mod, LazyModule):
            sys.stderr.write(f"{mod} is not LazyModule: {type(mod)}")
            missing = True
            continue
        try:
            _ = dir(mod)
        except Exception as err:  # pylint: disable=broad-exception-caught
            sys.stderr.write(f"Missing {mod.name} because {err}")
            missing = True
    return not missing


def compare_tables(t1, t2, columns=None):
    """Compare all the tables in two `astropy.table.Table`)

    Parameters
    ----------
    t1 : `astropy.table.Table`
       One table
    t2 : `astropy.table.Table`
       Another tables

    Returns
    -------
    identical : `bool`
        True if the tables are identical, False otherwise

    Notes
    -----
    For now this explicitly flattens each of the columns, to avoid issues with shape
    """
    if columns:
        t1.keep_columns(columns)
        t2.keep_columns(columns)
    if sorted(t1.colnames) != sorted(t2.colnames):  # pragma: no cover
        return False
    for cname in t1.colnames:
        c1 = t1[cname]
        c2 = t2[cname]
        if c1.dtype.name == 'object':  # pragma: no cover
            c1_data = np.vstack(c1.value)
        else:
            c1_data = np.array(c1)
        if c2.dtype.name == 'object':
            c2_data = np.vstack(c2.value)
        else:
            c2_data = np.array(c2)     
        if not np.allclose(c1_data.flat, c2_data.flat):  # pragma: no cover
            return False
    return True


def compare_table_dicts(d1, d2, strict=False, columns=None):
    """Compare all the tables in two `OrderedDict`, (`str`, `astropy.table.Table`)

    Parameters
    ----------
    d1 : `OrderedDict`, (`str`, `astropy.table.Table`)
       One dictionary of tables
    d2 : `OrderedDict`, (`str`, `astropy.table.Table`)
       Another dictionary of tables

    Returns
    -------
    identical : `bool`
        True if all the tables are identical, False otherwise
    """
    identical = True
    for k, v in d1.items():
        try:
            vv = d2[k]
        except KeyError:  # pragma: no cover
            vv = d2[k.upper()]
        if strict:  # pragma: no cover
            identical &= apDiffUtils.report_diff_values(v, vv)
        else:  # pragma: no cover
            identical &= compare_tables(v, vv, columns=columns)
    return identical


def make_test_data():
    """Make and return some test data"""
    nrow = 1000
    vect_size = 20
    mat_size = 5
    scalar = np.random.uniform(size=nrow)
    vect = np.random.uniform(size=nrow * vect_size).reshape(nrow, vect_size)
    matrix = np.random.uniform(size=nrow * mat_size * mat_size).reshape(nrow, mat_size, mat_size)
    data = dict(scalar=scalar, vect=vect, matrix=matrix)
    table = apTable.Table(data)
    table.meta["a"] = 1
    table.meta["b"] = None
    table.meta["c"] = [3, 4, 5]
    small_table = apTable.Table(dict(a=np.ones(21), b=np.zeros(21)))
    small_table.meta["small"] = True
    out_tables = dict(data=table, md=small_table)
    return out_tables
