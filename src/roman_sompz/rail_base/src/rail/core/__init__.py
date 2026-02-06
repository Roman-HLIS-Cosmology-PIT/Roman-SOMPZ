"""Core code for RAIL"""

from .common_params import (SHARED_PARAMS, SharedParams, copy_param,
                            set_param_default, set_param_defaults)
from .data import (DATA_STORE, DataHandle, DataStore, FitsHandle, Hdf5Handle,
                   ModelDict, ModelHandle, PqHandle, QPDictHandle, QPHandle,
                   QPOrTableHandle, TableHandle)
from .introspection import RailEnv
from .model import Model
from .point_estimation import PointEstimationMixin
from .stage import RailPipeline, RailStage, StageIO

__all__ = [
    "SharedParams",
    "copy_param",
    "set_param_default",
    "set_param_defaults",
    "SHARED_PARAMS",
    "DataHandle",
    "TableHandle",
    "Hdf5Handle",
    "FitsHandle",
    "PqHandle",
    "QPHandle",
    "QPDictHandle",
    "QPOrTableHandle",
    "Model",
    "ModelDict",
    "ModelHandle",
    "DataStore",
    "DATA_STORE",
    "PointEstimationMixin",
    "StageIO",
    "RailPipeline",
    "RailStage",
    "RailEnv",
]


def find_version() -> str:
    """Find the version"""
    # setuptools_scm should install a
    # file _version alongside this one.
    from . import _version  # pylint: disable=import-outside-toplevel

    return _version.version


try:
    __version__ = find_version()
except ImportError:  # pragma: no cover
    __version__ = "unknown"
