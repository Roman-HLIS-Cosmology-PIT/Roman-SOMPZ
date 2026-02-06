import os
from types import GeneratorType
import numpy as np

import pytest

from rail.core.data import Hdf5Handle, ModelHandle, TableHandle
from rail.core.stage import RailStage
from rail.core.common_params import SHARED_PARAMS
from rail.tools.table_tools import ColumnMapper, RowSelector, TableConverter
from rail.utils.path_utils import RAILDIR, find_rail_file
from rail.utils.catalog_utils_old import CatalogConfigBase
from rail.utils import catalog_utils


def test_find_rail_file() -> None:
    afile = find_rail_file(
        os.path.join("examples_data", "testdata", "test_dc2_training_9816.pq")
    )
    assert afile
    with pytest.raises(ValueError):
        _not_a_file = find_rail_file("not_a_file")


def test_util_stages() -> None:
    DS = RailStage.data_store
    DS.clear()
    datapath = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.pq"
    )

    data = DS.read_file("data", TableHandle, datapath)

    table_conv = TableConverter.make_stage(name="conv", output_format="numpyDict")
    col_map = ColumnMapper.make_stage(name="col_map", columns={})
    row_sel = RowSelector.make_stage(name="row_sel", start=1, stop=15)

    with pytest.raises(KeyError) as _errinfo:
        table_conv.get_handle("nope", allow_missing=False)

    _conv_data = table_conv(data)
    mapped_data = col_map(data)
    _sel_data = row_sel(mapped_data)

    row_sel_2 = RowSelector.make_stage(name="row_sel_2", start=1, stop=15)
    row_sel_2.set_data("input", mapped_data.data)
    handle = row_sel_2.get_handle("input")

    row_sel_3 = RowSelector.make_stage(
        name="row_sel_3", input=handle.path, start=1, stop=15
    )
    row_sel_3.set_data("input", None, do_read=True)

    for stage in [table_conv, col_map, row_sel]:
        os.remove(stage.get_output(stage.get_aliased_tag("output"), final_name=True))


def test_set_data_nonexistent_file() -> None:
    """Create an instance of a child class of RailStage. Exercise the `set_data`
    method and pass in a path to a nonexistent file. A `FileNotFound` exception
    should be raised.
    """

    col_map = ColumnMapper.make_stage(name="col_map", columns={})
    with pytest.raises(FileNotFoundError) as err:
        col_map.set_data("input", None, path="./bad_directory/no_file.py")
        assert "Unable to find file" in err.context


def test_set_data_real_file() -> None:
    """Create an instance of a child class of RailStage. Exercise the `set_data`
    method and pass in a path to model. The output of set_data should be `None`.
    """
    DS = RailStage.data_store
    DS.clear()
    model_path = os.path.join(
        RAILDIR,
        "rail",
        "examples_data",
        "estimation_data",
        "data",
        "CWW_HDFN_prior.pkl",
    )
    DS.add_data("model", None, ModelHandle, path=model_path)

    col_map = ColumnMapper.make_stage(name="col_map", columns={})

    ret_val = col_map.set_data("model", None, path=model_path, do_read=False)

    assert ret_val is None


def test_data_hdf5_iter() -> None:
    DS = RailStage.data_store
    DS.clear()

    datapath = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.hdf5"
    )

    # data = DS.read_file('data', TableHandle, datapath)
    th = Hdf5Handle("data", path=datapath)
    x = th.iterator(groupname="photometry", chunk_size=1000)

    assert isinstance(x, GeneratorType)
    for i, xx in enumerate(x):
        assert xx[0] == i * 1000
        assert xx[1] - xx[0] <= 1000

    _data = DS.read_file("input", TableHandle, datapath)
    cm = ColumnMapper.make_stage(
        input=datapath,
        chunk_size=1000,
        hdf5_groupname="photometry",
        columns=dict(id="bob"),
    )
    x = cm.input_iterator("input")

    assert isinstance(x, GeneratorType)

    for i, xx in enumerate(x):
        assert xx[0] == i * 1000
        assert xx[1] - xx[0] <= 1000


def test_catalog_tags():

    catalog_utils.clear()

    for key in CatalogConfigBase.subclasses().keys():
        if key in ["dp1_all"]:
            continue

        catalog_utils.apply_defaults_old(key)
        sp1 = SHARED_PARAMS.copy()

        catalog_utils.apply_defaults(key)
        sp2 = SHARED_PARAMS.copy()

        _band_name_dict = catalog_utils.get_active_tag().band_name_dict()

        for k2, v2 in sp1.items():
            if isinstance(v2, list):
                assert v2 == sp2[k2], f"For {key}:{k2}"
            elif isinstance(v2, dict):
                for k3, v3 in v2.items():
                    assert v3 == v2[k3], f"For {key}:{k2}"
            elif isinstance(v2, str):
                assert sp2[k2] == v2, f"For {key}:{k2}"
            elif np.isnan(v2):
                assert np.isnan(sp2[k2]), f"For {key}:{k2}"
            else:
                assert sp2[k2] == v2, f"For {key}:{k2}"
