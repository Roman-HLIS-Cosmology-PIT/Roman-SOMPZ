import os
import pickle
from types import GeneratorType

import numpy as np
import pytest

from rail.core.common_params import copy_param, set_param_default
from rail.core.data import (
    DataHandle,
    DataStore,
    FitsHandle,
    Hdf5Handle,
    ModelHandle,
    PqHandle,
    QPHandle,
    QPOrTableHandle,
)
from rail.core.model import Model
from rail.core.stage import RailStage
from rail.utils.catalog_utils import CatalogConfigBase
from rail.utils.catalog_utils_old import RomanPlusRubinCatalogConfig
from rail.utils.path_utils import RAILDIR

# def test_data_file():
#    with pytest.raises(ValueError) as errinfo:
#        df = DataFile('dummy', 'x')


def do_data_handle(datapath: str, handle_class: type[DataHandle]) -> DataHandle:
    _DS = RailStage.data_store

    th = handle_class("data", path=datapath)

    with pytest.raises(ValueError) as _errinfo:
        th.write()

    assert handle_class.__name__ in DataHandle.get_sub_classes()

    assert not th.has_data
    try:
        _check_size = th.size()
    except NotImplementedError as msg:
        if not isinstance(th, FitsHandle):
            raise NotImplementedError(msg) from msg

    with pytest.raises(ValueError) as _errinfo:
        th.write_chunk(0, 1)
    assert th.has_path
    assert th.is_written
    data = th.read()
    data2 = th.read()

    assert data is data2
    assert th.has_data
    assert th.make_name("data") == f"data.{handle_class.suffix}"

    th2 = handle_class("data2", data=data)
    assert th2.has_data
    assert th2.size() > 0

    assert not th2.has_path
    assert not th2.is_written
    with pytest.raises(ValueError) as _errinfo:
        th2.open()
    with pytest.raises(ValueError) as _errinfo:
        th2.write()
    with pytest.raises(ValueError) as _errinfo:
        th2.write_chunk(0, 1)

    assert th2.make_name("data2") == f"data2.{handle_class.suffix}"
    assert str(th)
    assert str(th2)
    return th


def test_pq_handle() -> None:
    datapath = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.pq"
    )
    handle = do_data_handle(datapath, PqHandle)
    pqfile = handle.open()
    assert pqfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None


def test_qp_handle() -> None:
    datapath = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "output_BPZ_lite.hdf5"
    )
    handle = do_data_handle(datapath, QPHandle)
    qpfile = handle.open()
    assert qpfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None

    with pytest.raises(TypeError) as _errInfo:
        _bad_dh = QPHandle(tag="bad_tag", data="this is not an Ensemble")


def test_qp_or_table_handle_qp() -> None:
    datapath = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "output_BPZ_lite.hdf5"
    )
    handle = do_data_handle(datapath, QPOrTableHandle)
    qpfile = handle.open()
    assert qpfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None

    x = handle.iterator(chunk_size=100)

    assert isinstance(x, GeneratorType)
    for i, xx in enumerate(x):
        assert xx[0] == i * 100
        assert xx[1] - xx[0] <= 100

    handle2 = QPOrTableHandle(tag="qp_or_table_qp_2", path=datapath)

    x2 = handle2.iterator(chunk_size=100)

    assert isinstance(x2, GeneratorType)
    for i2, xx2 in enumerate(x2):
        assert xx2[0] == i2 * 100
        assert xx2[1] - xx2[0] <= 100


def test_qp_or_table_handle_table() -> None:
    datapath = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.hdf5"
    )
    handle = do_data_handle(datapath, QPOrTableHandle)
    tablefile = handle.open()
    assert tablefile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None

    x = handle.iterator(chunk_size=100)

    assert isinstance(x, GeneratorType)
    for i, xx in enumerate(x):
        assert xx[0] == i * 100
        assert xx[1] - xx[0] <= 100

    handle2 = QPOrTableHandle(tag="qp_or_table_table_2", path=datapath)

    x2 = handle2.iterator(groupname="photometry", chunk_size=100)

    assert isinstance(x2, GeneratorType)
    for i2, xx2 in enumerate(x2):
        assert xx2[0] == i2 * 100
        assert xx2[1] - xx2[0] <= 100


def test_hdf5_handle() -> None:
    datapath = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.hdf5"
    )
    handle = do_data_handle(datapath, Hdf5Handle)
    with handle.open(mode="r") as f:
        assert f
        assert handle.fileObj is not None
    datapath_chunked = os.path.join(
        RAILDIR,
        "rail",
        "examples_data",
        "testdata",
        "test_dc2_training_9816_chunked.hdf5",
    )
    handle_chunked = Hdf5Handle("chunked", handle.data, path=datapath_chunked)
    from tables_io.utils.array_utils import (  # pylint: disable=import-outside-toplevel
        get_initialization_for_ODict,
        slice_dict,
    )

    num_rows = len(handle.data["photometry"]["id"])
    check_num_rows = len(handle()["photometry"]["id"])
    assert num_rows == check_num_rows
    chunk_size = 1000
    data = handle.data["photometry"]
    init_dict = get_initialization_for_ODict(data)
    with handle_chunked.open(mode="w") as fout:
        for k, v in init_dict.items():
            fout.create_dataset(k, v[0], v[1])
        for i in range(0, num_rows, chunk_size):
            start = i
            end = i + chunk_size
            end = min(end, num_rows)
            handle_chunked.set_data(
                slice_dict(handle.data["photometry"], slice(start, end)), partial=True
            )
            handle_chunked.write_chunk(start, end)
    write_size = handle_chunked.size()
    assert len(handle_chunked.data) <= 1000
    data_called = handle_chunked()
    assert len(data_called["id"]) == write_size
    read_chunked = Hdf5Handle("read_chunked", None, path=datapath_chunked)
    data_check = read_chunked.read()
    assert np.allclose(data["id"], data_check["id"])
    assert np.allclose(data_called["id"], data_check["id"])
    os.remove(datapath_chunked)


def test_fits_handle() -> None:
    datapath = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "output_BPZ_lite.fits"
    )
    handle = do_data_handle(datapath, FitsHandle)
    fitsfile = handle.open()
    assert fitsfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None


def test_model_handle() -> None:
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
    model_path_copy = os.path.join(
        RAILDIR,
        "rail",
        "examples_data",
        "estimation_data",
        "data",
        "CWW_HDFN_prior_copy.pkl",
    )
    model_path_wrap = os.path.join(
        RAILDIR,
        "rail",
        "examples_data",
        "estimation_data",
        "data",
        "CWW_HDFN_prior_wrap.pkl",
    )
    mh = ModelHandle("model", path=model_path)
    mh2 = ModelHandle("model2", path=model_path)
    Model.wrap(model_path, model_path_wrap)
    mh4 = ModelHandle("modelWrap", path=model_path_wrap)

    model1 = mh.read()
    model2 = mh2.read()

    model3 = mh.open()

    _model4 = mh4.read()

    assert model1 is model2
    assert model2 is model3

    mh3 = ModelHandle("model3", path=model_path_copy, data=model1)
    with mh3.open(mode="w") as fout:
        pickle.dump(obj=mh3.data, file=fout, protocol=pickle.HIGHEST_PROTOCOL)
    os.remove(model_path_copy)
    os.remove(model_path_wrap)


def test_data_store() -> None:
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    datapath_hdf5 = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.hdf5"
    )
    datapath_pq = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.pq"
    )
    datapath_hdf5_copy = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816_copy.hdf5"
    )
    datapath_pq_copy = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816_copy.pq"
    )

    DS.add_data("hdf5", None, Hdf5Handle, path=datapath_hdf5)
    DS.add_data("pq", None, PqHandle, path=datapath_pq)

    with DS.open("hdf5") as f:
        assert f

    data_pq = DS.read("pq")
    data_hdf5 = DS.read("hdf5")

    DS.add_data("pq_copy", data_pq, PqHandle, path=datapath_pq_copy)
    DS.add_data("hdf5_copy", data_hdf5, Hdf5Handle, path=datapath_hdf5_copy)
    DS.write("pq_copy")
    DS.write("hdf5_copy")

    with pytest.raises(KeyError) as _errinfo:
        DS.read("nope")
    with pytest.raises(KeyError) as _errinfo:
        DS.open("nope")
    with pytest.raises(KeyError) as _errinfo:
        DS.write("nope")

    with pytest.raises(TypeError) as _errinfo:
        DS["nope"] = None
    with pytest.raises(ValueError) as _errinfo:
        DS["pq"] = DS["pq"]
    with pytest.raises(ValueError) as _errinfo:
        DS.pq = DS["pq"]

    a_handle = DS.add_handle("pq_copy_2", PqHandle, path=datapath_pq_copy)
    assert a_handle

    assert repr(DS)

    DS2 = DataStore(pq=DS.pq)
    assert isinstance(DS2.pq, DataHandle)

    # pop the 'pq' data item to avoid overwriting file under git control
    DS.pop("pq")
    # pop the 'pq_copy_2' because it is empty
    DS.pop("pq_copy_2")

    DS.write_all()
    DS.write_all(force=True)

    os.remove(datapath_hdf5_copy)
    os.remove(datapath_pq_copy)


def test_common_params() -> None:
    par = copy_param("zmin")
    assert par.default == 0.0
    assert par.value == 0.0
    assert par.dtype == float

    set_param_default("zmin", 0.1)
    par = copy_param("zmin")
    assert par.default == 0.1
    assert par.value == 0.1
    assert par.dtype == float


def test_catalog_utils() -> None:
    CatalogConfigBase.apply("rubin")
    CatalogConfigBase.band_name_dict()["u"] = "LSST_obs_u"
    assert CatalogConfigBase.active_tag() == "rubin"
    assert CatalogConfigBase.active_class().tag == "rubin"

    CatalogConfigBase.apply("roman_plus_rubin")
    RomanPlusRubinCatalogConfig.band_name_dict()["u"] = "LSST_obs_u"
    assert CatalogConfigBase.active_tag() == "roman_plus_rubin"
    assert CatalogConfigBase.active_class().tag == "roman_plus_rubin"

    for tag in [
        "hsc",
        "roman_3band",
        "roman_7band",
        "roman_rubin",
        "com_cam",
        "com_cam_gaap",
        "com_cam_euclid",
        "roman_3band_rubin",
        "roman_7band_rubin",
        "com_cam_euclid_nir",
    ]:
        CatalogConfigBase.apply(tag)
        assert CatalogConfigBase.active_class().band_name_dict()

    CatalogConfigBase.apply("dc2")
    set_param_default("redshift_col", "redshift")

    a_class = CatalogConfigBase.get_class(
        "RomanPlusRubinCatalogConfig", "rail.utils.catalog_utils"
    )
    CatalogConfigBase.apply_class(a_class.__name__)

    assert "rubin" in CatalogConfigBase.subclasses()

    assert "RomanPlusRubinCatalogConfig" in CatalogConfigBase.subclasses_by_class()
