"""Rail-specific data management"""

from __future__ import annotations

import enum
import os
import pickle
from typing import Any, Callable, Iterable, TypeAlias, TypeVar

import qp
import tables_io
from tables_io import hdf5 as tab_hdf5

from .model import Model

T = TypeVar("T", bound="DataHandle")

# These are place-holders for if and when we enforce typing
DataLike: TypeAlias = Any
GroupLike: TypeAlias = Any
ModelLike: TypeAlias = Any
TableLike: TypeAlias = Any
FileLike: TypeAlias = Any


class DataHandle:  # pylint: disable=too-many-instance-attributes
    """Class to act as a handle for a bit of data.  Associating it with a file and
    providing tools to read & write it to that file

    """

    suffix: str | None = ""

    # This is to keep track of all the sub-types
    _data_handle_type_dict: dict[str, type[DataHandle]] = {}

    def __init__(
        self,
        tag: str,
        data: DataLike | None = None,
        path: str | None = None,
        creator: str | None = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        tag
            The tag under which this data handle can be found in the store

        data
            The associated data

        path
            The path to the associated file

        creator
            The name of the stage that created this data handle
        """
        self.tag = tag
        if data is not None:
            self._validate_data(data)
        self.data = data
        self.path = path
        self.creator = creator
        self.fileObj: FileLike = None
        self.groups: GroupLike = None
        self.partial: bool | None = False
        self.length: int | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register the subclass with the dict"""
        cls._data_handle_type_dict[cls.__name__] = cls

    @classmethod
    def get_sub_classes(cls) -> dict[str, type[DataHandle]]:
        """Get all the subclasses"""
        return cls._data_handle_type_dict

    @classmethod
    def get_sub_class(cls, class_name: str) -> type[DataHandle]:
        """Get a particular subclass by name"""
        return cls._data_handle_type_dict[class_name]

    @classmethod
    def print_sub_classes(cls) -> None:
        """Print the list of all the subclasses"""
        for key, val in cls._data_handle_type_dict.items():
            print(f"{key}: {val}")

    def open(self, **kwargs: Any) -> FileLike:
        """Open and return the associated file

        Parameters
        ----------
        **kwargs
            Passed to the call to open the file in question

        Returns
        -------
        FileLike
            Newly opened file

        Notes
        -----
        This will simply open the file and return a FileLike object to the caller.
        It will not read or cache the data
        """
        if self.path is None:
            raise ValueError("DataHandle.open() called but path has not been specified")
        self.fileObj = self._open(os.path.expandvars(self.path), **kwargs)
        return self.fileObj

    @classmethod
    def _open(cls, path: str, **kwargs: Any) -> FileLike:
        raise NotImplementedError("DataHandle._open")  # pragma: no cover

    def close(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Close the associated file"""
        self.fileObj = None

    def read(self, force: bool = False, **kwargs: Any) -> DataLike:
        """Read and return the data from the associated file

        Parameters
        ----------
        force
            If true, force re-reading the data

        **kwargs
            Passed to the call to read the data

        Returns
        -------
        DataLike
            Data that were read

        Notes
        -----
        This will read the entire file, and while useful for testing on small files,
        will not work on very large files.
        """
        if self.data is not None and not force:
            return self.data
        assert self.path is not None
        self.set_data(self._read(os.path.expandvars(self.path), **kwargs))
        return self.data

    def __call__(self, **kwargs: Any) -> DataLike:
        """Return the data, re-reading the fill if needed"""
        if self.has_data and not self.partial:
            return self.data
        return self.read(force=True, **kwargs)

    @classmethod
    def _read(cls, path: str, **kwargs: Any) -> DataLike:
        raise NotImplementedError("DataHandle._read")  # pragma: no cover

    def write(self, **kwargs: Any) -> None:
        """Write the data to the associated file"""
        if self.path is None:
            raise ValueError(
                "TableHandle.write() called but path has not been specified"
            )
        if self.data is None:
            raise ValueError(
                f"TableHandle.write() called for path {self.path} with no data"
            )
        outdir = os.path.dirname(os.path.abspath(os.path.expandvars(self.path)))
        if not os.path.exists(outdir):  # pragma: no cover
            os.makedirs(outdir, exist_ok=True)
        return self._write(self.data, os.path.expandvars(self.path), **kwargs)

    @classmethod
    def _write(cls, data: DataLike, path: str, **kwargs: Any) -> None:
        raise NotImplementedError("DataHandle._write")  # pragma: no cover

    def initialize_write(self, data_length: int, **kwargs: Any) -> None:
        """Initialize file to be written by chunks

        Parameters
        ----------
        data_length
            Number of rows of data that we will write, used to reserve space

        **kwargs
            Information about the columns we will write
        """
        if self.path is None:  # pragma: no cover
            raise ValueError(
                "TableHandle.write() called but path has not been specified"
            )
        self.groups, self.fileObj = self._initialize_write(
            self.data, os.path.expandvars(self.path), data_length, **kwargs
        )

    @classmethod
    def _initialize_write(
        cls,
        data: DataLike,
        path: str,
        data_length: int,
        **kwargs: Any,
    ) -> tuple[GroupLike, FileLike]:
        raise NotImplementedError("DataHandle._initialize_write")  # pragma: no cover

    def write_chunk(self, start: int, end: int, **kwargs: Any) -> None:
        """Write the data to the associated file

        Parameters
        ----------
        start
            Index of starting row for this chunk of data

        end
            Index of ending row for this chunk of data

        **kwargs
            Passed to call to write this chunk of data
        """
        if self.data is None:
            raise ValueError(
                f"TableHandle.write_chunk() called for path {self.path} with no data"
            )
        if self.fileObj is None:
            raise ValueError(
                f"TableHandle.write_chunk() called before open for {self.tag} : {self.path}"
            )
        return self._write_chunk(
            self.data, self.fileObj, self.groups, start, end, **kwargs
        )

    @classmethod
    def _write_chunk(
        cls,
        data: DataLike,
        fileObj: FileLike,
        groups: GroupLike,
        start: int,
        end: int,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("DataHandle._write_chunk")  # pragma: no cover

    def finalize_write(self, **kwargs: Any) -> None:
        """Finalize and close file written by chunks

        Parameters
        ----------
        **kwargs
            Passed to call to write this chunk of data
        """
        if self.fileObj is None:  # pragma: no cover
            raise ValueError(
                f"TableHandle.finalize_wite() called before open for {self.tag} : {self.path}"
            )
        self._finalize_write(self.data, self.fileObj, **kwargs)

    @classmethod
    def _finalize_write(cls, data: DataLike, fileObj: FileLike, **kwargs: Any) -> None:
        raise NotImplementedError("DataHandle._finalize_write")  # pragma: no cover

    def iterator(self, **kwargs: Any) -> Iterable:
        """Iterator over the data"""
        if self.data is not None and self.partial is False:
            return self._in_memory_iterator(**kwargs)
        assert self.path is not None
        return self._iterator(self.path, **kwargs)

    def set_data(self, data: DataLike, partial: bool = False) -> None:
        """Set the data for a chunk, and set the partial flag to true"""
        self._validate_data(data)
        self.data = data
        self.partial = partial

    @classmethod
    def _validate_data(cls, data: DataLike) -> None:  # pylint: disable=unused-argument
        """Make sure that the right type of data is being passed in"""
        return

    def size(self, **kwargs: Any) -> int:
        """Return the size of the data associated to this handle"""
        if self.partial:
            if self.length is not None:  # pragma: no cover
                return self.length
            assert self.path is not None
            return self._size(self.path, **kwargs)
        if self.data is not None:
            return self.data_size(**kwargs)
        assert self.path is not None
        return self._size(self.path, **kwargs)

    def _size(self, path: str, **kwargs: Any) -> int:
        raise NotImplementedError("DataHandle._size")  # pragma: no cover

    def data_size(self, **kwargs: Any) -> int:
        """Return the size of the in memorry data"""
        if self.data is None:  # pragma: no cover
            return 0
        return self._data_size(self.data, **kwargs)

    def _data_size(self, data: DataLike, **kwargs: Any) -> int:
        raise NotImplementedError("DataHandle._data_size")  # pragma: no cover

    def _in_memory_iterator(self, **kwargs: Any) -> Iterable:
        raise NotImplementedError("DataHandle._in_memory_iterator")  # pragma: no cover

    @classmethod
    def _iterator(cls, path: str, **kwargs: Any) -> Iterable:
        raise NotImplementedError("DataHandle._iterator")  # pragma: no cover

    @property
    def has_data(self) -> bool:
        """Return true if the data for this handle are loaded"""
        return self.data is not None

    @property
    def has_path(self) -> bool:
        """Return true if the path for the associated file is defined"""
        return self.path is not None

    @property
    def is_written(self) -> bool:
        """Return true if the associated file has been written"""
        if self.path is None:
            return False
        return os.path.exists(os.path.expandvars(self.path))

    def __str__(self) -> str:
        s = f"{type(self)} "
        if self.has_path:
            s += f"{self.path}, ("
        else:
            s += "None, ("
        if self.is_written:
            s += "w"
        if self.has_data:
            s += "d"
        s += ")"
        return s

    @classmethod
    def make_name(cls, tag: str) -> str:
        """Construct and return file name for a particular data tag"""
        if cls.suffix:
            return f"{tag}.{cls.suffix}"
        return tag  # pragma: no cover

    @classmethod
    def _check_data_columns(
        cls,
        path: str,
        columns_to_check: list[str],
        parent_groupname: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Checking if certain columns required by the stage is present in the data"""
        # path: the path to data file
        # columns_to_check: list of columns required by the specific stage
        # kwargs: other key arguments required by specific data type, see below
        raise NotImplementedError  # pragma: no cover


class TableHandle(DataHandle):
    """DataHandle for single tables of data"""

    suffix: str | None = None

    def set_data(self, data: TableLike, partial: bool = False) -> None:
        """Set the data for a chunk, and set the partial flag if this is not all the data"""
        self._validate_data(data)
        self.data = data
        self.partial = partial

    @classmethod
    def _validate_data(cls, data: TableLike) -> None:  # pylint: disable=unused-argument
        """Make sure that the right type of data is being passed in"""
        return

    @classmethod
    def _open(cls, path: str, **kwargs: Any) -> FileLike:
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io_open(path, **kwargs)  # pylint: disable=no-member

    @classmethod
    def _read(cls, path: str, **kwargs: Any) -> TableLike:
        """Read and return the data from the associated file"""
        return tables_io.read(path, **kwargs)

    @classmethod
    def _write(cls, data: TableLike, path: str, **kwargs: Any) -> None:
        """Write the data to the associated file"""
        return tables_io.write(data, path, **kwargs)

    def _size(self, path: str, **kwargs: Any) -> int:
        if path in [None, "none", "None"]:  # pragma: no cover
            return 0
        return tab_hdf5.get_input_data_length(path, **kwargs)

    def _data_size(self, data: TableLike, **kwargs: Any) -> int:
        group_name = kwargs.get("groupname", None)
        if group_name:
            try:
                data = data[group_name]
            except KeyError:  # pragma: no cover
                pass
        max_l = 0
        for _k, v in data.items():
            max_l = max(max_l, len(v))
        return max_l

    def _in_memory_iterator(self, **kwargs: Any) -> Iterable:
        nrows = self.data_size(**kwargs)
        groupname = kwargs.get("groupname", None)
        if isinstance(self.data, dict) and groupname:
            try:
                table = self.data[groupname]
            except KeyError:  # pragma: no cover
                table = self.data
        else:
            table = self.data
        for start, end in tab_hdf5.data_ranges_by_rank(
            nrows, kwargs.get("chunk_size", 100000), 1, 0
        ):
            if isinstance(self.data, dict):
                yield start, end, tables_io.array_utils.slice_dict(
                    table, slice(start, end)
                )
            else:  # pragma: no cover
                yield start, end, table[start:end]

    @classmethod
    def _iterator(cls, path: str, **kwargs: Any) -> Iterable:
        """Iterate over the data"""
        return tables_io.iteratorNative(path, **kwargs)

    @classmethod
    def _check_data_columns(
        cls,
        path: str,
        columns_to_check: list[str],
        parent_groupname: str | None = None,
        **kwargs: Any,
    ) -> None:
        tables_io.check_columns(
            path, columns_to_check, parent_groupname=parent_groupname, **kwargs
        )


class Hdf5Handle(TableHandle):  # pragma: no cover
    """DataHandle for a table written to HDF5"""

    suffix = "hdf5"

    @classmethod
    def _initialize_write(
        cls, data: TableLike, path: str, data_length: int, **kwargs: Any
    ) -> tuple[GroupLike, FileLike]:
        initial_dict = cls._get_allocation_kwds(data, data_length)
        comm = kwargs.get("communicator", None)
        group, fout = tab_hdf5.initialize_HDF5_write_single(
            path, groupname=None, comm=comm, **initial_dict
        )
        return group, fout

    @classmethod
    def _get_allocation_kwds(cls, data: TableLike, data_length: int) -> dict:
        keywords = {}
        for key, array in data.items():
            shape = list(array.shape)
            if not shape:
                continue
            shape[0] = data_length
            keywords[key] = (shape, array.dtype)
        return keywords

    @classmethod
    def _write_chunk(
        cls,
        data: TableLike,
        fileObj: FileLike,
        groups: GroupLike,
        start: int,
        end: int,
        **kwargs: Any,
    ) -> None:
        tab_hdf5.write_dict_to_HDF5_chunk_single(fileObj, data, start, end, **kwargs)

    @classmethod
    def _finalize_write(cls, data: TableLike, fileObj: FileLike, **kwargs: Any) -> None:
        return tab_hdf5.finalize_HDF5_write(fileObj, **kwargs)


class FitsHandle(TableHandle):
    """DataHandle for a table written to fits"""

    suffix = "fits"


class PqHandle(TableHandle):
    """DataHandle for a parquet table"""

    suffix = "pq"

    def _size(self, path: str, **kwargs: Any) -> int:
        return tab_hdf5.get_input_data_length(path, **kwargs)


class QPHandle(DataHandle):
    """DataHandle for qp ensembles"""

    suffix = "hdf5"

    @classmethod
    def _open(cls, path: str, **kwargs: Any) -> FileLike:
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a FileLike object to the caller.
        It will not read or cache the data
        """
        return tables_io.io_open(path, **kwargs)  # pylint: disable=no-member

    @classmethod
    def _read(cls, path: str, **kwargs: Any) -> qp.Ensemble:
        """Read and return the data from the associated file"""
        return qp.read(path)

    @classmethod
    def _write(cls, data: qp.Ensemble, path: str, **kwargs: Any) -> None:
        """Write the data to the associated file"""
        return data.write_to(path)

    @classmethod
    def _initialize_write(
        cls, data: qp.Ensemble, path: str, data_length: int, **kwargs: Any
    ) -> tuple[GroupLike, FileLike]:
        comm = kwargs.get("communicator", None)
        return data.initializeHdf5Write(path, data_length, comm)

    @classmethod
    def _write_chunk(
        cls,
        data: qp.Ensemble,
        fileObj: FileLike,
        groups: GroupLike,
        start: int,
        end: int,
        **kwargs: Any,
    ) -> None:
        return data.writeHdf5Chunk(fileObj, start, end)

    @classmethod
    def _finalize_write(
        cls, data: qp.Ensemble, fileObj: FileLike, **kwargs: Any
    ) -> None:
        return data.finalizeHdf5Write(fileObj)

    @classmethod
    def _validate_data(cls, data: qp.Ensemble) -> None:
        if not isinstance(data, qp.Ensemble):
            raise TypeError(
                f"Expected `data` to be a `qp.Ensemble`, but {type(data)} was provided."
                "Perhaps you meant to use `TableHandle`?"
            )

    def _size(self, path: str, **kwargs: Any) -> int:
        if self.data is not None and not self.partial:  # pragma: no cover
            return self._data_size(self.data)
        if path in [None, "none", "None"]:  # pragma: no cover
            return 0
        return tab_hdf5.get_input_data_length(path, groupname="data")

    def _data_size(self, data: qp.Ensemble, **kwargs: Any) -> int:
        return self.data.npdf

    def _in_memory_iterator(self, **kwargs: Any) -> Iterable:
        nrows = self.data.npdf
        for start, end in tab_hdf5.data_ranges_by_rank(
            nrows, kwargs.get("chunk_size", 100000), 1, 0
        ):
            yield start, end, self.data[start:end]

    @classmethod
    def _iterator(cls, path: str, **kwargs: Any) -> Iterable:
        """Iterate over the data"""
        kwargs.pop("groupname", "None")
        return qp.iterator(path, **kwargs)


class QPDictHandle(DataHandle):
    """DataHandle for dictionaries of qp ensembles"""

    suffix = "hdf5"

    @classmethod
    def _open(cls, path: str, **kwargs: Any) -> FileLike:
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a FileLike object to the caller.
        It will not read or cache the data
        """
        return tables_io.io_open(path, **kwargs)  # pylint: disable=no-member

    @classmethod
    def _read(cls, path: str, **kwargs: Any) -> dict[str, qp.Ensemble]:
        """Read and return the dictionary of qp.Ensembles from the associated file"""
        return qp.read_dict(path)

    @classmethod
    def _write(cls, data: dict[str, qp.Ensemble], path: str, **kwargs: Any) -> None:
        """Write the data (a dictionary of qp.Ensembles) to the associated file"""
        return qp.write_dict(path, data)


class QPOrTableHandle(QPHandle, Hdf5Handle):
    """DataHandle that will work with either qp.Ensembles or TableLike data"""

    suffix = "hdf5"

    class PdfOrValue(enum.Enum):
        unknown = -1
        distribution = 0
        point_estimate = 1
        both = 2

        def has_dist(self) -> bool:
            return self.value in [0, 2]

        def has_point(self) -> bool:
            return self.value in [1, 2]

    def is_qp(self) -> bool:
        """Check if the associated data or file is a QP ensemble"""
        if self.path in [None, "None", "none"]:
            return isinstance(self.data, qp.Ensemble)
        return qp.is_qp_file(self.path)

    def check_pdf_or_point(self) -> PdfOrValue:
        """Check the associated file to see if it is a QP pdf, point estimate or both"""
        if self.is_qp():
            return self.PdfOrValue.both
        return self.PdfOrValue.point_estimate

    @classmethod
    def _open(cls, path: str, **kwargs: Any) -> FileLike:
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io_open(path, **kwargs)  # pylint: disable=no-member

    @classmethod
    def _read(cls, path: str, **kwargs: Any) -> qp.Ensemble | TableLike:
        """Read and return the data from the associated file"""
        if qp.is_qp_file(path):
            return qp.read(path, **kwargs)
        return tables_io.read(path, **kwargs)

    @classmethod
    def _write(cls, data: qp.Ensemble | TableLike, path: str, **kwargs: Any) -> None:
        """Write the data to the associated file"""
        raise RuntimeError(
            "QPOrTableHandle should be used for input, not output"
        )  # pragma: no cover

    @classmethod
    def _initialize_write(
        cls, data: qp.Ensemble | TableLike, path: str, data_length: int, **kwargs: Any
    ) -> tuple[GroupLike, FileLike]:
        raise RuntimeError(
            "QPOrTableHandle should be used for input, not output"
        )  # pragma: no cover

    @classmethod
    def _write_chunk(
        cls,
        data: qp.Ensemble | TableLike,
        fileObj: FileLike,
        groups: GroupLike,
        start: int,
        end: int,
        **kwargs: Any,
    ) -> None:
        raise RuntimeError(
            "QPOrTableHandle should be used for input, not output"
        )  # pragma: no cover

    @classmethod
    def _finalize_write(
        cls, data: qp.Ensemble | TableLike, fileObj: FileLike, **kwargs: Any
    ) -> None:
        raise RuntimeError(
            "QPOrTableHandle should be used for input, not output"
        )  # pragma: no cover

    @classmethod
    def _validate_data(cls, data: qp.Ensemble | TableLike) -> None:
        pass

    def _size(self, path: str, **kwargs: Any) -> int:
        if self.is_qp():
            return QPHandle._size(self, path, **kwargs)
        return Hdf5Handle._size(self, path, **kwargs)

    def _data_size(self, data: qp.Ensemble | TableLike, **kwargs: Any) -> int:
        if self.is_qp():
            return self.data.npdf
        return Hdf5Handle._data_size(self, data, **kwargs)

    def _in_memory_iterator(self, **kwargs: Any) -> Iterable:
        if self.is_qp():
            return QPHandle._in_memory_iterator(self, **kwargs)
        return Hdf5Handle._in_memory_iterator(self, **kwargs)

    @classmethod
    def _iterator(cls, path: str, **kwargs: Any) -> Iterable:
        """Iterate over the data"""
        if qp.is_qp_file(path):
            return QPHandle._iterator(path, **kwargs)
        return Hdf5Handle._iterator(path, **kwargs)


def default_model_read(modelfile: str) -> ModelLike:
    """Default function to read model files, simply used pickle.load"""
    with open(modelfile, "rb") as fin:
        read_data = pickle.load(fin)
    if isinstance(read_data, Model):
        ret_data = read_data.data
    else:
        ret_data = read_data
    return ret_data


def default_model_write(model: ModelLike, path: str) -> None:
    """Write the model, this default implementation uses pickle"""
    with open(path, "wb") as fout:
        pickle.dump(obj=model, file=fout, protocol=pickle.HIGHEST_PROTOCOL)


class ModelDict(dict):
    """
    A specialized dict to keep track of individual estimation models objects:
    this is just a dict these additional features

    1. Keys are paths
    2. There is a read(path, force=False) method that reads a model object and
        inserts it into the dictionary
    3. There is a single static instance of this class

    """

    def open(self, path: str, mode: str, **kwargs: Any) -> FileLike:
        """Open the file and return the file handle"""
        return open(path, mode, **kwargs)  # pylint: disable=unspecified-encoding

    def read(
        self,
        path: str,
        force: bool = False,
        reader: Callable | None = None,
        **_kwargs: Any,
    ) -> ModelLike:  # pylint: disable=unused-argument
        """Read a model into this dict"""
        if reader is None:
            reader = default_model_read
        if force or path not in self:
            model = reader(path)
            self[path] = model
            return model
        return self[path]

    def write(
        self,
        model: ModelLike,
        path: str,
        force: bool = False,
        writer: Callable | None = None,
        **_kwargs: Any,
    ) -> None:  # pylint: disable=unused-argument
        """Write the model, this default implementation uses pickle"""
        if writer is None:
            writer = default_model_write
        if force or path not in self or not os.path.exists(path):
            self[path] = model
            writer(model, path)


class ModelHandle(DataHandle):
    """DataHandle for machine learning models"""

    suffix = "pkl"

    model_factory = ModelDict()

    @classmethod
    def _open(cls, path: str, **kwargs: Any) -> FileLike:
        """Open and return the associated file"""
        kwcopy = kwargs.copy()
        if kwcopy.pop("mode", "r") == "w":
            return cls.model_factory.open(path, mode="wb", **kwcopy)
        force = kwargs.pop("force", False)
        reader = kwargs.pop("reader", None)
        assert isinstance(force, bool)
        return cls.model_factory.read(path, force=force, reader=reader, **kwargs)

    @classmethod
    def _read(cls, path: str, **kwargs: Any) -> ModelLike:
        """Read and return the data from the associated file"""
        force = kwargs.pop("force", False)
        reader = kwargs.pop("reader", None)
        assert isinstance(force, bool)
        return cls.model_factory.read(path, force=force, reader=reader, **kwargs)

    @classmethod
    def _write(cls, data: ModelLike, path: str, **kwargs: Any) -> None:
        """Write the data to the associated file"""
        force = kwargs.pop("force", False)
        writer = kwargs.pop("writer", None)
        assert isinstance(force, bool)
        return cls.model_factory.write(data, path, force=force, writer=writer, **kwargs)


class DataStore(dict):
    """Class to provide a transient data store

    This class:

    1. associates data products with keys
    2. provides functions to read and write the various data produces to associated files

    """

    allow_overwrite = False

    def __init__(self, **kwargs: Any) -> None:
        """Build from keywords

        Note
        ----
        All of the values must be data handles of this will raise a TypeError
        """
        super(dict, self).__init__()
        for key, val in kwargs.items():
            self[key] = val

    def __str__(self) -> str:
        """Override __str__ casting to deal with `TableHandle` objects in the map"""
        s = "{"
        for key, val in self.items():
            s += f"  {key}:{val}\n"
        s += "}"
        return s

    def __repr__(self) -> str:
        """A custom representation"""
        s = "DataStore\n"
        s += self.__str__()
        return s

    def __setitem__(self, key: str, value: DataHandle) -> None:
        """Override the __setitem__ to work with ``TableHandle``"""
        if not isinstance(value, DataHandle):
            raise TypeError(
                f"Can only add objects of type DataHandle to DataStore, not {type(value)}"
            )
        check = self.get(key)
        if check is not None and not self.allow_overwrite:
            raise ValueError(
                f"DataStore already has an item with key {key},"
                f"of type {type(check)}, created by {check.creator}"
            )
        dict.__setitem__(self, key, value)

    def __getattr__(self, key: str) -> DataHandle | None:
        """Allow attribute-like parameter access"""
        try:
            return self.__getitem__(key)
        except KeyError as msg:
            # Kludge to get docstrings to work
            if key in ["__objclass__"]:  # pragma: no cover
                return None
            raise KeyError from msg

    def __setattr__(self, key: str, value: DataHandle) -> None:
        """Allow attribute-like parameter setting"""
        self.__setitem__(key, value)

    def add_data(
        self,
        key: str,
        data: DataLike,
        handle_class: type[DataHandle],
        path: str | None = None,
        creator: str = "DataStore",
    ) -> DataHandle:
        """Create a handle for some data, and insert it into the DataStore"""
        handle = handle_class(key, path=path, data=data, creator=creator)
        self[key] = handle
        return handle

    def add_handle(
        self,
        key: str,
        handle_class: type[DataHandle],
        path: str,
        creator: str = "DataStore",
    ) -> DataHandle:
        """Create a handle for some data, and insert it into the DataStore"""
        handle = handle_class(key, path=path, data=None, creator=creator)
        self[key] = handle
        return handle

    def read_file(
        self,
        key: str,
        handle_class: type[DataHandle],
        path: str,
        creator: str = "DataStore",
        **kwargs: Any,
    ) -> DataHandle:
        """Create a handle, use it to read a file, and insert it into the DataStore"""
        handle = handle_class(key, path=path, data=None, creator=creator)
        handle.read(**kwargs)
        self[key] = handle
        return handle

    def read(self, key: str, force: bool = False, **kwargs: Any) -> DataLike:
        """Read the data associated to a particular key"""
        try:
            handle = self[key]
        except KeyError as msg:
            raise KeyError(f"Failed to read data {key} because {msg}") from msg
        return handle.read(force, **kwargs)

    def open(self, key: str, mode: str = "r", **kwargs: Any) -> FileLike:
        """Open and return the file associated to a particular key"""
        try:
            handle = self[key]
        except KeyError as msg:
            raise KeyError(f"Failed to open data {key} because {msg}") from msg
        return handle.open(mode=mode, **kwargs)

    def write(self, key: str, **kwargs: Any) -> None:
        """Write the data associated to a particular key"""
        try:
            handle = self[key]
        except KeyError as msg:
            raise KeyError(f"Failed to write data {key} because {msg}") from msg
        return handle.write(**kwargs)

    def write_all(self, force: bool = False, **kwargs: Any) -> None:
        """Write all the data in this DataStore"""
        for key, handle in self.items():
            local_kwargs = kwargs.get(key, {})
            if handle.is_written and not force:
                continue
            handle.write(**local_kwargs)


_DATA_STORE = DataStore()


def DATA_STORE() -> DataStore:
    """Return the factory instance"""
    return _DATA_STORE
