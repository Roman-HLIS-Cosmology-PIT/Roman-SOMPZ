"""Functions to store analysis results as astropy data tables  """

from collections import OrderedDict
from deprecated.sphinx import deprecated

from .io_utils import write, read
from .conv.conv_tabledict import convert
from .types import table_type


@deprecated(
    reason="This class is deprecated as it is not currently being used by tables_io.",
    version="1.0.0",
)
class TableDict(OrderedDict):
    """Warning: This class is being deprecated as of version 1.0.0.

    Object to collect various types of table-like objects

    This class is a dictionary mapping name to table-like
    and a few helper functions, e.g., to add new tables to the dictionary
    and to read and write files, either as FITS or HDF5 files.
    """

    def __setitem__(self, key, value):
        try:
            _ = table_type(value)
        except TypeError as msg:
            raise TypeError(f"item {value} was not recognized as a table.") from msg

        return OrderedDict.__setitem__(self, key, value)

    def write(self, basepath, fmt=None):
        """Write tables to the corresponding file type

        Parameters
        ----------
        basepath : `str`
            base path for output files.  Suffix will be added based on type
        fmt : `str` or `None`
            The output file format, If `None` this will use `write_native`
        """
        return write.write(self, basepath, fmt)

    def convert(self, tType):
        """Build a new TableDict by converting all the table in the object to a different type

        Parameters
        ----------
        tType : `int`
            The type to convert to

        Returns
        -------
        td : `TableDict`
            The new TableDict
        """
        return TableDict(convert(self, tType))

    @classmethod
    def read(cls, filepath, tType=None, fmt=None, keys=None):
        """Read a file to the corresponding table type

        Parameters
        ----------
        filepath : `str`
            File to load
        tType : `int` or `None`
            Table type, if `None` this will use `readNative`
        fmt : `str` or `None`
            File format, if `None` it will be taken from the file extension
        keys : `list` or `None`
            Keys to read for parquet files

        Returns
        -------
        tableDict : `TableDict`
            The data
        """
        return cls(read.read(filepath, tType, fmt, keys))
