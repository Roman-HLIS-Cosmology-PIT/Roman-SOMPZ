"""Wrapper class for ML Models"""

from __future__ import annotations

import pickle
from typing import Any


class Model:
    """Class to act as wrapper for ML models

    This will attach metadata to the ML model, and provide tools
    for validation, versioning and tracking provenance.

    """

    def __init__(
        self,
        data: Any,
        creation_class_name: str,
        version: int = 0,
        catalog_tag: str | None = None,
        provenance: dict | None = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        data
            Model data

        creation_class_name
            Name of class that created this model

        version
            Version of the model

        catalog_tag
            Associated CatalogConfigBase that defined training dataset

        provenance
            Provenance infomration
        """
        self.data = data
        self.creation_class_name = creation_class_name
        self.version = version
        self.catalog_tag = catalog_tag
        if provenance is not None:
            self.provenance = provenance.copy()
        else:
            self.provenance = {}

    def validate(self, creation_class_name: str | None, version: int | None) -> None:
        """
        Parameters
        ----------
        creation_class_name
            Name of class that created this model

        version
            Version of the model

        Raises
        ------
        TypeError : Either creation_class_name or version does not match
        """
        if (
            creation_class_name is not None
            and creation_class_name != self.creation_class_name
        ):
            raise TypeError(
                "Model.creation_class_name does not match.  "
                f"{creation_class_name} != {self.creation_class_name}"
            )
        if version is not None and version != self.version:
            raise TypeError(
                f"Model.version does not match.  {version} != {self.version}"
            )

    @classmethod
    def read(
        cls,
        path: str,
        creation_class_name: str = "dummy",
        version: int = 0,
        catalog_tag: str | None = None,
        provenance: dict | None = None,
    ) -> Model:
        """Read a model from a file.

        Note that this will promote the data to a Model if it is not already

        Parameters
        ----------
        path
            File to read

        creation_class_name:
            Name of class that created this model

        version:
            Version of the model

        catalog_tag
            Associated CatalogConfigBase that defined training dataset

        provenance:
            Provenance infomration

        Returns
        -------
        Model
            Newly read Model
        """
        with open(path, "rb") as fin:
            read_data = pickle.load(fin)

        if isinstance(read_data, Model):
            return read_data

        if provenance is None:
            provenance = {}

        return cls(read_data, creation_class_name, version, catalog_tag, provenance)

    @classmethod
    def wrap(
        cls,
        inpath: str,
        outpath: str,
        creation_class_name: str = "dummy",
        version: int = 0,
        catalog_tag: str | None = None,
        provenance: dict | None = None,
    ) -> Model:
        """Read a model from a file and write it as a `Model` if it is not already

        Parameters
        ----------
        inpath
            File to read

        outpath:
            File to write

        creation_class_name:
            Name of class that created this model

        version:
            Version of the model

        catalog_tag
            Associated CatalogConfigBase that defined training dataset

        provenance:
            Provenance information

        Returns
        -------
        Model
            Newly read & converted Model
        """
        the_model = cls.read(
            inpath, creation_class_name, version, catalog_tag, provenance
        )
        the_model.write(outpath)
        return the_model

    @classmethod
    def dump(
        cls,
        obj: Any,
        path: str,
        creation_class_name: str = "dummy",
        version: int = 0,
        catalog_tag: str | None = None,
        provenance: dict | None = None,
    ) -> Model:
        """Write an object to a model file

        This will promote it to a Model if it isn't already

        Parameters
        ----------
        obj
            Object to dump

        path:
            File to write

        creation_class_name:
            Name of class that created this model

        version:
            Version of the model

        catalog_tag
            Associated CatalogConfigBase that defined training dataset

        provenance:
            Provenance information

        Returns
        -------
        Model
            Newly converted Model
        """

        if isinstance(obj, Model):
            write_obj = obj
        else:
            if provenance is None:
                provenance = {}
            write_obj = cls(obj, creation_class_name, version, catalog_tag, provenance)

        write_obj.write(path)
        return write_obj

    def write(
        self,
        path: str,
    ) -> None:
        """Write a model to a file

        Parameters
        ----------
        path
            File to write
        """
        with open(path, "wb") as fout:
            pickle.dump(obj=self, file=fout, protocol=pickle.HIGHEST_PROTOCOL)
