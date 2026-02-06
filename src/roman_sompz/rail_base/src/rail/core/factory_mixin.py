import os
from typing import Any, TypeVar

import yaml

from .configurable import Configurable

T = TypeVar("T", bound="RailFactoryMixin")
C = TypeVar("C", bound="Configurable")


class RailFactoryMixin:
    """A Factory can make specific type or types of components, assign
    names to each, and keep track of what it has made.

    This implements:

    1. having a single instance of each sub-class of factory,
    2. having the factory be abble to handle one or more client classes,
    3. creating objects of the sub-classes from yaml,
    4. keeping track of the created object in dictionaries keyed by name,
    5. writing the current content of the factory to a yaml file.
    """

    client_classes: list[type[Configurable]]

    _instance: Any | None = None

    yaml_tag: str = ""

    def __init__(self) -> None:
        self._the_dicts: dict[str, dict] = {}
        self.loaded_files: list[str] = []

    def add_dict(self, configurable_class: type[C]) -> dict[str, C]:
        """Add a dictionary for one of the client classes

        Parameters
        ----------
        configurable_class: type[C]
            Client class in question

        Returns
        -------
        dict[str, C]:
            Newly created emtpy dict

        Notes
        -----
        This should be called in the c'tor of the factory for each of the
        client classes
        """
        a_dict: dict[str, C] = {}
        self._the_dicts[configurable_class.yaml_tag] = a_dict
        return a_dict

    def add_to_dict(self, the_object: C) -> None:
        """Add an object one of 'C' client class to the corresponding dict

        Parameters
        ----------
        the_object: C
            Object in question

        Notes
        -----
        This should be called by the factory when inserting objects of the client classes
        """
        the_class = type(the_object)
        try:
            the_dict = self._the_dicts[the_class.yaml_tag]
        except KeyError as missing_key:
            raise KeyError(
                f"Tried to add object with {the_class.yaml_tag}, "
                "but factory has {list(self._the_dicts.keys())}"
            ) from missing_key
        if the_object.config.name in the_dict:  # pragma: no cover
            raise KeyError(f"{the_class} {the_object.config.name} is already defined")
        the_dict[the_object.config.name] = the_object

    def load_object_from_yaml_tag(
        self, configurable_class: type[C], yaml_tag: dict[str, Any]
    ) -> None:
        """Create and add an object of one of the client classes from a yaml tag

        Parameters
        ----------
        configurable_class: type[C]
            Client class in question

        yaml_tag: dict[str, Any]
            Yaml used to create the object
        """
        the_object = configurable_class(**yaml_tag)
        self.add_to_dict(the_object)

    @classmethod
    def instance(cls: type[T]) -> T:
        """Return the singleton instance of the factory"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        """Clear the contents of the factory"""
        cls.instance().clear_instance()

    @classmethod
    def print_contents(cls) -> None:
        """Print the contents of the factory"""
        cls.instance().print_instance_contents()

    @classmethod
    def load_yaml(cls, yaml_file: str) -> None:
        """Load a yaml file

        Parameters
        ----------
        yaml_file: str
            File to read and load

        Notes
        -----
        See class helpstring for yaml format
        """
        cls.instance().load_instance_yaml(yaml_file)

    @classmethod
    def load_yaml_tag(
        cls,
        yaml_config: list[dict[str, Any]],
        from_file: str,
    ) -> None:  # pragma: no cover
        """Load from a yaml tag

        Parameters
        ----------
        yaml_config: list[dict[str, Any]]
            Yaml tag used to load

        from_file: str
            File it was loaded from, used to aviod reloading

        Notes
        -----
        See class helpstring for yaml format
        """
        cls.instance().load_instance_yaml_tag(yaml_config, from_file)

    @classmethod
    def to_yaml_dict(cls) -> dict:
        """Construct a dictionary to write to a yaml file"""
        return cls.instance().to_instance_yaml_dict()

    @classmethod
    def write_yaml(cls, yaml_file: str) -> None:
        """Write to a yaml file

        Parameters
        ----------
        yaml_file: str
            Yaml file to write

        Notes
        -----
        See class helpstring for yaml format
        """
        the_dict = cls.to_yaml_dict()
        with open(os.path.expandvars(yaml_file), mode="w", encoding="utf-8") as fout:
            yaml.dump(the_dict, fout)

    def clear_instance(self) -> None:
        """Clear out the contents of the factory"""
        self.loaded_files.clear()
        for val in self._the_dicts.values():
            val.clear()

    def print_instance_contents(self) -> None:
        """Print the contents of the factory"""
        for dict_name, a_dict in self._the_dicts.items():
            print("----------------")
            print(f"{dict_name}")
            for item_name, item in a_dict.items():
                print(f"  {item_name}: {item}")

    def load_instance_yaml_tag(
        self,
        yaml_config: list[dict[str, Any]],
        from_file: str,
    ) -> None:
        """Read a yaml tag and load the factory accordingy

        Parameters
        ----------
        yaml_config: list[dict[str, Any]]
            Yaml tag to load

        from_file: str
            File it was loaded from, used to aviod reloading

        Notes
        -----
        See class description for yaml file syntax
        """
        if from_file in self.loaded_files:  # pragma: no cover
            print(f"{from_file} already loaded by {type(self)}")
            return
        self.loaded_files.append(from_file)

        for yaml_item in yaml_config:
            found_key = False
            for val in self.client_classes:
                if val.yaml_tag in yaml_item:
                    found_key = True
                    yaml_vals = yaml_item[val.yaml_tag]
                    self.load_object_from_yaml_tag(val, yaml_vals)
            if not found_key:  # pragma: no cover
                good_keys = [val.yaml_tag for val in self.client_classes]
                raise KeyError(f"Expecting one of {good_keys} not: {yaml_item.keys()})")

    def load_instance_yaml(self, yaml_file: str) -> None:
        """Read a yaml file and load the factory accordingly

        Parameters
        ----------
        yaml_file: str
            File to read

        Notes
        -----
        See class description for yaml file syntax
        """
        if yaml_file in self.loaded_files:  # pragma: no cover
            print(f"{yaml_file} already loaded by {type(self)}")
            return

        with open(os.path.expandvars(yaml_file), encoding="utf-8") as fin:
            yaml_data = yaml.safe_load(fin)

        try:
            this_config = yaml_data[self.yaml_tag]
        except KeyError as missing_key:
            raise KeyError(
                f"Did not find key {self.yaml_tag} in {yaml_file}"
            ) from missing_key

        self.load_instance_yaml_tag(this_config, yaml_file)

    def to_instance_yaml_dict(self) -> dict:
        """Write the content of the factory to a dict for export to a yaml file"""
        main_list: list[dict] = []
        for _a_dict_name, a_dict in self._the_dicts.items():
            for value_ in a_dict.values():
                main_list.append(value_.to_yaml_dict())
        return {self.yaml_tag: main_list}
