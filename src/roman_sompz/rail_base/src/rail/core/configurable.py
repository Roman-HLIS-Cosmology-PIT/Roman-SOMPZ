from __future__ import annotations

from typing import Any

from ceci.config import StageConfig, StageParameter


class Configurable:
    """Base class used to attach Ceci.StageParamters to a class

    This implements:

    1. being able to define parameters that are attached to a class
    2. being able to create an object of that class from a dict with the required paramters
    3. checking that all the required parameters are present and of the correct types
    4. check that there are no additional parameters given
    5. being able to write a snapshot of the current values of the paramters to yaml

    Subclasses should:

    1. add parameters to the config_options class member
    2. set the yaml_tag class member to a unique value

    """

    config_options: dict[str, StageParameter] = dict(
        name=StageParameter(str, 0.0, fmt="%s", msg="Name for the configurable"),
    )

    yaml_tag: str = ""

    @staticmethod
    def merge_named_lists(
        input_lists: list[list[Configurable]],
    ) -> list[Configurable]:  # pragma: no cover
        """Merge lists, removing duplicate items

        Parameters
        ----------
        input_lists
            Lists to be merged

        Returns
        -------
        list[Configurable]
            The merged list
        """
        sort_dict: dict[str, Configurable] = {}
        for list_ in input_lists:
            for item_ in list_:
                sort_dict[item_.config.name] = item_
        return list(sort_dict.values())

    @classmethod
    def full_class_name(cls) -> str:
        """Return the full name of the class, including the parent module"""
        return f"{cls.__module__}.{cls.__name__}"

    def __init__(self, **kwargs: Any) -> None:
        """C'tor

        Parameters
        ----------
        **kwargs
            Configuration parameters for this object, must match
            class.config_options data members
        """
        self._config = StageConfig(**self.config_options)
        self._set_config(**kwargs)

    def _set_config(self, **kwargs: Any) -> None:
        kwcopy = kwargs.copy()
        for key in self.config.keys():
            if key in kwargs:
                self.config[key] = kwcopy.pop(key)
            else:  # pragma: no cover
                attr = self.config.get(key)
                if attr.required:
                    raise ValueError(f"Missing configuration option {key}")
                self.config[key] = attr.default
        if kwcopy:  # pragma: no cover
            raise ValueError(
                f"Unrecogonized configruation parameters {kwcopy.keys()} "
                f"for type {type(self)}.  "
                f"Known parameters are {list(self.config.to_dict().keys())}"
            )

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default_value: Any | None = None) -> Any:
        """Get a particular item by key, return default value if it is not present"""
        try:
            return self[key]
        except KeyError:  # pragma: no cover
            return default_value

    @property
    def config(self) -> StageConfig:
        """Return the underlying configuration"""
        return self._config

    def __repr__(self) -> str:
        return f"{self.config.name}"

    def to_yaml_dict(self) -> dict[str, dict[str, Any]]:
        """Create a yaml-convertable dict for this object"""
        return {self.yaml_tag: self.config.to_dict()}
