from __future__ import annotations

from rail.core.utils import find_rail_file

from .catalog_utils_old import CatalogConfigBase
from .catalog_tag_factory import BandFactory, CatalogTagFactory
from .catalog_tag import CatalogTag

DEFAULT_CATAlOG_TAG_FILE: str = find_rail_file("examples_data/catalog_data/default_catalogs.yaml")
LOADED_DEFAULTS = False


def load_yaml(yaml_file: str) -> None:
    """Load catalog configurations from a yaml file"""
    BandFactory.load_yaml(yaml_file)
    CatalogTagFactory.load_yaml(yaml_file)


def clear() -> None:
    """Clear the configurations"""
    global LOADED_DEFAULTS
    LOADED_DEFAULTS = False
    BandFactory.clear()
    CatalogTagFactory.clear()


def get_tag(tag: str) -> CatalogTag:
    """Return a particular tag"""
    return CatalogTag.get_tag(tag)


def get_active_tag() -> CatalogTag | None:
    """Return the currently active tag"""
    return CatalogTag.get_active_tag()


def apply_defaults(tag: str) -> None:
    """Apply a particular catalog configuraiton"""
    global LOADED_DEFAULTS
    if not LOADED_DEFAULTS:
        load_yaml(DEFAULT_CATAlOG_TAG_FILE)
        LOADED_DEFAULTS = True
    CatalogTag.apply(tag)


apply_defaults_old = CatalogConfigBase.apply
