import os

import pytest

from rail.utils.catalog_tag import Band, CatalogTag
from rail.utils.catalog_tag_factory import BandFactory, CatalogTagFactory
from rail.utils import catalog_utils


def test_load_catalog_tag_yaml() -> None:

    # clear the catalog configurations
    catalog_utils.clear()

    # Load defaults
    catalog_utils.apply_defaults("com_cam")

    assert catalog_utils.get_active_tag().config.name == "com_cam"
    assert catalog_utils.get_tag('com_cam').config.name == "com_cam"

    # Printing
    BandFactory.print_contents()
    CatalogTagFactory.print_contents()

    # test the band stuff
    the_bands = BandFactory.get_bands()
    assert isinstance(the_bands["comcam_u"], Band)

    the_band_names = BandFactory.get_band_names()
    assert "comcam_u" in the_band_names

    a_band = BandFactory.get_band("comcam_u")
    assert isinstance(a_band, Band)

    assert a_band.full_class_name().find("Band") >= 0
    assert a_band.get("a_env") == 4.81
    a_band.to_yaml_dict()["Band"]["a_env"] = 4.81

    # test the catalog tag stuff
    the_catalog_tags = CatalogTagFactory.get_catalog_tags()
    assert isinstance(the_catalog_tags["com_cam"], CatalogTag)

    the_catalog_tag_names = CatalogTagFactory.get_catalog_tag_names()
    assert "com_cam" in the_catalog_tag_names

    a_catalog_tag = CatalogTagFactory.get_catalog_tag("com_cam")
    assert isinstance(a_catalog_tag, CatalogTag)

    # Test the interactive stuff
    catalog_utils.clear()
    BandFactory.add_band(a_band)
    CatalogTagFactory.add_catalog_tag(a_catalog_tag)

    check_band = BandFactory.get_band("comcam_u")
    assert isinstance(check_band, Band)

    check_catalog_tag = CatalogTagFactory.get_catalog_tag("com_cam")
    assert isinstance(check_catalog_tag, CatalogTag)

    # check writing the yaml dict
    CatalogTagFactory.write_yaml("tests/temp.yaml")
    CatalogTagFactory.clear()
    CatalogTagFactory.load_yaml("tests/temp.yaml")
    os.unlink("tests/temp.yaml")

    check_catalog_tag = CatalogTagFactory.get_catalog_tag("com_cam")
    assert isinstance(check_catalog_tag, CatalogTag)
