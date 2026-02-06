from __future__ import annotations

from rail.core.factory_mixin import RailFactoryMixin
from .catalog_tag import Band, CatalogTag


class BandFactory(RailFactoryMixin):
    """Factory class to make Band

    Expected usage is that user will define a yaml file with the various
    band that they wish to use with the following example syntax:

    .. highlight:: yaml
    .. code-block:: yaml

      Bands:
        - Band
            filter: comcam_u
            a_env: 4.81
        - Band
            filter: comcam_g
            a_env: 3.64


    """

    yaml_tag: str = "Bands"

    client_classes = [Band]

    _instance: BandFactory | None = None

    def __init__(self) -> None:
        """C'tor, build an empty BandFactor"""
        RailFactoryMixin.__init__(self)
        self._bands = self.add_dict(Band)

    @classmethod
    def get_bands(cls) -> dict[str, Band]:
        """Return the dict of all the Bands"""
        return cls.instance().bands

    @classmethod
    def get_band_names(cls) -> list[str]:
        """Return the names of the Bands"""
        return list(cls.instance().bands.keys())

    @classmethod
    def get_band(cls, name: str) -> Band:
        """Get Band by it's assigned name

        Parameters
        ----------
        name:
            Name of the Band to return

        Returns
        -------
        Band:
            Band in question
        """
        try:
            return cls.instance().bands[name]
        except KeyError as msg:  # pragma: no cover
            raise KeyError(
                f"Band named {name} not found in BandFactory "
                f"{list(cls.instance().bands.keys())}"
            ) from msg

    @classmethod
    def add_band(cls, band: Band) -> None:
        """Add a particular Band to the factory"""
        cls.instance().add_to_dict(band)

    @property
    def bands(self) -> dict[str, Band]:
        """Return the dictionary of Bands"""
        return self._bands

    def clear_instance(self) -> None:
        """Clear out the contents of the factory"""
        Band.clear()
        RailFactoryMixin.clear_instance(self)

    def print_instance_contents(self) -> None:
        """Print the contents of the factory"""
        print("----------------")
        print("Bands:")
        RailFactoryMixin.print_instance_contents(self)


class CatalogTagFactory(RailFactoryMixin):
    """Factory class to make CatalogTag

    Expected usage is that user will define a yaml file with the various
    band that they wish to use with the following example syntax:

    .. highlight:: yaml
    .. code-block:: yaml

      CatalogTags:
        - CatalogTag:
            tag: "com_cam"
            mag_column_template: "{band}_cModelMag"
            mag_err_column_template: "{band}_cModelMagErr"
            filter_temlpate: "comcam_{band}"
            band_list: ['u', 'g', 'r', 'i', 'z', 'y']
            bands:
            u:
              mag_limit: 26.4
            g:
              mag_limit: 27.8
            r:
              mag_limit: 27.1
            i:
              mag_limit: 26.7
            z:
              mag_limit: 25.8
            y:
              mag_limit: 24.6
    """

    yaml_tag: str = "CatalogTags"

    client_classes = [CatalogTag]

    _instance: CatalogTagFactory | None = None

    def __init__(self) -> None:
        """C'tor, build an empty BandFactor"""
        RailFactoryMixin.__init__(self)
        self._catalog_tags = self.add_dict(CatalogTag)

    def clear_instance(self) -> None:
        """Clear out the contents of the factory"""
        CatalogTag.clear()
        RailFactoryMixin.clear_instance(self)

    @classmethod
    def get_catalog_tags(cls) -> dict[str, CatalogTag]:
        """Return the dict of all the CatalogTags"""
        return cls.instance().catalog_tags

    @classmethod
    def get_catalog_tag_names(cls) -> list[str]:
        """Return the names of the CatalogTags"""
        return list(cls.instance().catalog_tags.keys())

    @classmethod
    def get_catalog_tag(cls, name: str) -> CatalogTag:
        """Get a CatalogTag by it's assigned name

        Parameters
        ----------
        name:
            Name of the CatalogTag to return

        Returns
        -------
        CatalogTag:
            CatalogTag in question
        """
        try:
            return cls.instance().catalog_tags[name]
        except KeyError as msg:  # pragma: no cover
            raise KeyError(
                f"CatalogTag named {name} not found in CatalogTagFactory "
                f"{list(cls.instance().catalog_tags.keys())}"
            ) from msg

    @classmethod
    def add_catalog_tag(cls, catalog_tag: CatalogTag) -> None:
        """Add a particular CatalogTag to the factory"""
        cls.instance().add_to_dict(catalog_tag)

    @property
    def catalog_tags(self) -> dict[str, CatalogTag]:
        """Return the dictionary of CatalogTags"""
        return self._catalog_tags

    def print_instance_contents(self) -> None:
        """Print the contents of the factory"""
        print("----------------")
        print("CatalogTags:")
        RailFactoryMixin.print_instance_contents(self)
