from __future__ import annotations

from typing import Any

import numpy as np

from ceci.config import StageParameter as Param
from rail.core import common_params
from rail.core.configurable import Configurable


class Band(Configurable):
    """Information about a particular filter Band
    used for photometric redshifts.

    This is just a name and a redenning
    parameter: a_env.   

    The name is used for lookup and should be unique.
    It should match the name of the associated filter file.
    These are typically found in:
    rail_base/src/rail/examples_data/estimation_data/data/FILTER

    The a_env parameter can be computed from filter 
    curves using code in `rail_astro_tools`:

    https://github.com/LSSTDESC/rail_astro_tools/blob/main/src/rail/tools/filter_tools.py
    """
    
    config_options: dict[str, Param] = dict(
        name=Param(str, None, required=True, msg="Name for this band"),
        a_env=Param(float, None, required=True, msg="Reddening parameter"),
    )

    yaml_tag: str = "Band"

    _band_dict: dict[str, Band] = {}

    def __init__(self, **kwargs: Any) -> None:
        """C'tor

        Parameters
        ----------
        kwargs: Any
            Configuration parameters for this Band, must match
            class.config_options data members
        """
        Configurable.__init__(self, **kwargs)
        if self.config.name in self._band_dict:  # pragma: no cover
            raise KeyError(f"Duplicate band {self.config.name}")
        self._band_dict[self.config.name] = self

    @classmethod
    def get_band(cls, key: str) -> Band:
        """Return a particular band"""
        return cls._band_dict[key]

    @classmethod
    def clear(cls) -> None:
        cls._band_dict.clear()


class CatalogTag(Configurable):
    """Helper class to specify particulars about a particular catalog.

    This keep track of things like the mapping between band names and columns,
    names of special columns like the objectId and the 'true' redshift, gaurd
    values used to specify non-detections or non-observations.

    Expected usage is that user will define a yaml file with the various
    datasets that they wish to use with the following example syntax:

    .. highlight:: yaml
    .. code-block:: yaml

      Catalogs:
        - CatalogTag:
            name: "com_cam"
            mag_column_template: "{band}_cModelMag"
            mag_err_column_template: "{band}_cModelMagErr"
            filter_template: "comcam_{band}"
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

    .. highlight:: yaml
    .. code-block:: yaml

    Notes
    -----
    The files in the yaml file should match the class config_options.

    The mapping between bands and magnitude, magnitude error columns
    is constructed by looping over the band_list add resolving
    mag_column_template and mag_err_column_template.

    Similarly, the mapping between bands and filters is constructed
    by looping over the band_list add resolving filter_template.

    The `bands` field is used to provide band-specific overrides.
    At a minimum this must include 5-sigma limiting magnitdues.

    But it may also include specfic overrides for the names of the
    mag_column, mag_err_columns and filter template file.
    """
    
    config_options = dict(
        name=Param(
            str, None, required=True, msg="Tag to associate to this Catalog type"
        ),
        mag_column_template=Param(
            str,
            None,
            required=True,
            msg="Template to construct band magnitude column names",
        ),
        mag_err_column_template=Param(
            str,
            None,
            required=True,
            msg="Template to construct band magnitude error column names",
        ),
        filter_template=Param(
            str, None, required=True, msg="Template to filter file names"
        ),
        redshift_col=Param(str, "redshift", msg="Name of the redshift column"),
        object_id_col=Param(str, "objectId", msg="Name of the object id column"),
        nondetect_val=Param(float, np.nan, msg="Guard value for non-detections"),
        nonobserved_val=Param(float, np.inf, msg="Guard value for non-observations"),
        hdf5_groupname=Param(
            str, "", msg="HDF5 group with photometry information, blank for top-level"
        ),
        ref_band=Param(str, "i", msg="Reference band"),
        zp_error_default=Param(float, 0.1, msg="Default zero-point error"),
        replace_error_val_default=Param(float, 0.1, msg="Replacement error value"),
        bands=Param(dict, {}, msg="Band definitions"),
        band_list=Param(list, [], msg="Names of bands in order"),
        zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
        zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
        nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
        chunk_size=Param(
            int, 10000, msg="Number of object per chunk for parallel processing",
        ),
        calc_summary_stats=Param(
            dtype=bool,
            default=False,
            msg="Compute summary statistics",
        ),
        calculated_point_estimates=Param(
            dtype=list,
            default=[],
            msg="List of strings defining which point estimates to automatically calculate using `qp.Ensemble`."
            "Options include, 'mean', 'mode', 'median'.",
        ),
        recompute_point_estimates=Param(
            dtype=bool,
            default=False,
            msg="Force recomputation of point estimates",
        ),
    )

    yaml_tag: str = "CatalogTag"

    _tag_dict: dict[str, CatalogTag] = {}

    _active_tag: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        """C'tor

        Parameters
        ----------
        kwargs: Any
            Configuration parameters for this Band, must match
            class.config_options data members
        """
        Configurable.__init__(self, **kwargs)
        if self.config.name in self._tag_dict:
            raise KeyError(f"Duplicate CatalogTag {self.config.name}")
        self._tag_dict[self.config.name] = self
        self._base_dict: dict = {}

    def _build_base_dict(self) -> dict:
        """Construct the dict of overrides for the shared paramters"""
        set_params: list[str] = [
            "redshift_col",
            "nondetect_val",
            "nonobserved_val",
            "hdf5_groupname",
            "zmin",
            "zmax",
            "nzbins",
            "chunk_size",
            "calc_summary_stats",
            "calculated_point_estimates",
            "recompute_point_estimates",            
        ]
        base_dict: dict = {key: self.config[key] for key in set_params}
        shared_pars = self._build_shared_params()
        base_dict.update(
            ref_band=self._build_ref_band(),
            **shared_pars,
        )
        return base_dict

    @classmethod
    def clear(cls) -> None:
        cls._active_tag = None
        cls._tag_dict.clear()

    def band_name_dict(self) -> dict[str, str]:
        """Retrun the mapping from band to column names"""
        ret_dict: dict[str, str] = {}
        for band_name in self.config.band_list:
            band_info: dict = self.config.bands[band_name]
            if "mag_column_name" in band_info:
                mag_column_name = band_info["mag_column_name"]
            else:
                mag_column_name = self.config.mag_column_template.format(band=band_name)
            ret_dict[band_name] = mag_column_name
        return ret_dict

    def _build_shared_params(self) -> dict:
        """Build the various shared params"""

        bands: list[str] = []
        err_bands: list[str] = []
        err_dict: dict[str, str] = {}
        mag_limits: dict[str, float] = {}
        band_a_env: dict[str, float] = {}
        filter_list: list[str] = []
        zp_errors: list[float] = []
        replace_error_vals: list[float] = []

        for band_name in self.config.band_list:
            band_info: dict = self.config.bands[band_name]
            if "filter" in band_info:
                filter_name = band_info["filter"]
            else:
                filter_name = self.config.filter_template.format(band=band_name)
            filter_list.append(filter_name)
            band_val = Band.get_band(filter_name)
            band_a_env[band_name] = band_val.config.a_env
            if "mag_column_name" in band_info:
                mag_column_name = band_info["mag_column_name"]
            else:
                mag_column_name = self.config.mag_column_template.format(band=band_name)
            if "mag_err_column_name" in band_info:
                mag_err_column_name = band_info["mag_err_column_name"]
            else:
                mag_err_column_name = self.config.mag_err_column_template.format(
                    band=band_name
                )
            bands.append(mag_column_name)
            err_bands.append(mag_err_column_name)
            err_dict[mag_column_name] = mag_err_column_name
            mag_limits[mag_column_name] = band_info["mag_limit"]
            if "zp_error" in band_info:  # pragma: no cover
                zp_err = band_info["zp_error"]
            else:
                zp_err = self.config.zp_error_default
            zp_errors.append(zp_err)
            if "replace_error_val" in band_info:  # pragma: no cover
                rev = band_info["replace_error_val"]
            else:
                rev = self.config.replace_error_val_default
            replace_error_vals.append(rev)

        return dict(
            bands=bands,
            err_bands=err_bands,
            err_dict=err_dict,
            mag_limits=mag_limits,
            band_a_env=band_a_env,
            filter_list=filter_list,
            zp_errors=zp_errors,
            replace_error_vals=replace_error_vals,
        )

    def _build_ref_band(self) -> str:
        """Contruct the name of the reference band"""
        return self.config.mag_column_template.format(band=self.config.ref_band)

    def apply_tag(self) -> None:
        """Apply this tag"""
        self._base_dict = self._build_base_dict()
        common_params.set_param_defaults(**self._base_dict)

    @classmethod
    def get_active_tag(cls) -> CatalogTag | None:
        """Return the currently active tag"""
        if cls._active_tag is None:  # pragma: no cover
            return None
        return cls.get_tag(cls._active_tag)

    @classmethod
    def get_tag(cls, tag: str) -> CatalogTag:
        if tag not in cls._tag_dict:  # pragma: no cover
            raise KeyError(
                f"Did not find tag: {tag} in known CatalogTags: {list(cls._tag_dict.keys())}"
            )
        return cls._tag_dict[tag]

    @classmethod
    def apply(cls, tag: str) -> None:
        """Activate a particular tag"""
        tag_object = cls.get_tag(tag)
        cls._active_tag = tag
        tag_object.apply_tag()
