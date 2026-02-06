from __future__ import annotations
    
from typing import Any, TypeVar
import warnings

import numpy as np

from rail.core import common_params

T = TypeVar("T", bound="CatalogConfigBase")


class CatalogConfigBase:
    """Class that wraps the settings of shared configuration
    parameters needed work with the particular column names
    in a given catalog type
    """

    _sub_classes_by_class: dict[str, type[CatalogConfigBase]] = {}
    _sub_classes: dict[str, type[CatalogConfigBase]] = {}

    tag: str | None = None
    bandlist: list[str] = []
    maglims: list[float] = []
    a_env: list[float] = []
    hdf5_groupname: str = ""
    band_template: str = ""
    band_err_template: str = ""
    filter_file_template: str = ""
    ref_band: str = ""
    redshift_col: str = ""
    object_id_col: str = ""
    lsst_err_band_replace: list[float] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    replace_error_vals: list[float] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    zp_errors: list[float] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    _active_tag: str | None = None
    _active_class: type | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        assert cls.tag is not None
        cls._sub_classes_by_class[cls.__name__] = cls
        cls._sub_classes[cls.tag] = cls

    @classmethod
    def active_tag(cls) -> str | None:
        """Return the currently active tag"""
        return cls._active_tag

    @classmethod
    def active_class(cls: type[T]) -> type[T] | None:
        """Return the currently active class"""
        return cls._active_class

    @classmethod
    def subclasses_by_class(cls) -> dict[str, type[CatalogConfigBase]]:
        """Return the dict of all the sub-classes keyed by class name"""
        return cls._sub_classes_by_class

    @classmethod
    def subclasses(cls) -> dict[str, type[CatalogConfigBase]]:
        """Return the dict of all the sub-classes keyed by tag"""
        return cls._sub_classes

    @classmethod
    def get_class(cls, class_name: str, module_name: str) -> type[CatalogConfigBase]:
        """Return a class my name, loading it if needed"""
        if class_name not in cls._sub_classes_by_class:  # pragma: no cover
            __import__(module_name)
        return cls._sub_classes_by_class[class_name]

    @classmethod
    def apply_class(cls, class_name: str) -> None:
        """Activate a particular class"""
        the_class = cls._sub_classes_by_class[class_name]
        cls._active_tag = the_class.tag
        cls._active_class = the_class
        cls._active_class._apply()

    @classmethod
    def apply(cls, tag: str) -> None:
        """Activate a particular tag"""
        cls._active_tag = tag
        cls._active_class = cls._sub_classes[tag]
        cls._active_class._apply()

    @classmethod
    def _apply(cls) -> None:
        warnings.warn("CatalogConfigBase is depreacted and will be removed, use CatalogTag instead")
        base_dict = cls.build_base_dict()
        common_params.set_param_defaults(**base_dict)
        cls._apply_hook()

    @classmethod
    def _apply_hook(cls) -> None:
        return

    @classmethod
    def band_name_dict(cls) -> dict[str, str]:
        """Retrun the mapping from band to column names"""
        return {band: cls.band_template.format(band=band) for band in cls.bandlist}

    @classmethod
    def _build_maglim_dict(cls) -> dict[str, float]:
        """Construct the mapping from band to limiting magnitude"""
        maglim_dict = {}
        for band, limx in zip(cls.bandlist, cls.maglims):
            maglim_dict[cls.band_template.format(band=band)] = limx
        return maglim_dict

    @classmethod
    def _build_a_env_dict(cls) -> dict[str, float]:
        """Construct the mapping from band to a_env"""
        a_env_dict = {}
        for band, limx in zip(cls.bandlist, cls.a_env):
            a_env_dict[cls.band_template.format(band=band)] = limx
        return a_env_dict

    @classmethod
    def _build_band_names(cls) -> list[str]:
        """Construct the list of column names for the filter magnitudes"""
        return [cls.band_template.format(band=band) for band in cls.bandlist]

    @classmethod
    def _build_band_err_names(cls) -> list[str]:
        """Construct the list of column names for uncertainties on the the filter magnitudes"""
        return [cls.band_err_template.format(band=band) for band in cls.bandlist]

    @classmethod
    def _build_err_dict(cls) -> dict[str, str| None]:
        """Construct the mapping from magnitude columns to associated uncertainties"""
        the_dict: dict[str, str|None] = {
            cls.band_template.format(band=band): cls.band_err_template.format(band=band)
            for band in cls.bandlist
        }
        the_dict[cls.redshift_col] = None
        return the_dict

    @classmethod
    def _build_ref_band(cls, ref_band: str = "i") -> str:
        """Contruct the name of the reference band"""
        return cls.band_template.format(band=ref_band)

    @classmethod
    def _build_filter_file_bandlist(cls) -> list[str]:
        """Contruct the list names used to find the filter files"""
        return [cls.filter_file_template.format(band=band) for band in cls.bandlist]

    @classmethod
    def build_base_dict(cls) -> dict:
        """Construct the dict of overrides for the shared paramters"""
        base_dict: dict = {}
        base_dict.update(
            bands=cls._build_band_names(),
            err_bands=cls._build_band_err_names(),
            err_dict=cls._build_err_dict(),
            mag_limits=cls._build_maglim_dict(),
            band_a_env=cls._build_a_env_dict(),
            ref_band=cls._build_ref_band(cls.ref_band),
            redshift_col=cls.redshift_col,
            nondetect_val=np.nan,
            nonobserved_val=np.inf,
            hdf5_groupname=cls.hdf5_groupname,
            replace_error_vals=cls.replace_error_vals,
            filter_list=cls._build_filter_file_bandlist(),
            zp_errors=cls.zp_errors,
        )
        return base_dict


class HscCatalogConfig(CatalogConfigBase):
    """Configuration for HSC data"""

    tag = "hsc"
    bandlist = ['g', 'r', 'i', 'z', 'y']
    maglims = [27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "HSC{band}_cmodel_dered"
    band_err_template = "{band}_cmodel_magerr"
    filter_file_template = "hsc_{band}"
    ref_band = "i"
    redshift_col = "specz_redshift"
    object_id_col = "object_id"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1]


class Dc2CatalogConfig(CatalogConfigBase):
    """Configuration for DC2 data"""

    tag = "dc2"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "mag_{band}_cModel_obj_dered"
    band_err_template = "magerr_{band}_cModel_obj"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i"
    redshift_col = "redshift_true"
    object_id_col = "objectId_obj"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


class RubinCatalogConfig(CatalogConfigBase):
    """Configuration for Rubin project data"""

    tag = "rubin"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "LSST_obs_{band}"
    band_err_template = "LSST_obs_{band}_err"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i"
    redshift_col = "true_redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


class Roman3BandCatalogConfig(CatalogConfigBase):
    """Configuration for Rubin data from Roman / Rubin simulations"""

    tag = "roman_3band"
    bandlist = ["Y106", "J129", "H158"]
    maglims = [26.4, 26.4, 26.4]
    a_env = [1.14025753, 0.83118224, 0.59966235]
    band_template = "ROMAN_obs_{band}"
    band_err_template = "ROMAN_obs_{band}_err"
    filter_file_template = "roman_{band}"
    ref_band = "Y106"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1]


class Roman7BandCatalogConfig(CatalogConfigBase):
    """Configuration for Rubin data from Roman / Rubin simulations"""

    tag = "roman_7band"
    bandlist = ["Z087", "Y106", "J129", "W146", "H158", "F184", "K213"]
    maglims = [27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4]
    a_env = [
        1.57491325,
        1.14025753,
        0.83118224,
        0.68098202,
        0.59966235,
        0.46923204,
        0.37072579,
    ]
    band_template = "ROMAN_obs_{band}"
    band_err_template = "ROMAN_obs_{band}_err"
    filter_file_template = "roman_{band}"
    ref_band = "Y106"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


class RomanRubinCatalogConfig(CatalogConfigBase):
    """Configuration for Rubin data from Roman / Rubin simulations"""

    tag = "roman_rubin"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "LSST_obs_{band}"
    band_err_template = "LSST_obs_{band}_err"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


class ComCamCatalogConfig(CatalogConfigBase):
    """Configuration for ComCam data"""

    tag = "com_cam"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [26.4, 27.8, 27.1, 26.7, 25.8, 24.6]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "{band}_cModelMag"
    band_err_template = "{band}_cModelMagErr"
    filter_file_template = "comcam_{band}"
    ref_band = "i"
    redshift_col = "redshift"


class RubinFLCatalogConfig(CatalogConfigBase):
    """Configuration for Rubin first look data """

    tag = "rubin_fl"
    bandlist = ['u', 'g', 'r', 'i']
    maglims = [26.4, 27.8, 27.1, 26.7]
    a_env = [4.81, 3.64, 2.70, 2.06]
    band_template = "{band}_cModelMag"
    band_err_template = "{band}_cModelMagErr"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    replace_error_vals = [0.1, 0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1, 0.1]


class ComCamGaapCatalogConfig(CatalogConfigBase):
    """Configuration for ComCam data"""

    tag = "com_cam_gaap"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [26.4, 27.8, 27.1, 26.7, 25.8, 24.6]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "{band}_gaap1p0Mag"
    band_err_template = "{band}_gaap1p0MagErr"
    filter_file_template = "comcam_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        
class ComCamGaap4BandCatalogConfig(CatalogConfigBase):
    """Configuration for ComCam data"""

    tag = "com_cam_gaap_4band"
    bandlist = ['g', 'r', 'i', 'z']
    maglims = [27.8, 27.1, 26.7, 25.8]
    a_env = [3.64, 2.70, 2.06, 1.58]
    band_template = "{band}_gaap1p0Mag"
    band_err_template = "{band}_gaap1p0MagErr"
    filter_file_template = "comcam_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [ 0.1, 0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1, 0.1]


class ComCamKronCatalogConfig(CatalogConfigBase):
    """Configuration for ComCam data"""

    tag = "com_cam_kron"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [26.4, 27.8, 27.1, 26.7, 25.8, 24.6]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "{band}_kronMag"
    band_err_template = "{band}_kronMagErr"
    filter_file_template = "comcam_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


class ComCamEuclidCatalogConfig(CatalogConfigBase):
    """Configuration for ComCam data"""

    tag = "com_cam_euclid"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [26.4, 27.8, 27.1, 26.7, 25.8, 24.6]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "{band}_gaap1p0Mag"
    band_err_template = "{band}_gaap1p0MagErr"
    filter_file_template = "comcam_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    @classmethod
    def band_name_dict(cls) -> dict[str, str]:
        bands = super().band_name_dict()
        bands["vis"] = "euclid_vis_psfMag"
        bands["Y"] = "euclid_y_unifMag"
        bands["J"] = "euclid_j_unifMag"
        bands["H"] = "euclid_h_unifMag"
        return bands

    @classmethod
    def _build_maglim_dict(cls) -> dict[str, float]:
        maglim_dict = super()._build_maglim_dict()
        maglim_dict["euclid_vis_psfMag"] = 26.0
        maglim_dict["euclid_y_unifMag"] = 23.8
        maglim_dict["euclid_j_unifMag"] = 24.0
        maglim_dict["euclid_h_unifMag"] = 24.0
        return maglim_dict

    @classmethod
    def _build_a_env_dict(cls) -> dict[str, float]:
        a_env_dict = super()._build_a_env_dict()
        a_env_dict["euclid_vis_psfMag"] = 0.0
        a_env_dict["euclid_y_unifMag"] = 0.0
        a_env_dict["euclid_j_unifMag"] = 0.0
        a_env_dict["euclid_h_unifMag"] = 0.0
        return a_env_dict

    @classmethod
    def _build_band_names(cls) -> list[str]:
        bands = [cls.band_template.format(band=band) for band in cls.bandlist]
        bands += [
            "euclid_vis_psfMag",
            "euclid_y_unifMag",
            "euclid_j_unifMag",
            "euclid_h_unifMag",
        ]
        return bands

    @classmethod
    def _build_band_err_names(cls) -> list[str]:
        band_errs = [cls.band_err_template.format(band=band) for band in cls.bandlist]
        band_errs += [
            "euclid_vis_psfMagErr",
            "euclid_y_unifMagErr",
            "euclid_j_unifMagErr",
            "euclid_h_unifMagErr",
        ]
        return band_errs

    @classmethod
    def _build_err_dict(cls) -> dict[str, str | None]:
        the_dict = super()._build_err_dict()
        the_dict["euclid_vis_psfMag"] = "euclid_vis_psfMagErr"
        the_dict["euclid_y_unifMag"] = "euclid_y_unifMagErr"
        the_dict["euclid_j_unifMag"] = "euclid_j_unifMagErr"
        the_dict["euclid_h_unifMag"] = "euclid_h_unifMagErr"
        return the_dict

    @classmethod
    def _build_ref_band(cls, ref_band: str = "i") -> str:
        return cls.band_template.format(band=ref_band)

    @classmethod
    def _build_filter_file_bandlist(cls) -> list[str]:
        """Contruct the name of the reference band"""
        filter_list = [
            cls.filter_file_template.format(band=band) for band in cls.bandlist
        ]
        filter_list += [
            "euclid_vis",
            "euclid_Y",
            "euclid_J",
            "euclid_H",
        ]
        return filter_list


class ComCamEuclidNIRCatalogConfig(CatalogConfigBase):
    """Configuration for ComCam data"""

    tag = "com_cam_euclid_nir"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [26.4, 27.8, 27.1, 26.7, 25.8, 24.6]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "{band}_gaap1p0Mag"
    band_err_template = "{band}_gaap1p0MagErr"
    filter_file_template = "comcam_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    @classmethod
    def band_name_dict(cls) -> dict[str, str]:
        bands = super().band_name_dict()
        bands["Y"] = "euclid_y_unifMag"
        bands["J"] = "euclid_j_unifMag"
        bands["H"] = "euclid_h_unifMag"
        return bands

    @classmethod
    def _build_maglim_dict(cls) -> dict[str, float]:
        maglim_dict = super()._build_maglim_dict()
        maglim_dict["euclid_y_unifMag"] = 23.8
        maglim_dict["euclid_j_unifMag"] = 24.0
        maglim_dict["euclid_h_unifMag"] = 24.0
        return maglim_dict

    @classmethod
    def _build_a_env_dict(cls) -> dict[str, float]:
        a_env_dict = super()._build_a_env_dict()
        a_env_dict["euclid_y_unifMag"] = 0.0
        a_env_dict["euclid_j_unifMag"] = 0.0
        a_env_dict["euclid_h_unifMag"] = 0.0
        return a_env_dict

    @classmethod
    def _build_band_names(cls) -> list[str]:
        bands = [cls.band_template.format(band=band) for band in cls.bandlist]
        bands += [
            "euclid_y_unifMag",
            "euclid_j_unifMag",
            "euclid_h_unifMag",
        ]
        return bands

    @classmethod
    def _build_band_err_names(cls) -> list[str]:
        band_errs = [cls.band_err_template.format(band=band) for band in cls.bandlist]
        band_errs += [
            "euclid_y_unifMagErr",
            "euclid_j_unifMagErr",
            "euclid_h_unifMagErr",
        ]
        return band_errs

    @classmethod
    def _build_err_dict(cls) -> dict[str, str | None]:
        the_dict = super()._build_err_dict()
        the_dict["euclid_y_unifMag"] = "euclid_y_unifMagErr"
        the_dict["euclid_j_unifMag"] = "euclid_j_unifMagErr"
        the_dict["euclid_h_unifMag"] = "euclid_h_unifMagErr"
        return the_dict

    @classmethod
    def _build_ref_band(cls, ref_band: str = "i") -> str:
        return cls.band_template.format(band=ref_band)

    @classmethod
    def _build_filter_file_bandlist(cls) -> list[str]:
        """Contruct the name of the reference band"""
        filter_list = [
            cls.filter_file_template.format(band=band) for band in cls.bandlist
        ]
        filter_list += [
            "euclid_Y",
            "euclid_J",
            "euclid_H",
        ]
        return filter_list

class RomanPlusRubinCatalogConfig(CatalogConfigBase):
    """Configuration for Roman + Rubin bands in Roman / Rubin simulations"""

    tag = "roman_plus_rubin"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "LSST_obs_{band}"
    band_err_template = "LSST_obs_{band}_err"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    @classmethod
    def band_name_dict(cls) -> dict[str, str]:
        bands = super().band_name_dict()
        bands["Y"] = "ROMAN_obs_Y106"
        bands["J"] = "ROMAN_obs_J129"
        bands["H"] = "ROMAN_obs_H158"
        bands["F"] = "ROMAN_obs_F184"
        return bands

    @classmethod
    def _build_maglim_dict(cls) -> dict[str, float]:
        maglim_dict = super()._build_maglim_dict()
        maglim_dict["ROMAN_obs_Y106"] = 27.6
        maglim_dict["ROMAN_obs_J129"] = 27.8
        maglim_dict["ROMAN_obs_H158"] = 28.1
        maglim_dict["ROMAN_obs_F184"] = 27.5
        return maglim_dict

    @classmethod
    def _build_a_env_dict(cls) -> dict[str, float]:
        a_env_dict = super()._build_a_env_dict()
        a_env_dict["ROMAN_obs_Y106"] = 0.83118224
        a_env_dict["ROMAN_obs_J129"] = 0.68098202
        a_env_dict["ROMAN_obs_H158"] = 0.59966235
        a_env_dict["ROMAN_obs_F184"] = 0.46923204
        return a_env_dict

    @classmethod
    def _build_band_names(cls) -> list[str]:
        bands = [cls.band_template.format(band=band) for band in cls.bandlist]
        bands += [
            "ROMAN_obs_Y106",
            "ROMAN_obs_J129",
            "ROMAN_obs_H158",
            "ROMAN_obs_F184",
        ]
        return bands

    @classmethod
    def _build_band_err_names(cls) -> list[str]:
        band_errs = [cls.band_err_template.format(band=band) for band in cls.bandlist]
        band_errs += [
            "ROMAN_obs_Y106_err",
            "ROMAN_obs_J129_err",
            "ROMAN_obs_H158_err",
            "ROMAN_obs_F184_err",
        ]
        return band_errs

    @classmethod
    def _build_err_dict(cls) -> dict[str, str | None]:
        the_dict = super()._build_err_dict()
        the_dict["ROMAN_obs_Y106"] = "ROMAN_obs_Y106_err"
        the_dict["ROMAN_obs_J129"] = "ROMAN_obs_J129_err"
        the_dict["ROMAN_obs_H158"] = "ROMAN_obs_H158_err"
        the_dict["ROMAN_obs_F184"] = "ROMAN_obs_F184_err"
        return the_dict

    @classmethod
    def _build_ref_band(cls, ref_band: str = "i") -> str:
        return cls.band_template.format(band=ref_band)

    @classmethod
    def _build_filter_file_bandlist(cls) -> list[str]:
        """Contruct the name of the reference band"""
        filter_list = [
            cls.filter_file_template.format(band=band) for band in cls.bandlist
        ]
        filter_list += [
            "roman_Y106",
            "roman_J129",
            "roman_H158",
            "roman_F184",
        ]
        return filter_list


class Roman3BandPlusRubinCatalogConfig(CatalogConfigBase):
    """Configuration for Roman3Band + Rubin bands in Roman / Rubin simulations"""

    tag = "roman_3band_rubin"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "LSST_obs_{band}"
    band_err_template = "LSST_obs_{band}_err"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    @classmethod
    def band_name_dict(cls) -> dict[str, str]:
        bands = super().band_name_dict()
        bands["Y"] = "ROMAN_obs_Y106"
        bands["J"] = "ROMAN_obs_J129"
        bands["H"] = "ROMAN_obs_H158"
        return bands

    @classmethod
    def _build_maglim_dict(cls) -> dict[str, float]:
        maglim_dict = super()._build_maglim_dict()
        maglim_dict["ROMAN_obs_Y106"] = 26.5
        maglim_dict["ROMAN_obs_J129"] = 26.4
        maglim_dict["ROMAN_obs_H158"] = 26.4
        return maglim_dict

    @classmethod
    def _build_a_env_dict(cls) -> dict[str, float]:
        a_env_dict = super()._build_a_env_dict()
        a_env_dict["ROMAN_obs_Y106"] = 0.83118224
        a_env_dict["ROMAN_obs_J129"] = 0.68098202
        a_env_dict["ROMAN_obs_H158"] = 0.59966235
        return a_env_dict

    @classmethod
    def _build_band_names(cls) -> list[str]:
        bands = [cls.band_template.format(band=band) for band in cls.bandlist]
        bands += [
            "ROMAN_obs_Y106",
            "ROMAN_obs_J129",
            "ROMAN_obs_H158",
        ]
        return bands

    @classmethod
    def _build_band_err_names(cls) -> list[str]:
        band_errs = [cls.band_err_template.format(band=band) for band in cls.bandlist]
        band_errs += [
            "ROMAN_obs_Y106_err",
            "ROMAN_obs_J129_err",
            "ROMAN_obs_H158_err",
        ]
        return band_errs

    @classmethod
    def _build_err_dict(cls) -> dict[str, str | None]:
        the_dict = super()._build_err_dict()
        the_dict["ROMAN_obs_Y106"] = "ROMAN_obs_Y106_err"
        the_dict["ROMAN_obs_J129"] = "ROMAN_obs_J129_err"
        the_dict["ROMAN_obs_H158"] = "ROMAN_obs_H158_err"
        return the_dict

    @classmethod
    def _build_ref_band(cls, ref_band: str = "i") -> str:
        return cls.band_template.format(band=ref_band)

    @classmethod
    def _build_filter_file_bandlist(cls) -> list[str]:
        """Contruct the name of the reference band"""
        filter_list = [
            cls.filter_file_template.format(band=band) for band in cls.bandlist
        ]
        filter_list += [
            "roman_Y106",
            "roman_J129",
            "roman_H158",
        ]
        return filter_list


class Roman7BandPlusRubinCatalogConfig(CatalogConfigBase):
    """Configuration for Roman3Band + Rubin bands in Roman / Rubin simulations"""

    tag = "roman_7band_rubin"
    bandlist = ['u', 'g', 'r', 'i', 'z', 'y']
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "LSST_obs_{band}"
    band_err_template = "LSST_obs_{band}_err"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
    zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    @classmethod
    def band_name_dict(cls) -> dict[str, str]:
        bands = super().band_name_dict()

        bands["Z"] = "ROMAN_obs_Z087"
        bands["Y"] = "ROMAN_obs_Y106"
        bands["J"] = "ROMAN_obs_J129"
        bands["W"] = "ROMAN_obs_W146"
        bands["H"] = "ROMAN_obs_H158"
        bands["F"] = "ROMAN_obs_F184"
        bands["K"] = "ROMAN_obs_K213"
        return bands

    @classmethod
    def _build_maglim_dict(cls) -> dict[str, float]:
        maglim_dict = super()._build_maglim_dict()
        maglim_dict["ROMAN_obs_Z087"] = 27.7
        maglim_dict["ROMAN_obs_Y106"] = 27.7
        maglim_dict["ROMAN_obs_J129"] = 27.6
        maglim_dict["ROMAN_obs_W146"] = 28.3
        maglim_dict["ROMAN_obs_H158"] = 27.5
        maglim_dict["ROMAN_obs_F184"] = 27.0
        maglim_dict["ROMAN_obs_K213"] = 25.9
        return maglim_dict

    @classmethod
    def _build_a_env_dict(cls) -> dict[str, float]:
        a_env_dict = super()._build_a_env_dict()
        a_env_dict["ROMAN_obs_Z087"] = 1.57491325
        a_env_dict["ROMAN_obs_Y106"] = 1.14025753
        a_env_dict["ROMAN_obs_J129"] = 0.83118224
        a_env_dict["ROMAN_obs_W146"] = 0.68098202
        a_env_dict["ROMAN_obs_H158"] = 0.59966235
        a_env_dict["ROMAN_obs_F184"] = 0.46923204
        a_env_dict["ROMAN_obs_K213"] = 0.37072579
        return a_env_dict

    @classmethod
    def _build_band_names(cls) -> list[str]:
        bands = [cls.band_template.format(band=band) for band in cls.bandlist]
        bands += [
            "ROMAN_obs_Z087",
            "ROMAN_obs_Y106",
            "ROMAN_obs_J129",
            "ROMAN_obs_W146",
            "ROMAN_obs_H158",
            "ROMAN_obs_F184",
            "ROMAN_obs_K213",
        ]
        return bands

    @classmethod
    def _build_band_err_names(cls) -> list[str]:
        band_errs = [cls.band_err_template.format(band=band) for band in cls.bandlist]
        band_errs += [
            "ROMAN_obs_Z087_err",
            "ROMAN_obs_Y106_err",
            "ROMAN_obs_J129_err",
            "ROMAN_obs_W146_err",
            "ROMAN_obs_H158_err",
            "ROMAN_obs_F184_err",
            "ROMAN_obs_K213_err",
        ]
        return band_errs

    @classmethod
    def _build_err_dict(cls) -> dict[str, str | None]:
        the_dict = super()._build_err_dict()
        the_dict["ROMAN_obs_Z087"] = "ROMAN_obs_Z087_err"
        the_dict["ROMAN_obs_Y106"] = "ROMAN_obs_Y106_err"
        the_dict["ROMAN_obs_J129"] = "ROMAN_obs_J129_err"
        the_dict["ROMAN_obs_W146"] = "ROMAN_obs_W146_err"
        the_dict["ROMAN_obs_H158"] = "ROMAN_obs_H158_err"
        the_dict["ROMAN_obs_F184"] = "ROMAN_obs_F184_err"
        the_dict["ROMAN_obs_K213"] = "ROMAN_obs_K213_err"
        return the_dict

    @classmethod
    def _build_ref_band(cls, ref_band: str = "i") -> str:
        return cls.band_template.format(band=ref_band)

    @classmethod
    def _build_filter_file_bandlist(cls) -> list[str]:
        """Contruct the name of the reference band"""
        filter_list = [
            cls.filter_file_template.format(band=band) for band in cls.bandlist
        ]
        filter_list += [
            "roman_Z087",
            "roman_Y106",
            "roman_J129",
            "roman_W146",
            "roman_H158",
            "roman_F184",
            "roman_K213",
        ]
        return filter_list



DP1_FILTERS = 'ugrizy'
DP1_FLUX_TYPES = ['cModel', 'kron', 'sersic', 'gaap1p0', 'psf', 'gaap3p0']
DP1_A_ENV_IN = Dc2CatalogConfig.build_base_dict()['band_a_env']
DP1_MAGLIMS_IN = Dc2CatalogConfig.build_base_dict()['mag_limits']
DP1_A_ENV_LIST = []
DP1_REPLACE_ERR_VALS = []
DP1_MAGLIMS = []
DP1_ZP_ERRORS = []
DP1_BANDS = []
for filter_ in DP1_FILTERS:
    for flux_type_ in DP1_FLUX_TYPES:
        band = f"{filter_}_{flux_type_}"
        DP1_BANDS.append(band)
        DP1_A_ENV_LIST.append(DP1_A_ENV_IN[f"mag_{filter_}_cModel_obj_dered"])
        DP1_MAGLIMS.append(DP1_MAGLIMS_IN[f"mag_{filter_}_cModel_obj_dered"])
        DP1_REPLACE_ERR_VALS.append(0.1)
        DP1_ZP_ERRORS.append(0.1)


class DP1AllFluxesCatalogConfig(CatalogConfigBase):
    """Configuration for ComCam data"""

    tag = "dp1_all"
    bandlist = DP1_BANDS
    maglims = DP1_MAGLIMS
    a_env = DP1_A_ENV_LIST
    band_template = "{band}Mag"
    band_err_template = "{band}MagErr"
    filter_file_template = "DC2LSST_{band}"
    ref_band = "i_psf"
    redshift_col = "redshift"
    object_id_col = "objectId"
    hdf5_groupname = ""
    replace_error_vals = DP1_REPLACE_ERR_VALS
    zp_errors = DP1_ZP_ERRORS




apply_defaults = CatalogConfigBase.apply
