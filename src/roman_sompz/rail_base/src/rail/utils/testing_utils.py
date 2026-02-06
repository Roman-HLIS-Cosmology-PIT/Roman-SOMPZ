"""Utility functions to test alogrithms"""

import os
from typing import Any

import ceci

from rail.cli.rail.scripts import build_pipeline
from rail.core.data import TableHandle
from rail.core.stage import RailStage
from rail.estimation.estimator import CatEstimator
from rail.estimation.informer import CatInformer
from rail.utils.path_utils import RAILDIR

traindata = os.path.join(RAILDIR, "rail/examples_data/testdata/training_100gal.hdf5")
validdata = os.path.join(RAILDIR, "rail/examples_data/testdata/validation_10gal.hdf5")
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def one_algo(
    key: str,
    single_trainer: type[CatInformer],
    single_estimator: type[CatEstimator],
    train_kwargs: dict,
    estim_kwargs: dict,
    is_classifier: bool = False,
) -> Any:
    """
    A basic test of running an estimator subclass.
    Run inform, write temporary trained model to
    'tempmodelfile.tmp', run photo-z algorithm.
    Then, load tempmodelfile.tmp and re-run, return
    both datasets.
    """
    DS.clear()
    training_data = DS.read_file("training_data", TableHandle, traindata)
    validation_data = DS.read_file("validation_data", TableHandle, validdata)

    if single_trainer is not None:
        train_pz = single_trainer.make_stage(**train_kwargs)
        train_pz.inform(training_data)

    pz = single_estimator.make_stage(name=key, **estim_kwargs)
    if not is_classifier:
        estim = pz.estimate(validation_data)
    elif is_classifier:  # pragma: no cover
        estim = pz.classify(validation_data)
    pz_2 = None
    estim_2 = estim
    pz_3 = None
    estim_3 = estim

    copy_estim_kwargs = estim_kwargs.copy()
    model_file = copy_estim_kwargs.pop("model", "None")

    if model_file != "None":
        copy_estim_kwargs["model"] = model_file
        pz_2 = single_estimator.make_stage(name=f"{pz.name}_copy", **copy_estim_kwargs)
        if not is_classifier:
            estim_2 = pz_2.estimate(validation_data)
        elif is_classifier:  # pragma: no cover
            estim_2 = pz_2.classify(validation_data)

    if single_trainer is not None and "model" in single_trainer.output_tags():
        copy3_estim_kwargs = estim_kwargs.copy()
        copy3_estim_kwargs["model"] = train_pz.get_handle("model")
        pz_3 = single_estimator.make_stage(
            name=f"{pz.name}_copy3", **copy3_estim_kwargs
        )
        if not is_classifier:
            estim_3 = pz_3.estimate(validation_data)
        elif is_classifier:  # pragma: no cover
            estim_3 = pz_3.classify(validation_data)

    os.remove(pz.get_output(pz.get_aliased_tag("output"), final_name=True))
    if pz_2 is not None:
        os.remove(pz_2.get_output(pz_2.get_aliased_tag("output"), final_name=True))

    if pz_3 is not None:
        os.remove(pz_3.get_output(pz_3.get_aliased_tag("output"), final_name=True))
    model_file = estim_kwargs.get("model", "None")
    if model_file != "None":
        try:
            os.remove(model_file)
        except FileNotFoundError:  # pragma: no cover
            pass
    return estim.data, estim_2.data, estim_3.data


def check_stage_params(stage_class: type[RailStage]) -> str | None:
    legal_types = (int, float, bool, list, dict, str, None)

    for key, val in stage_class.config_options.items():
        def_val_dtype = None
        if isinstance(val, ceci.config.StageConfig):
            continue
        if isinstance(val, ceci.config.StageParameter):
            dtype = val.dtype
            if val.default is not None:
                def_val_dtype = type(val.default)
        elif isinstance(val, type):
            dtype = val
        else:
            dtype = type(val)
        if dtype not in legal_types:  # pragma: no cover
            return f"Illegal parameter type for {stage_class.name}.{key} {dtype}"
        if def_val_dtype not in legal_types:  # pragma: no cover
            return f"Illegal parameter default value type for {stage_class.name}.{key} {def_val_dtype}"
    return None


def build_and_read_pipeline(pipeline_class: str, **kwargs: Any) -> None:
    short_name = pipeline_class.split(".")[-1]
    yaml_file = f"{short_name}.yml"
    config_yaml_file = f"{short_name}_config.yml"
    build_pipeline(pipeline_class, yaml_file, "rubin", **kwargs)
    _pr = ceci.Pipeline.read(yaml_file)
    os.unlink(yaml_file)
    os.unlink(config_yaml_file)
