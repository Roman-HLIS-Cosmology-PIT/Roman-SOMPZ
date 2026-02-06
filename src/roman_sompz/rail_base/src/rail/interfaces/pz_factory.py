from ceci.stage import PipelineStage
from qp import Ensemble

from rail.core.data import DataHandle
from rail.core.stage import RailStage
from rail.estimation.estimator import CatEstimator


class PZFactory:
    """Factory class to provide a unified interface to
    rail p(z) estimation algorithms.
    """

    _stage_dict: dict[str, CatEstimator] = {}

    @classmethod
    def reset(cls) -> None:
        """Reset the dictionary of cached stage objects"""
        cls._stage_dict = {}

    @classmethod
    def build_stage_instance(
        cls,
        stage_name: str,
        stage_class: type[CatEstimator],
        model_path: str | dict,
        data_path: str = "none",
        **config_params: dict,
    ) -> CatEstimator:
        """Build and configure an estimator that can evalute
        p(z) given an input catalog

        Parameters
        ----------
        stage_name: str
            Name of the stage instance, used to construct output file name

        stage_class: type
            Python class for the stage

        model_path: str| dict
            Path to the model file used by this estimator, or the model itself

        data_path: str
            Path to the input data, defaults to 'none'

        config_params: dict
            Configuration parameters for the stage

        Returns
        -------
        stage_obj: CatEstimator
            Newly constructed and configured Estimator instance
        """
        stage_obj = stage_class.make_stage(
            name=stage_name, model=model_path, input=data_path, **config_params
        )
        cls._stage_dict[stage_name] = stage_obj
        return stage_obj

    @classmethod
    def build_cat_estimator_stage(
        cls,
        stage_name: str,
        class_name: str,
        module_name: str,
        model_path: str,
        data_path: str = "none",
        **config_params: dict,
    ) -> CatEstimator:
        """Build and configure an estimator that can evalute
        p(z) given an input catalog

        Parameters
        ----------
        stage_name: str
            Name of the stage instance, used to construct output file name

        class_name: str
            Name of the class, e.g., TrainZEstimator, used to find the class

        module_name: str
            Name of the python module that constains the class, used to import it

        model_path: str
            Path to the model file used by this estimator

        data_path: str
            Path to the input data, defaults to 'none'

        config_params: dict
            Configuration parameters for the stage

        Returns
        -------
        stage_obj: CatEstimator
            Newly constructed and configured Estimator instance
        """
        stage_class = PipelineStage.get_stage(class_name, module_name)
        return cls.build_stage_instance(
            stage_name, stage_class, model_path, data_path, **config_params
        )

    @classmethod
    def get_cat_estimator_stage(
        cls,
        stage_name: str,
    ) -> CatEstimator:
        """Return a cached p(z) estimator"""
        try:
            return cls._stage_dict[stage_name]
        except KeyError as msg:
            raise KeyError(
                f"Could not find stage named {stage_name}, did you build it?"
                f"Existing stages are: {list(cls._stage_dict.keys())}"
            ) from msg

    @staticmethod
    def run_cat_estimator_stage(
        stage_obj: CatEstimator,
        data_path: str,
    ) -> DataHandle:
        """Run a p(z) estimator on an input data file

        Parameters
        ----------
        stage_obj: CatEstimator
            Object that will do the estimation

        Returns
        -------
        data_handle: DataHandle
            Object that can give access to the data
        """
        RailStage.data_store.clear()
        handle = stage_obj.get_handle("input", path=data_path, allow_missing=True)
        return stage_obj.estimate(handle)

    @staticmethod
    def estimate_single_pz(
        stage_obj: CatEstimator,
        data_table: dict,
        input_size: int = 1,
    ) -> Ensemble:
        """Run a p(z) estimator on some objects

        Parameters
        ----------
        stage_obj: CatEstimator
            Object that will do the estimation

        data_table: dict
            Input data presented as dict of numpy arrays objects

        input_size: int
            Number of objects in the input table

        Returns
        -------
        pz : qp.Ensemble
            Output pz
        """
        RailStage.data_store.clear()
        if stage_obj.model is None:
            stage_obj.open_model(**stage_obj.config)
        stage_obj._input_length = input_size
        stage_obj._process_chunk(0, input_size, data_table, True)
        return stage_obj._output_handle.data
