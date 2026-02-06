from ceci.stage import PipelineStage

from rail.core.data import DataHandle
from rail.core.stage import RailStage


class ToolFactory:
    """Factory class to provide a unified interface to
    rail tool stages
    """

    _stage_dict: dict[str, RailStage] = {}

    @classmethod
    def reset(cls) -> None:
        """Reset the dictionary of cached stage objects"""
        cls._stage_dict = {}

    @classmethod
    def build_stage_instance(
        cls,
        stage_name: str,
        stage_class: type[RailStage],
        data_path: str = "none",
        **config_params: dict,
    ) -> RailStage:
        """Build and configure a rail tool stage

        Parameters
        ----------
        stage_name: str
            Name of the stage instance, used to construct output file name

        stage_class: type
            Python class for the stage

        data_path: str
            Path to the input data, defaults to 'none'

        config_params: dict
            Configuration parameters for the stage

        Returns
        -------
        stage_obj: RailStage
            Newly constructed and configured RailStage instance
        """
        stage_obj = stage_class.make_stage(
            name=stage_name, input=data_path, **config_params
        )
        cls._stage_dict[stage_name] = stage_obj
        return stage_obj

    @classmethod
    def build_tool_stage(
        cls,
        stage_name: str,
        class_name: str,
        module_name: str,
        data_path: str = "none",
        **config_params: dict,
    ) -> RailStage:
        """Build and configure a rail tool stage

        Parameters
        ----------
        stage_name: str
            Name of the stage instance, used to construct output file name

        class_name: str
            Name of the class, used to find the class

        module_name: str
            Name of the python module that constains the class, used to import it

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
            stage_name, stage_class, data_path, **config_params
        )

    @classmethod
    def get_tool_stage(
        cls,
        stage_name: str,
    ) -> RailStage:
        """Return a tool stage"""
        try:
            return cls._stage_dict[stage_name]
        except KeyError as msg:
            raise KeyError(
                f"Could not find stage named {stage_name}, did you build it?"
                f"Existing stages are: {list(cls._stage_dict.keys())}"
            ) from msg

    @staticmethod
    def run_tool_stage(
        stage_obj: RailStage,
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
        return stage_obj(handle)
