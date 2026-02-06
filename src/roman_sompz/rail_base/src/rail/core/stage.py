""" Base class for PipelineStages in Rail """

from __future__ import annotations

import os
from math import ceil
from typing import Any, Iterable, TypeVar
import yaml

from ceci.config import StageParameter as Param
from ceci.pipeline import MiniPipeline
from ceci.stage import PipelineStage

from .data import DATA_STORE, DataHandle, DataLike, ModelHandle

T = TypeVar("T", bound="RailPipeline")
S = TypeVar("S", bound="RailStage")


class StageIO:
    """A small utility class for Stage Input/ Output

    This make it possible to get access to stage inputs and outputs
    as attributes rather that by using the get_handle() method.

    In short it maps

    a_stage.get_handle('input', allow_missing=True) to a_stage.input

    This allows users to be more concise when writing pipelines.
    """

    def __init__(self, parent: RailStage):
        self._parent = parent

    def __getattr__(self, item: str) -> DataHandle:
        return self._parent.get_handle(item, allow_missing=True)


class RailStageBuild:
    """A small utility class that building stages

    This provides a mechasim to get the name of the stage from the
    attribute name in the Pipeline the stage belongs to.

    I.e., we can do:

    a_pipe.stage_name = StageClass.build(...)

    And get a stage named 'stage_name', rather than having to do:

    a_stage = StageClass.make_stage(..)
    a_pipe.add_stage(a_stage)
    """

    def __init__(self, stage_class: type[RailStage], **kwargs: Any):
        self.stage_class: type[PipelineStage] = stage_class
        self._kwargs: dict = kwargs
        self._stage: RailStage | None = None

    @property
    def io(self) -> StageIO | None:  # pragma: no cover
        if self._stage:
            return self._stage.io
        return None

    def build(self, name: str) -> RailStage:
        """Actually build the stage, this is called by the pipeline the stage
        belongs to

        Parameters
        ----------
        name
            The name for this stage we are building

        Returns
        -------
        RailStage
            The newly built stage
        """
        self._stage = self.stage_class.make_and_connect(name=name, **self._kwargs)
        assert self._stage is not None
        return self._stage


class RailPipeline(MiniPipeline):
    """A pipeline intended for interactive use

    Mainly this allows for more concise pipeline specification, along the lines of:

    self.stage_1 = Stage1Class.build(...)
    self.stage_2 = Stage2Class.build(connections=dict(input=self.stage1.io.output), ...)

    And end up with a fully specified pipeline.
    """

    pipeline_classes: dict[str, type[RailPipeline]] = {}

    def __init_subclass__(cls) -> None:
        cls.pipeline_classes[cls.__name__] = cls

    @classmethod
    def print_classes(cls) -> None:
        for key, val in cls.pipeline_classes.items():
            print(f"{key} {val}")

    @classmethod
    def get_pipeline_class(cls, name: str) -> type[RailPipeline]:
        try:
            return cls.pipeline_classes[name]
        except KeyError as msg:
            raise KeyError(
                f"Could not find pipeline class {name} in {list(cls.pipeline_classes.keys())}"
            ) from msg

    @staticmethod
    def load_pipeline_class(class_name: str) -> type[RailPipeline]:
        """Import a particular RailPipeline subclass by name

        Parameters
        ----------
        class_name
            Full name of the class, e.g., rail.core.stage.RailPipeline

        Returns
        -------
        type[RailPipeline]
            Requested Pipeline sub-class
        """
        tokens = class_name.split(".")
        module = ".".join(tokens[:-1])
        class_name = tokens[-1]
        __import__(module)
        pipe_class = RailPipeline.get_pipeline_class(class_name)
        return pipe_class

    @staticmethod
    def build_and_write(
        class_name: str,
        output_yaml: str,
        input_dict: dict | None = None,
        stages_config: dict | None = None,
        output_dir: str = ".",
        log_dir: str = ".",
        **kwargs: Any,
    ) -> None:
        """Build a RailPipeline and write the config yaml for for it

        Parameters
        ----------
        class_name
            Full name of the class, e.g., rail.core.stage.RailPipeline

        output_yaml
            Path to the output yaml file

        input_dict
            Dict of all the inputs needed to run the pipeline

        stages_config
            Stage configuration overrides

        output_dir
            Directory to write pipeline outputs to

        log_dir
            Directory to write pipeline log files to

        **kwargs
            Passed as arguements to the pipeline constructor
        """
        pipe_class = RailPipeline.get_pipeline_class(class_name)
        pipe = pipe_class(**kwargs)

        full_input_dict = pipe_class.default_input_dict.copy()
        if input_dict is not None:
            full_input_dict.update(**input_dict)
        pipe.initialize(
            full_input_dict,
            dict(
                output_dir=output_dir,
                log_dir=log_dir,
                resume=False,
            ),
            stages_config,
        )

        # make sure stage_config_dict is a dict of stages_config
        if isinstance(stages_config, str): # pragma: no cover
            with open(stages_config, "r") as f:
                stages_config_dict = yaml.safe_load(f) or {}
        else:
            stages_config_dict = stages_config or {}
        
        # loop through stages
        for key, exec_cfg in pipe.stage_execution_config.items():
            stage_cfg = (stages_config_dict.get(key) or {})
            nprocess = stage_cfg.get("nprocess")
        
            if nprocess is None:
                continue
            else: # pragma: no cover
        
                exec_cfg.nprocess = nprocess
            
                # bump max_threads if needed
                site_cfg = exec_cfg.site.config
                if nprocess > site_cfg.get("max_threads", 0):
                    print(f"{class_name} update max_threads to {nprocess}")
                    site_cfg["max_threads"] = nprocess
        pipe.save(output_yaml)

    def __init__(self) -> None:
        MiniPipeline.__init__(self, [], dict(name="mini"))

    def __setattr__(
        self, name: str, value: RailStageBuild | Any
    ) -> PipelineStage | Any:
        if isinstance(value, RailStageBuild):
            stage = value.build(name)
            self.add_stage(stage)
            return stage
        return MiniPipeline.__setattr__(self, name, value)


class RailStage(PipelineStage):
    """Base class for rail stages

    This inherits from `ceci.PipelineStage` and implements rail-specific data handling
    In particular, this provides some very useful features:

    1.  Access to the `DataStore`, which keeps track of the various data used in a pipeline, and
    provides access to each by a unique key.

    2.  Functionality to help manage multiple instances of a particular class of stage.
    The original ceci design didn't have a mechanism to handle this.  If you tried
    you would run into name clashes between the different instances.  In `ceci` 1.7 we
    added functionality to `ceci` to allow you to have multiple instances of a single class,
    in particular we distinguish between the class name (`cls.name`) and and the name of
    the particular instance (`self.instance_name`) and added aliasing for inputs and outputs,
    so that different instances of `PipelineStage` would be able to give different names
    to their inputs and outputs.  However, using that functionality in a consistent way
    requires a bit of care.  So here we are providing methods to do that, and to do it in
    a way that uses the `DataStore` to keep track of the various data products.

    Notes
    -----
    These methods typically take a tag as input (i.e., something like "input"),
    but use the "aliased_tag" (i.e., something like "inform_pz_input") when interacting
    with the DataStore.

    In particular, the `get_handle()`, `get_data()` and `input_iterator()` will get the data
    from the DataStore under the aliased tag.  E.g., if you call `self.get_data('input')` for
    a `Stage` that has aliased "input" to "special_pz_input", it will
    get the data associated to "special_pz_input" in the DataStore.

    Similarly, `add_handle()` and `set_data()` will add the data to the DataStore under the aliased tag
    e.g., if you call `self.set_data('input')` for a `Stage` that has
    aliased "input" to "special_pz_input", it will store the data in the DataStore
    under the key "special_pz_input".

    And `connect_input()` will do the alias lookup both on the input and output.
    I.e., it is the same as calling
    `self.set_data(inputTag, other.get_handle(outputTag, allow_missing=True), do_read=False)`
    """

    config_options = dict(
        output_mode=Param(str, "default", msg="What to do with the outputs")
    )

    data_store = DATA_STORE()

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Constructor:
        Do RailStage specific initialization"""
        super().__init__(args, **kwargs)
        self._input_length: int | None = None
        self.io = StageIO(self)
        self.stage_columns: list[str] | None = None

    @classmethod
    def make_and_connect(cls: type[S], **kwargs: Any) -> S:
        """Make a stage and connects it to other stages

        Notes
        -----
        kwargs are used to set stage configuration,
        the should be key, value pairs, where the key
        is the parameter name and the value is value we want to assign

        The 'connections' keyword is special, it is a dict[str, DataHandle]
        and should define the Input connections for this stage

        Returns
        -------
        A stage
        """
        connections = kwargs.pop("connections", {})
        stage = cls.make_stage(**kwargs)
        for key, val in connections.items():
            stage.set_data(key, val, do_read=False)
        return stage

    @classmethod
    def build(cls, **kwargs: Any) -> RailStageBuild:
        """Return an object that can be used to build a stage"""
        return RailStageBuild(cls, **kwargs)

    def get_handle(
        self, tag: str, path: str | None = None, allow_missing: bool = False
    ) -> DataHandle:
        """Gets a DataHandle associated to a particular tag

        Parameters
        ----------
        tag
            The tag (from cls.inputs or cls.outputs) for this data

        path
            The path to the data, only needed if we might need to read the data

        allow_missing
            If False this will raise a key error if the tag is not in the DataStore

        Returns
        -------
        DataHandle
            The handle that give access to the associated data
        """
        aliased_tag = self.get_aliased_tag(tag)
        handle = self.data_store.get(aliased_tag)
        if handle is None:
            if not allow_missing:
                raise KeyError(
                    f"{self.instance_name} failed to get data by handle {aliased_tag}, associated to {tag}"
                )
            handle = self.add_handle(tag, path=path)
        return handle

    def add_handle(
        self, tag: str, data: DataLike = None, path: str | None = None
    ) -> DataHandle:
        """Adds a DataHandle associated to a particular tag

        Parameters
        ----------
        tag
            The tag (from cls.inputs or cls.outputs) for this data

        data
            If not None these data will be associated to the handle

        path
            If not None, this will be the path used to read the data

        Returns
        -------
        DataHandle
            The handle that gives access to the associated data
        """
        aliased_tag = self.get_aliased_tag(tag)
        if aliased_tag in self._inputs:
            if path is None:
                path = self.get_input(aliased_tag)
            handle_type = self.get_input_type(tag)
        else:
            if path is None:
                path = self.get_output(aliased_tag)
            handle_type = self.get_output_type(tag)
        handle = handle_type(
            aliased_tag, path=path, data=data, creator=self.instance_name
        )
        print(
            f"Inserting handle into data store.  {aliased_tag}: {handle.path}, {handle.creator}"
        )
        self.data_store[aliased_tag] = handle
        return handle

    def get_data(self, tag: str, allow_missing: bool = True) -> DataLike:
        """Gets the data associated to a particular tag

        Notes
        -----
        1. This gets the data via the DataHandle, and can and will read the data
        from disk if needed.

        Parameters
        ----------
        tag
            The tag (from cls.inputs or cls.outputs) for this data

        allow_missing
            If False this will raise a key error if the tag is not in the DataStore

        Returns
        -------
        DataLike
            The data accesed by the handle assocated to the tag
        """
        handle = self.get_handle(tag, allow_missing=allow_missing)
        if not handle.has_data:
            handle.read()
        return handle()

    def set_data(
        self, tag: str, data: DataLike, path: str | None = None, do_read: bool = True
    ) -> DataLike:
        """Sets the data associated to a particular tag

        Notes
        -----
        1. If data is a DataHandle and tag is one of the input tags,
        then this will add an alias between the two, i.e., it will
        set `self.config.alias[tag] = data.tag`.  This allows the user to
        make connections between stages simply by passing DataHandles between
        them.

        Parameters
        ----------
        tag
            The tag (from cls.inputs or cls.outputs) for this data

        data
            The data being set,

        path
            Can be used to set the path for the data

        do_read
            If True, will read the data if it is not set

        Returns
        -------
        DataLike
            The data accesed by the handle assocated to the tag
        """
        # First we grab the handle that we will be using
        handle = self.get_handle(tag, path=path, allow_missing=True)

        if isinstance(data, DataHandle):
            # If we were passed a DataHandle, we use that
            aliased_tag = data.tag
            if tag in self.input_tags():
                self._aliases[tag] = aliased_tag
                if data.has_path:
                    self._inputs[tag] = data.path
                    handle.path = data.path
                if data.has_data:
                    handle.data = data.data
                elif do_read:
                    handle.read()
        elif not isinstance(data, (type(None), str)):
            # If we were passed data, we use that and reset the path
            handle.data = data
            if path in ['None', 'none', None]:
                handle.path = 'None'
        else:
            # Data is None, we use the path
            if path not in ['None', 'none', None]:
                # Path exists, use that                
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Unable to find file: {path}")
                handle.path = path
                if do_read:
                    # Read the data using the handle
                    handle.read()

        return handle.data

    def add_data(self, tag: str, data: DataLike = None) -> DataLike:
        """Adds a handle to the DataStore associated to a particular tag and
        attaches data to it.

        Parameters
        ----------
        tag
            The tag (from cls.inputs or cls.outputs) for this data

        data
            Data being added

        Returns
        -------
        DataLike
            The data accesed by the handle assocated to the tag
        """
        handle = self.add_handle(tag, data=data)
        return handle.data

    def input_iterator(self, tag: str, **kwargs: Any) -> Iterable:
        """Iterate the input assocated to a particular tag

        Parameters
        ----------
        tag
            The tag (from cls.inputs or cls.outputs) for this data

        **kwargs
            These will be passed to the Handle's iterator method

        Returns
        -------
        Fixme

        """
        handle = self.get_handle(tag, allow_missing=True)

        try:
            groupname = kwargs.get("groupname", self.config.hdf5_groupname)
        except Exception:
            groupname = None
        if groupname == "None":
            groupname = None

        chunk_size = kwargs.get("chunk_size", self.config.chunk_size)

        on_disk: bool= False
        in_memory: bool= False
        if handle.path not in [None, 'None', 'none']:
            on_disk = True
        if handle.data is not None:
            in_memory = True
        print("groupname", groupname is None, groupname) 
        if on_disk:
            if groupname is None:
                self._input_length = handle.size()
            else:
                self._input_length = handle.size(groupname=groupname)

            total_chunks_needed = ceil(self._input_length / chunk_size)
            # If the number of process is larger than we need, we reduce chunk_size
            # so that all of the processes have some data to work with.
            if total_chunks_needed < self.size:  # pragma: no cover
                self.config.chunk_size = int(ceil(self._input_length / self.size))
                chunk_size = self.config.chunk_size
                print(
                    "Warning: You are reserving more processes than needed, reducing chunk size to",
                    chunk_size,
                    "to use all of the processes",
                )
            if groupname is None:
                kwcopy = dict(
                    chunk_size=chunk_size,
                    rank=self.rank,
                    parallel_size=self.size,
                )
            else: 
                kwcopy = dict(
                    groupname=groupname,
                    chunk_size=chunk_size,
                    rank=self.rank,
                    parallel_size=self.size,
                )
            kwcopy.update(**kwargs)
            return handle.iterator(**kwcopy)

        # If data is in memory and not in a file, it means is small enough to process it
        # in a single chunk.
        elif in_memory:  # pragma: no cover
            if self.config.hdf5_groupname:
                test_data = self.get_data(tag)[self.config.hdf5_groupname]
                self._input_length = self.get_handle(tag).data_size(
                    groupname=self.config.hdf5_groupname
                )
            else:
                test_data = self.get_data(tag)
                self._input_length = self.get_handle(tag).data_size()
            s = 0
            iterator = [[s, self._input_length, test_data]]
            return iterator

        # Data is neither on disk or in memory, return empty list
        else:
            return []
        
    def connect_input(
        self,
        other: PipelineStage,
        inputTag: str | None = None,
        outputTag: str | None = None,
    ) -> DataHandle:
        """Connect another stage to this stage as an input

        Parameters
        ----------
        other
             The stage whose output is being connected

        inputTag
             Which input tag of this stage to connect to.  None -> self.inputs[0]

        outputTag
             Which output tag of the other stage to connect to.  None -> other.outputs[0]

        Returns
        -------
        DataHandle
            The input handle for this stage
        """
        if inputTag is None:
            inputTag = self.inputs[0][0]  # pylint: disable=no-member
        if outputTag is None:
            outputTag = other.outputs[0][0]
        handle = other.get_handle(outputTag, allow_missing=True)
        return self.set_data(inputTag, handle, do_read=False)

    def _finalize_tag(self, tag: str) -> str:
        """Finalize the data for a particular tag.

        This can be overridden by sub-classes for more complicated behavior
        """
        handle = self.get_handle(tag, allow_missing=True)
        if self.config.output_mode == "default":
            assert handle.path is not None
            if not os.path.exists(handle.path) or not handle.partial:
                handle.write()
        final_name = PipelineStage._finalize_tag(self, tag)
        handle.path = final_name
        return final_name

    def _check_column_names(
        self, data: Any, columns_to_check: list[str], **kwargs: Any
    ) -> None:
        try:
            groupname = kwargs.get("groupname", self.config.hdf5_groupname)
        except Exception:  # pragma: no cover
            groupname = None

        if not groupname:
            groupname = None

        if isinstance(data, DataHandle) and not data.has_data:
            if data.has_path:
                # data handle only has a path, read the columns from the path
                path = data.path
                assert path is not None
                data._check_data_columns(
                    path, columns_to_check, parent_groupname=groupname, **kwargs
                )
            elif not data.has_path:  # pragma: no cover
                print("The data handle does not contain data or path.")

        else:
            # data has been read in, access the columns in the table/dictionary directly
            if isinstance(data, DataHandle) and data.has_data:
                if groupname in [None, ""]:
                    col_list = list(data.data.keys())
                else:
                    col_list = list(data.data[groupname].keys())
            else:
                # data is passed as a table
                if groupname in [None, ""]:
                    col_list = list(data.keys())
                else:
                    col_list = list(data[groupname].keys())
            # check columns
            intersection = set(columns_to_check).intersection(col_list)
            if len(intersection) < len(columns_to_check):
                diff = set(columns_to_check) - intersection
                raise KeyError("The following columns are not found: ", diff)

    def _get_stage_columns(self) -> None:
        self.stage_columns = None  # pragma: no cover

    def open_model(self, tag: str = "model", **kwargs: Any) -> ModelLike:
        """Load the mode and/or attach it to this Stage

        Parameters
        ----------
        tag
            Input tag associated to the model

        **kwargs
            Should include 'model', see notes

        Notes
        -----
        The keyword arguement 'model' should be either

        1. an object with a trained model,
        2. a path pointing to a file that can be read to obtain the trained model,
        3. or a `ModelHandle` providing access to the trained model.

        Returns
        -------
        Any
            The object encapsulating the trained model.
        """
        if tag not in self.input_tags():  # pragma: no cover
            raise KeyError(f"Stage {self} can not open model with input tag {tag}")
        model = kwargs.get(tag, None)
        if model is None or model == "None":  # pragma: no cover
            self.model = None
            return self.model
        if isinstance(model, str):
            self.model = self.set_data(tag, data=None, path=model)
            self.config[tag] = model
            return self.model
        if isinstance(model, ModelHandle):
            if model.has_path:
                self.config[tag] = model.path
        self.model = self.set_data(tag, model)
        return self.model
