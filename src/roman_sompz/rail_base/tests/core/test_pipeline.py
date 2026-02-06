import os

import ceci
import numpy as np
import pytest

from rail.core.stage import RailPipeline, RailStage
from rail.tools.table_tools import ColumnMapper, TableConverter
from rail.utils.path_utils import RAILDIR


def test_pipeline() -> None:
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()

    input_file = os.path.join(
        RAILDIR, "rail/examples_data/goldenspike_data/data//test_flow_data.pq"
    )
    bands = ["u", "g", "r", "i", "z", "y"]
    _band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"mag_{band}_lsst": f"{band}_lsst" for band in bands}
    _post_grid = [float(x) for x in np.linspace(0.0, 5, 21)]

    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )

    table_conv_test = TableConverter.make_stage(
        name="table_conv_test", output_format="numpyDict", seed=12345
    )

    pipe = ceci.Pipeline.interactive()
    stages = [
        col_remapper_test,
        table_conv_test,
    ]
    for stage in stages:
        pipe.add_stage(stage)

    table_conv_test.connect_input(col_remapper_test)

    pipe.initialize(
        dict(input=input_file), dict(output_dir=".", log_dir=".", resume=False), None
    )

    pipe.save("stage.yaml")

    pr = ceci.Pipeline.read("stage.yaml")
    pr.run()

    os.remove("stage.yaml")
    os.remove("stage_config.yml")

    outputs = pr.find_all_outputs()
    for output_ in outputs.values():
        try:
            os.remove(output_)
        except FileNotFoundError:
            pass
    logfiles = [f"{stage.instance_name}.out" for stage in pr.stages]
    for logfile_ in logfiles:
        try:
            os.remove(logfile_)
        except FileNotFoundError:
            pass


def test_golden_v2() -> None:
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()
    pipe = RailPipeline()

    input_file = os.path.join(
        RAILDIR, "rail/examples_data/goldenspike_data/data//test_flow_data.pq"
    )
    bands = ["u", "g", "r", "i", "z", "y"]
    _band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"mag_{band}_lsst": f"{band}_lsst" for band in bands}
    _post_grid = [float(x) for x in np.linspace(0.0, 5, 21)]

    pipe.col_remapper_test = ColumnMapper.build(
        hdf5_groupname="",
        columns=rename_dict,
    )

    pipe.table_conv_test = TableConverter.build(
        connections=dict(
            input=pipe.col_remapper_test.io.output  # pylint: disable=no-member
        ),
        output_format="numpyDict",
        seed=12345,
    )

    pipe.initialize(
        dict(input=input_file), dict(output_dir=".", log_dir=".", resume=False), None
    )
    pipe.save("stage.yaml")

    pr = ceci.Pipeline.read("stage.yaml")
    pr.run()

    os.remove("stage.yaml")
    os.remove("stage_config.yml")

    outputs = pr.find_all_outputs()
    for output_ in outputs.values():
        try:
            os.remove(output_)
        except FileNotFoundError:
            pass
    logfiles = [f"{stage.instance_name}.out" for stage in pr.stages]
    for logfile_ in logfiles:
        try:
            os.remove(logfile_)
        except FileNotFoundError:
            pass


def test_load_pipeline() -> None:
    train_z_class = RailPipeline.load_pipeline_class(
        "rail.pipelines.estimation.train_z_pipeline.TrainZPipeline"
    )

    check = RailPipeline.get_pipeline_class("TrainZPipeline")
    assert check == train_z_class

    RailPipeline.print_classes()

    with pytest.raises(KeyError):
        RailPipeline.get_pipeline_class("Does not exist")
