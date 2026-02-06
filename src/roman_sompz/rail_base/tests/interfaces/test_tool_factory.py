import os

import pytest

from rail.interfaces import ToolFactory
from rail.utils.path_utils import find_rail_file


def test_tool_factory() -> None:
    stage = ToolFactory.build_tool_stage(
        "pq_to_hdf",
        "TableConverter",
        "rail.tools.table_tools",
        "dummy.in",
        output_format="numpyDict",
    )

    input_file = find_rail_file("examples_data/testdata/rubin_dm_dc2_example.pq")

    _out_handle = ToolFactory.run_tool_stage(
        stage,
        input_file,
    )

    check_stage = ToolFactory.get_tool_stage("pq_to_hdf")
    assert check_stage == stage

    with pytest.raises(KeyError):
        ToolFactory.get_tool_stage("nope")

    ToolFactory.reset()
    assert not ToolFactory._stage_dict

    try:
        os.unlink("inprogress_output_pq_to_hdf.hdf5")
    except FileNotFoundError:
        pass
    try:
        os.unlink("output_pq_to_hdf.hdf5")
    except FileNotFoundError:
        pass
