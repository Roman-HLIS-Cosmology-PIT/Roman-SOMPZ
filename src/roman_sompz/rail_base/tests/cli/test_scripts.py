import glob
import os

import pytest

from rail.cli.rail import scripts
from rail.cli.rail.options import GitMode, args_to_dict


def test_args_to_dict() -> None:
    args = ["x=3", "y=2"]
    as_dict = args_to_dict(args)
    assert as_dict["x"] == "3"
    assert as_dict["y"] == "2"

    with pytest.raises(ValueError):
        args_to_dict(["x"])
    with pytest.raises(ValueError):
        args_to_dict(["x=3=3"])


def test_render_nb() -> None:
    nb_dir = "./tests/cli/"
    nb_files = glob.glob(os.path.join(nb_dir, "*.ipynb"))
    scripts.render_nb("docs", False, True, nb_files, skip=[])
    scripts.render_nb(
        "docs", True, True, nb_files, skip=["./tests/cli/single_number.ipynb"]
    )
    scripts.render_nb(
        "docs", True, False, nb_files, skip=["./tests/cli/single_number.ipynb"]
    )


def test_clone_source() -> None:
    scripts.clone_source("..", GitMode.ssh, True, "rail_packages.yml")
    scripts.clone_source("..", GitMode.https, True, "rail_packages.yml")
    scripts.clone_source("..", GitMode.cli, True, "rail_packages.yml")


def test_update_source() -> None:
    scripts.update_source("..", True, "rail_packages.yml")


def test_install() -> None:
    scripts.install("..", False, True, "rail_packages.yml")
    scripts.install("..", True, True, "rail_packages.yml")


def test_info() -> None:
    scripts.info(print_all=True)

    
def test_git_stuff() -> None:
    scripts.git_status("..", True, "rail_packages.yml")
    scripts.git_diff("..", True, "rail_packages.yml")
    scripts.git_describe("..", True, "rail_packages.yml")

    
def test_build_pipeline() -> None:
    scripts.build_pipeline(
        "rail.pipelines.estimation.train_z_pipeline.TrainZPipeline",
        "trainz_pipe.yaml",
        input_dict=dict(input="dummy.ibn"),
    )

    os.unlink("trainz_pipe.yaml")
    os.unlink("trainz_pipe_config.yml")
