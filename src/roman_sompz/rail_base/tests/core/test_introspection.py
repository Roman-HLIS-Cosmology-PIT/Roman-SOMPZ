import tempfile

import rail
import rail.stages
from rail.core import RailEnv
from rail.core.data import DataHandle, TableHandle


def test_print_rail_packages() -> None:
    RailEnv.print_rail_packages()


def test_print_rail_namespaces() -> None:
    RailEnv.print_rail_namespaces()


def test_print_rail_modules() -> None:
    RailEnv.print_rail_modules()


def test_print_rail_namespace_tree() -> None:
    RailEnv.print_rail_namespace_tree()


def test_import_and_attach_all() -> None:
    rail.stages.import_and_attach_all()
    RailEnv.print_rail_stage_dict()


def test_api_rst() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        RailEnv.do_api_rst(tmpdirname)
        RailEnv.do_stage_type_api_rst(tmpdirname)


def test_data_handle_dict() -> None:
    DataHandle.print_sub_classes()
    assert DataHandle.get_sub_class("TableHandle") == TableHandle
