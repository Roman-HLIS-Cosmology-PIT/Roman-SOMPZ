import rail.stages
from rail.core import RailStage
from rail.utils.testing_utils import check_stage_params

rail.stages.import_and_attach_all()


def test_all_stage_params() -> None:
    fail_list = []

    for _key, val in RailStage.pipeline_stages.items():
        fail_string = check_stage_params(val[0])
        if fail_string:
            fail_list.append(fail_string)

    if fail_list:
        for fail_ in fail_list:
            print(fail_)
        raise TypeError(f"Found {len(fail_list)} unparseable parameters")
