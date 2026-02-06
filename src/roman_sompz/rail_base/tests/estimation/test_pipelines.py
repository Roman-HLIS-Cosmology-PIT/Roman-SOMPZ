import pytest

from rail.utils.testing_utils import build_and_read_pipeline


@pytest.mark.parametrize(
    "pipeline_class",
    [
        "rail.pipelines.estimation.train_z_pipeline.TrainZPipeline",
    ],
)
def test_build_and_read_pipeline(pipeline_class: str) -> None:
    build_and_read_pipeline(pipeline_class)
