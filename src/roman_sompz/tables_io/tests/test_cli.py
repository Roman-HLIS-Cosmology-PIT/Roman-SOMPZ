"""
Unit tests for cli tools
"""

from click.testing import CliRunner
from tables_io.cli import convert, concatanate


def test_cli_convert(test_dir, tmp_path):
    """Testing the main Convert CLI"""

    input_file = test_dir / "data/pandas_test_hdf5.h5"
    output_file = tmp_path / "test_convert.fits"

    runner = CliRunner()
    result = runner.invoke(
        convert, ["--input", str(input_file), "--output", str(output_file)]
    )

    assert result.exit_code == 0
    assert output_file.exists()



def test_cli_concat(test_dir, tmp_path):
    """Testing the main Convert CLI"""

    input_file = test_dir / "data/pandas_test_hdf5.h5"
    output_file = tmp_path / "test_convert.hdf5"

    runner = CliRunner()
    result = runner.invoke(
        concatanate, ["--output", str(output_file), str(input_file), str(input_file), str(input_file)]
    )

    assert result.exit_code == 0
    assert output_file.exists()
