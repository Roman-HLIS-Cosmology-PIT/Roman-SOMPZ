"""cli for tables_io.convert"""

from functools import partial
from typing import Any

import click

from . import types
import tables_io


class PartialOption:
    """Wraps click.option with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.option, *param_decls, cls=partial(click.Option), **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._partial(*args, **kwargs)


class PartialArgument:
    """Wraps click.argument with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.argument, *param_decls, cls=click.Argument, **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return self._partial(*args, **kwargs)


inputs = PartialArgument("inputs", nargs=-1)


input = PartialOption(
    "--input",
    type=click.Path(),
    help=f"input filename; suffix should be one of {list(types.FILE_FORMAT_SUFFIXS.keys())}",
)

output = PartialOption(
    "--output",
    type=click.Path(),
    help=f"output filename; suffix should be one of {list(types.FILE_FORMAT_SUFFIXS.keys())}",
)


@click.group()
@click.version_option(tables_io._version)
def cli() -> None:
    """tables_io utility scripts"""


@cli.command()
@input()
@output()
def convert(input, output):
    """Convert a file with tabular data from one format to another"""

    input_fname = input
    output_fname = output
    output_format = output_fname.split(".")[-1]

    print(f"Converting {input_fname} to {output_fname}")

    # This is the enum of the output format type, based on the suffix
    suffix = types.FILE_FORMAT_SUFFIXS[output_format]

    # This is the enum of the corresponding table type
    table_format = types.TABLE_FORMAT[suffix]

    t_in = tables_io.read(input_fname)
    t_out = tables_io.convert(t_in, table_format)
    _written = tables_io.write(t_out, output_fname)

    print("Done converting file")

    return 0


@cli.command(name='concatanate')
@inputs()
@output()
def concatanate(inputs, output):
    """Concatanate a list of tables"""

    input_fnames = inputs
    output_fname = output
    output_format = output_fname.split(".")[-1]

    print(f"Concatanating {input_fnames} to {output_fname}")

    # This is the enum of the output format type, based on the suffix
    suffix = types.FILE_FORMAT_SUFFIXS[output_format]

    # This is the enum of the corresponding table type
    t_format = types.TABLE_FORMAT[suffix]

    table_list = []
    for input_fname in input_fnames:
        t_in = tables_io.read(input_fname)
        table_list.append(t_in)

    t_out = tables_io.concat_table(table_list, t_format)

    _written = tables_io.write(t_out, output_fname)

    print("Done Concatanating file")

    return 0
