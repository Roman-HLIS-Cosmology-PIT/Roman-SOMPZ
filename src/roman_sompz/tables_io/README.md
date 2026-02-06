![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/LSSTDESC/tables_io)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LSSTDESC/tables_io/smoke-test.yml)
![Read the Docs](https://img.shields.io/readthedocs/tables-io)

[![Template](https://img.shields.io/badge/Template-RAIL%20Specific%20Fork%20LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://github.com/LSSTDESC/RAIL-project-template/tree/main)

# tables_io

`tables_io` provides an interface for a variety of non-ASCII file formats that are commonly used within the [LSST DESC collaboration](https://lsstdesc.org/). It allows users to read in data from multiple types of files through one convenient interface.

See [read the docs](https://tables-io.readthedocs.io/en/latest/index.html) for documentation.

## Features:

- reads and writes files that contain one or more data tables
- supports a variety of file types (`fits`, `hdf5`, `parquet`) and tabular formats (`astropy`, `pandas`, `pyarrow`, `numpy`)
- allows easy conversions between file formats and in memory tabular formats
- ability to do chunked reads and writes of `HDF5` and `parquet` files

`tables_io` is currently being used in the following packages:

- [`qp`](https://github.com/LSSTDESC/qp)
- [`RAIL`](https://github.com/LSSTDESC/rail)

## People

- [Eric Charles](https://github.com/LSSTDESC/qp/issues/new?body=@eacharles) (SLAC)

## License, Contributing etc

The code in this repo is available for re-use under the MIT license.
