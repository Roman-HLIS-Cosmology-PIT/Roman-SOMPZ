# Developer Set Up

(developer-environment-setup)=

## Developer Environment Setup

For the installation of `tables_io` for the purpose of development, we recommend that you use a separate [Anaconda](https://docs.anaconda.com/anaconda/install/) virtual environment with `tables_io` installed in "editable" mode with all dev optional dependencies added.

In this guide, we will name this developer environment `tables_io_dev` and we will assume that an Anaconda with a minimum Python version of 3.9 has been previously installed.

To install the developer environment:

::::{tab-set}

:::{tab-item} bash

```bash
# Clone the repo and enter it
git clone https://github.com/LSSTDESC/tables_io.git
cd tables_io

# Creating the environment from the YAML
conda env create -n tables_io_dev -f environment.yml

# Activate the environment
conda activate tables_io_dev

# Install tables_io in editable mode with dev dependencies
pip install -e .[dev]
```

:::

:::{tab-item} zsh

```bash
# Clone the repo and enter it
git clone https://github.com/LSSTDESC/tables_io.git
cd tables_io

# Creating the environment from the YAML
conda env create -n tables_io_dev -f environment.yml

# Activate the environment
conda activate tables_io_dev

# Install tables_io in editable mode with dev dependencies
pip install -e '.[dev]'
```

:::

::::

To install without using Anaconda, you can instead create a python virtual environment:

::::{tab-set}

:::{tab-item} bash

```bash
# Create the virtual environment
python -m venv tables_io_dev

# Activate the virtual environment
source tables_io_dev/bin/activate

# Install tables_io in editable mode with dev dependencies
pip install -e .[dev]
```

:::

:::{tab-item} zsh

```bash
# Create the virtual environment
python -m venv tables_io_dev

# Activate the virtual environment
source tables_io_dev/bin/activate

# Install tables_io in editable mode with dev dependencies
pip install -e '.[dev]'
```

:::

::::

(running-tests)=

## Running Tests

All tests are coordinated via [pytest](https://docs.pytest.org/en/stable/). To run the tests:

```bash

python -m pytest tests

```
