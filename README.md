# Roman SOMPZ

Self-Organizing Map Photo-z (SOMPZ) implementation for Roman Space Telescope photometric redshift estimation.

## Overview

Roman SOMPZ uses self-organizing maps to estimate photometric redshifts from multi-band photometry. This package provides tools for:
- Creating photometry catalogs
- Training SOM models
- Assign SOM cells for each galaxy 
- Estimating n(z) distributions
- Perform uncertainty quantifications
- Summarize n(z) realizations
- Generating diagnostic plots

## Flow chart 
![plot](./job/flow-chart.png)

### Prerequisites

- Python 3.11+
- pip

### Install from source

1. Clone the repository with submodules:
```bash
git clone --recurse-submodules https://github.com/yourusername/romansompz.git
cd romansompz
```
3. Install the package:
```bash
sh install.sh
```

### Submodule dependencies

This project uses the following as git submodules:
- `rail_base` - Core RAIL (Redshift Assessment Infrastructure Layers) framework
- `tables_io` - I/O utilities for tabular data

If you cloned without `--recurse-submodules`, initialize them with:
```bash
git submodule update --init --recursive
```

## Quick Start
cd job
ceci ../yaml/test_all.yaml


## Usage Examples

See job/run_ceci.sh for examples

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## Contact

- **Author**: Chun-Hao To, Boyan Yin, Diogo Souza 
- **Email**: chunhaoto@gmail.com, boyan.yin@duke.edu, diogo.henrique@unesp.br
