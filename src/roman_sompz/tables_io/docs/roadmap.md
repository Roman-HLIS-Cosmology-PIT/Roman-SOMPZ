# Roadmap

## This version

### Breaking changes

- code reorganization means that some IO functions that were previously accessed are no longer available in that place, i.e. `initialize_HDF5_write`. Some of these functions have been made available in the created `hdf5` module.
- testUtils was moved to the tests folder and is no longer accessible for normal code

## Moving forward

- depreciate the classes and functions marked for deprecation
