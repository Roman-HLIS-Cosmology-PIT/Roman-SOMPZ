============================================
tables_io: Tabular data read/write interface
============================================


``tables_io`` provides an interface for a variety of non-ASCII file formats that are commonly used within the `LSST DESC collaboration <https://lsstdesc.org/>`_. It allows users to read in data from multiple types of files through one convenient interface.

**Useful links:** `Github <https://github.com/LSSTDESC/tables_io>`_ | `PyPI <https://pypi.org/project/tables-io/>`_

Features:
---------

* reads and writes files that contain one or more data tables 
* supports a variety of file types (``fits``, ``hdf5``, ``parquet``) and tabular formats (``astropy``, ``pandas``, ``pyarrow``, ``numpy``) (see :ref:`supported-file-formats`)
* allows easy conversions between file formats and in memory tabular formats
* ability to do chunked reads and writes of ``HDF5`` and ``parquet`` files


``tables_io`` is currently being used in the following packages:

    * `qp <https://github.com/LSSTDESC/qp>`_ 
    * `RAIL <https://github.com/LSSTDESC/rail>`_

.. _cards-clickable: 

.. grid:: 2
    :gutter: 3
    :margin: 5 5 0 0
    :padding: 0

    .. grid-item-card::
        :link: quickstart
        :link-type: doc 
        :link-alt: quickstart
        :text-align: center 

        :fas:`fa-solid fa-rocket; fa-5x`

        **Getting Started**


    .. grid-item-card::
        :link: autoapi/index
        :link-type: doc
        :link-alt: api-reference
        :text-align: center

        :fas:`fa-solid fa-terminal; fa-5x`

        **API Reference**
 

.. toctree::
    :hidden:
    :maxdepth: 4
    :caption: Documentation

    quickstart
    functionoverview
    cookbook
    API Reference <autoapi/index>

.. toctree:: 
    :hidden:
    :maxdepth: 4
    :caption: Developer Documentation

    devinstall
    devstyle
    devtechdebt
    roadmap.md

.. toctree:: 
    :hidden:
    :maxdepth: 4
    :caption: Legacy Notebooks

    notebooks/index.rst

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: More

    license
    acknowledgements 
    GitHub <https://github.com/LSSTDESC/tables_io>
    