import importlib
import os
import pkgutil
from types import ModuleType

import setuptools

import rail


class RailEnv:
    """Singleton class that manages introspection into the RAIL installation

    This will idenfity all of the installed RAIL packages, namespaces and
    files, and navigate the connections between them.

    This will also generate restructed text (`.rst`) files for the Sphinx
    autodoc documentation
    """

    _packages: dict[str, pkgutil.ModuleInfo] = {}
    _namespace_path_dict: dict[str, list[str]] = {}
    _namespace_module_dict: dict[str, list[pkgutil.ModuleInfo]] = {}
    _namespace_sub_dict: dict[str, list[str]] = {}
    _module_dict: dict[str, list[str]] = {}
    _module_path_dict: dict[str, str] = {}
    _tree: dict[str, dict] = {}
    _stage_dict: dict[str, list[str]] = {}
    _base_stages: list[type] = []

    _skip_packages: list[str] = [
        "rail.projects",
        "rail.plotting",
        "rail.cli.rail_plot",
        "rail.cli.rail_project",
    ]

    _base_packages: list[str] = [
        "rail.core",
        "rail.stages",
        "rail.interfaces",
    ]

    _base_stages_names = [
        "CatClassifier",
        "PZClassifier",
        "CatEstimator",
        "CatInformer",
        "PzInformer",
        "CatSummarizer",
        "PZSummarizer",
        "SZPZSummarizer",
        "Degrader",
        "Noisifier",
        "Selector",
        "Modeler",
        "Creator",
        "PosteriorCalculator",
        "Evaluator",
    ]

    _module_api_options: str = """   :members:
   :undoc-members:
   :show-inheritance:
"""

    _module_no_index_api_options: str = """   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
"""

    _rail_core_api_options: str = """   :members:
   :undoc-members:
   :show-inheritance:
"""

    _package_api_options: str = """   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
"""

    @classmethod
    def list_rail_packages(cls) -> dict[str, pkgutil.ModuleInfo]:
        """List all the packages that are available in the RAIL ecosystem

        Returns
        -------
        dict[str,str]:
            Dict mapping the package names to the path to the package

        """
        cls._packages = {
            pkg.name: pkg
            for pkg in pkgutil.iter_modules(rail.__path__, rail.__name__ + ".")
        }
        return cls._packages

    @classmethod
    def print_rail_packages(cls) -> None:
        """Print all the packages that are available in the RAIL ecosystem"""
        if not cls._packages:  # pragma: no cover
            cls.list_rail_packages()
        for pkg_name, pkg in cls._packages.items():
            assert isinstance(pkg[0], importlib.machinery.FileFinder)
            path = pkg[0].path
            print(f"{pkg_name} @ {path}")

    @classmethod
    def list_rail_namespaces(cls) -> dict[str, list[str]]:
        """List all the namespaces within rail

        Returns
        -------
        dict[str, list[str]]:
            Dict mapping the namespaces to the paths contributing to
            each namespace
        """
        cls._namespace_path_dict.clear()

        for path_ in rail.__path__:
            namespaces = setuptools.find_namespace_packages(path_, exclude=["_*"])
            for namespace_ in namespaces:
                # exclude stuff that starts with 'examples_data'
                if namespace_.find("examples_data") == 0:
                    continue
                full_ns = f"rail.{namespace_}"
                if full_ns in cls._namespace_path_dict:  # pragma: no cover
                    cls._namespace_path_dict[full_ns].append(path_)
                else:
                    cls._namespace_path_dict[full_ns] = [path_]

                full_path = os.path.join(path_, namespace_.replace(".", "/"))
                cls._namespace_sub_dict[full_ns] = [
                    f"{full_ns}.{ns_}"
                    for ns_ in setuptools.find_namespace_packages(
                        full_path, exclude=["_*"]
                    )
                ]

        return cls._namespace_path_dict

    @classmethod
    def print_rail_namespaces(cls) -> None:
        """Print all the namespaces that are available in the RAIL ecosystem"""
        if not cls._namespace_path_dict:
            cls.list_rail_namespaces()
        for key, val in cls._namespace_path_dict.items():
            print(f"Namespace {key}")
            for vv in val:
                print(f"     {vv}")
        for key, val in cls._namespace_sub_dict.items():
            print(f"Namespace members {key}")
            for vv in val:
                print(f"     {vv}")

    @classmethod
    def list_rail_modules(cls) -> dict[str, str]:
        """List all modules within rail

        Returns
        -------
        dict[str, str]
            Dict mapping module names to their import paths
        """
        cls._module_dict.clear()
        cls._module_path_dict.clear()
        cls._namespace_module_dict.clear()
        if not cls._namespace_path_dict:  # pragma: no cover
            cls.list_rail_namespaces()
        for key, val in cls._namespace_path_dict.items():
            cls._namespace_module_dict[key] = []
            for vv in val:
                fullpath = os.path.join(vv, key.replace(".", "/")[5:])
                modules = list(pkgutil.iter_modules([fullpath], key + "."))

                for module_ in modules:
                    # Skip hidden files
                    if module_.name.find("._") >= 0:
                        continue
                    if module_.name in cls._module_dict:  # pragma: no cover
                        cls._module_dict[module_.name].append(key)
                    else:
                        cls._module_dict[module_.name] = [key]
                    cls._namespace_module_dict[key].append(module_)
                    assert isinstance(module_[0], importlib.machinery.FileFinder)
                    cls._module_path_dict[module_.name] = module_[0].path

        return cls._module_path_dict

    @classmethod
    def print_rail_modules(cls) -> None:
        """Print all the moduels that are available in the RAIL ecosystem"""
        if not cls._module_dict:
            cls.list_rail_modules()

        for key, val in cls._module_dict.items():
            print(f"Module {key}")
            for vv in val:
                print(f"     {vv}")

    @classmethod
    def build_rail_namespace_tree(cls) -> dict[str, dict]:
        """Build a tree of the namespaces and packages in rail

        Returns
        -------
        dict[str, list[dict]]:
           Tree of the namespaces and packages in rail
        """
        cls._tree.clear()
        if not cls._namespace_module_dict:  # pragma: no cover
            cls.list_rail_modules()

        if not cls._packages:  # pragma: no cover
            cls.list_rail_packages()

        # This is tricky, we are reconstrucing the source code tree
        # from the various import names by parsing the names and
        # identifying parents
        level_dict: dict[int, set[str]] = {}
        for key in cls._namespace_module_dict:
            count = key.count(".") - 1
            if count in level_dict:
                level_dict[count].add(key)
            else:
                level_dict[count] = {key}

        for key in cls._namespace_sub_dict:
            count = key.count(".") - 1
            level_dict[count].add(key)

        def _recursive_get(a_dict: dict, keys: list[str]) -> dict:
            ret_dict = a_dict
            use_keys = []
            for key_ in keys:
                use_keys.append(key_)
                full_key = ".".join(use_keys)
                full_key = f"rail.{full_key}"
                ret_dict = ret_dict[full_key]
            return ret_dict

        depth = max(level_dict.keys())

        for current_depth in range(depth + 1):
            for key in level_dict[current_depth]:
                subs = set()

                for module_ in cls._namespace_module_dict.get(key, []):
                    subs.add(module_.name)
                for ns_ in cls._namespace_sub_dict.get(key, []):
                    subs.add(ns_)

                a_dict: dict[str, dict] = {sub_: {} for sub_ in subs}

                parent_keys = key.split(".")[1 : current_depth + 1]
                parent_dict = _recursive_get(cls._tree, parent_keys)

                if key in parent_dict:
                    for kk, vv in a_dict.items():
                        parent_dict[key][kk] = vv
                else:
                    parent_dict[key] = a_dict

        return cls._tree

    @classmethod
    def pretty_print_tree(cls, the_dict: dict | None = None, indent: str = "") -> None:
        """Utility function to help print the namespace tree

        This can be called recurisvely to walk the tree structure, which has nested dicts

        Parameters
        ----------
        the_dict:
            Current dictionary to print, if None it will print cls._tree

        indent:
            Indentation string prepended to each line
        """
        if the_dict is None:  # pragma: no cover
            the_dict = cls._tree
        for key, val in the_dict.items():
            nsname = f"{key}"
            if nsname in cls._packages:
                pkg_type = "Package"
            elif nsname in cls._module_dict:
                pkg_type = "Module"
            else:
                pkg_type = "Namespace"

            print(f"{indent}{pkg_type} {nsname}")

            cls.pretty_print_tree(val, indent=indent + "    ")

    @classmethod
    def print_rail_namespace_tree(cls) -> None:
        """Print the namespace tree in a nice way"""
        if not cls._tree:
            cls.build_rail_namespace_tree()
        cls.pretty_print_tree(cls._tree)

    @classmethod
    def do_module_api_str(cls, basedir: str, key: str, options: str) -> None:
        """Build the api rst file for a rail module

        Parameters
        ----------
        basedir
            Directory to write file to

        key
            Name of the rail module

        options
            Pre-formatted autodoc options
        """

        api_pkg_toc = f"{key} module\n"
        api_pkg_toc += "-" * len(api_pkg_toc)

        api_pkg_toc += f"""

.. automodule:: {key}
{options}

"""
        print(f"Writing {key}.rst")
        with open(
            os.path.join(basedir, "api", f"{key}.rst"), "w", encoding="utf-8"
        ) as apitocfile:
            apitocfile.write(api_pkg_toc)

    @classmethod
    def do_pkg_api_rst(
        cls, basedir: str, key: str, val: dict, options: str, module_options: str
    ) -> None:
        """Build the api rst file for a rail package

        Parameters
        ----------
        basedir
            Directory to write file to

        key
            Name of the rail package

        val
            Namespace tree for the package

        options
            Pre-formatted autodoc options for the package

        module_options
            Pre-formatted autodoc options for modules
        """

        api_pkg_toc = f"{key} package\n"
        api_pkg_toc += "=" * len(api_pkg_toc)

        api_pkg_toc += f"""

.. automodule:: {key}
{options}

.. toctree::
   :maxdepth: 4
   :caption: {key}

"""

        for k2 in val:
            api_pkg_toc += f"   {k2}\n"
            cls.do_module_api_str(basedir, k2, module_options)

        print(f"Writing {key}.rst")
        with open(
            os.path.join(basedir, "api", f"{key}.rst"), "w", encoding="utf-8"
        ) as apitocfile:
            apitocfile.write(api_pkg_toc)

    @classmethod
    def do_namespace_api_rst(cls, basedir: str, key: str, val: dict) -> None:
        """Build the api rst file for a rail namespace

        Parameters
        ----------
        basedir
            Directory to write file to

        key
            Name of the rail namespace

        val:
            Namespace tree for the namespace
        """

        api_pkg_toc = f"{key} namespace\n"
        api_pkg_toc += "=" * len(api_pkg_toc)

        api_pkg_toc += """

.. py:module:: {key}

.. toctree::
   :maxdepth: 4
   :caption: {key}

{sub_packages}
"""

        sub_packages = ""
        for k2, v2 in val.items():
            if k2 in cls._skip_packages:  # pragma: no cover
                continue

            sub_packages += f"   {k2}\n"
            if k2 in cls._module_dict:
                cls.do_module_api_str(basedir, k2, cls._module_api_options)
                continue
            cls.do_namespace_api_rst(basedir, k2, v2)

        api_pkg_toc = api_pkg_toc.format(
            key=key,
            sub_packages=sub_packages,
        )

        print(f"Writing {key}.rst")
        with open(
            os.path.join(basedir, "api", f"{key}.rst"), "w", encoding="utf-8"
        ) as apitocfile:
            apitocfile.write(api_pkg_toc)

    @classmethod
    def do_api_rst(cls, basedir: str = ".") -> None:
        """Build the top-level API documentation

        Parameters
        ----------
        basedir
            Directory to write file to
        """

        if not cls._tree:  # pragma: no cover
            cls.build_rail_namespace_tree()

        apitoc = """

*************
Base Packages
*************

.. toctree::
   :maxdepth: 4
   :caption: Base Packages

{base_packages}


**********
Namespaces
**********

.. toctree::
   :maxdepth: 4
   :caption: Namespaces

{namespaces}


******************
Algorithm Packages
******************

.. toctree::
   :maxdepth: 4
   :caption: Algorithm Packages

{algorithm_packages}

"""
        try:
            os.makedirs(basedir)
        except Exception:
            pass

        try:
            os.makedirs(os.path.join(basedir, "api"))
        except Exception:  # pragma: no cover
            pass

        base_packages = ""
        namespaces = ""
        algorithm_packages = ""

        for key, val in cls._tree.items():
            nsname = f"{key}"
            nsfile = os.path.join("api", f"{nsname}")

            if nsname in cls._packages:
                # Skip rail_projects
                if nsname in cls._skip_packages:  # pragma: no cover
                    continue
                if nsname in cls._base_packages:
                    base_packages += f"    {nsfile}\n"
                    cls.do_pkg_api_rst(
                        basedir,
                        key,
                        val,
                        cls._rail_core_api_options,
                        cls._module_no_index_api_options,
                    )
                else:  # pragma: no cover
                    algorithm_packages += f"   {nsfile}\n"
                    cls.do_pkg_api_rst(
                        basedir,
                        key,
                        val,
                        cls._package_api_options,
                        cls._module_api_options,
                    )
            else:
                cls.do_namespace_api_rst(basedir, key, val)
                namespaces += f"   {nsfile}\n"

        apitoc = apitoc.format(
            base_packages=base_packages,
            namespaces=namespaces,
            algorithm_packages=algorithm_packages,
        )
        with open(
            os.path.join(basedir, "api.rst"), "w", encoding="utf-8"
        ) as apitocfile:
            apitocfile.write(apitoc)

    @classmethod
    def import_all_packages(cls, silent: bool = False) -> None:
        """Import all the packages that are available in the RAIL ecosystem"""
        pkgs = cls.list_rail_packages()
        for pkg in pkgs:
            try:
                _imported_module = importlib.import_module(pkg)
                if not silent:
                    print(f"Imported {pkg}")
            except Exception as msg:  # pragma: no cover
                if not silent:
                    print(f"Failed to import {pkg} because: {str(msg)}")

    @classmethod
    def attach_stages(cls, to_module: ModuleType) -> None:
        """Attach all the available stages to this module

        Parameters
        ----------
        to_module
            python module we are attaching stages to

        Notes
        -----
        This allow you to do:

        .. highlight:: python
        .. code-block:: python

            from rail.stages import *

        """
        from rail.core.stage import \
            RailStage  # pylint: disable=import-outside-toplevel

        cls._stage_dict.clear()
        cls._stage_dict["RailStage"] = []
        cls._base_stages.clear()

        n_base_classes = 0
        n_stages = 0

        for stage_name, stage_info in RailStage.incomplete_pipeline_stages.items():
            if stage_info[0] in [RailStage]:
                cls._stage_dict[stage_info[0].__name__] = []
            if stage_info[0].__name__ in cls._base_stages_names:
                cls._base_stages.append(stage_info[0])
                cls._stage_dict[stage_info[0].__name__] = []
                n_base_classes += 1

        for stage_name, stage_info in RailStage.pipeline_stages.items():
            if stage_info[0].__name__ in cls._base_stages_names:
                cls._base_stages.append(stage_info[0])
                cls._stage_dict[stage_info[0].__name__] = []
                n_base_classes += 1
                continue

            setattr(to_module, stage_name, stage_info[0])
            n_stages += 1

        for stage_name, stage_info in RailStage.pipeline_stages.items():
            baseclass = "RailStage"
            for possible_base in cls._base_stages:
                if issubclass(stage_info[0], possible_base):
                    baseclass = possible_base.__name__
                    break
            cls._stage_dict[baseclass].append(stage_name)

        print(
            f"Attached {n_base_classes} base classes and {n_stages} fully formed stages to rail.stages"
        )

    @classmethod
    def print_rail_stage_dict(cls) -> None:
        """Print an dict of all the RailSages organized by their base class"""
        from rail.core.stage import \
            RailStage  # pylint: disable=import-outside-toplevel

        for key, val in cls._stage_dict.items():
            print(f"BaseClass {key}")
            for vv in val:
                stage_class = RailStage.pipeline_stages[vv][0]
                print(f"  {vv} {stage_class.__module__}.{stage_class.__name__}")

    @classmethod
    def do_stage_type_api_rst(cls, basedir: str = ".") -> None:
        """Genarate the rst files for the stage tpye documentation"""
        from rail.core.stage import \
            RailStage  # pylint: disable=import-outside-toplevel

        os.makedirs(os.path.join(basedir, "api"), exist_ok=True)

        for key, val in cls._stage_dict.items():
            if key in RailStage.incomplete_pipeline_stages:
                base_class = RailStage.incomplete_pipeline_stages[key][0]
            else:
                base_class = RailStage.pipeline_stages[key][0]

            api_stage_type = f"{key} Stage Type\n"
            api_stage_type += "*" * len(api_stage_type)
            api_stage_type += "\n"

            api_stage_type += (
                f".. autoclass:: {base_class.__module__}.{base_class.__name__}\n"
            )
            api_stage_type += "    :noindex:\n\n"

            subheader = f"{key} Sub-Classes\n"
            api_stage_type += subheader
            api_stage_type += "=" * len(subheader)
            api_stage_type += "\n"

            for vv in val:
                stage_class = RailStage.pipeline_stages[vv][0]

                api_stage_type += (
                    f".. autoclass:: {stage_class.__module__}.{stage_class.__name__}\n"
                )
                api_stage_type += "   :noindex:\n\n"

            print(f"Writing {key}_stage_type.rst")
            with open(
                os.path.join(basedir, "api", f"{key}_stage_type.rst"),
                "w",
                encoding="utf-8",
            ) as api_stage_type_file:
                api_stage_type_file.write(api_stage_type)

        header = "Types of RAIL stages\n"
        api_stage_type_index = ""
        api_stage_type_index += "*" * len(header)
        api_stage_type_index += "\n"
        api_stage_type_index += header
        api_stage_type_index += "*" * len(header)
        api_stage_type_index += "\n\n"

        api_stage_type_index += ".. toctree::\n"
        api_stage_type_index += "    :maxdepth: 4\n\n"

        for key in cls._stage_dict:
            api_stage_type_index += f"    {key}_stage_type\n"

        print("Writing stage_types.rst")
        with open(
            os.path.join(basedir, "api", "stage_types.rst"), "w", encoding="utf-8"
        ) as api_stage_type_index_file:
            api_stage_type_index_file.write(api_stage_type_index)
