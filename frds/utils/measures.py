"""Utility functions related to measures"""

import inspect
import importlib


def get_all_measures(category=None):
    """Return a list of (name, module) sorted by name"""
    measures_mod = importlib.import_module("frds.measures_func")
    for name, module in inspect.getmembers(measures_mod, inspect.ismodule):
        funcs = [f.lower() for f, _ in inspect.getmembers(module, inspect.isfunction)]
        if "estimation" in funcs:
            if category is None:
                yield name, module
            else:
                if getattr(module, "MEASURE_TYPE", None) == category:
                    yield name, module


def get_estimation_function_of_measure(measure):
    """Given a measure module, return the estimation function"""
    if not inspect.ismodule(measure):
        raise ValueError
    for name, function in inspect.getmembers(measure, inspect.isfunction):
        if name.lower() == "estimation":
            return function
    # The measure module doesn't implement its estimation function
    raise NotImplementedError


def get_name_of_measure(measure):
    if not inspect.ismodule(measure):
        raise ValueError
    return getattr(measure, "MEASURE_NAME", measure.__name__)


def get_doc_url_of_measure(measure):
    if not inspect.ismodule(measure):
        raise ValueError
    return getattr(measure, "DOC_URL", "https://frds.io")


if __name__ == "__main__":
    for name, module in get_all_measures():
        print(name, module)
        fn = get_estimation_function_of_measure(module)
        doc = inspect.getdoc(module)
        print(inspect.cleandoc(doc))
        res = fn()
        print(res)
        print(module.MEASURE_TYPE)
