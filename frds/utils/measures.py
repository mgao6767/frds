"""Utility functions related to measures"""

import inspect
import importlib


def get_all_measures():
    """Return a list of (name, module) sorted by name"""
    measures_mod = importlib.import_module("frds.measures_func")
    return [
        (name, module)
        for name, module in inspect.getmembers(measures_mod, inspect.ismodule)
        # Of the modules inside frds.measures, yield those having an estimation function
        if "estimation"
        in [
            fname.lower() for fname, _ in inspect.getmembers(module, inspect.isfunction)
        ]
    ]


def get_estimation_function_of_measure(measure):
    """Given a measure module, return the estimation function"""
    if not inspect.ismodule(measure):
        raise ValueError
    for name, function in inspect.getmembers(measure, inspect.isfunction):
        if name.lower() == "estimation":
            return function
    # The measure module doesn't implement its estimation function
    raise NotImplementedError


if __name__ == "__main__":
    for name, module in get_all_measures():
        print(name, module)
        fn = get_estimation_function_of_measure(module)
        doc = inspect.getdoc(module)
        print(inspect.cleandoc(doc))
        res = fn()
        print(res)
        print(module.MEASURE_TYPE)
