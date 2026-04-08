"""Method registry for Group B experiments."""

import importlib

METHODS = ["ae", "aeot", "otswd", "nf", "fm", "cnf", "ddpm", "vae"]


def get_method(name):
    return importlib.import_module(f".{name}", package="methods")
