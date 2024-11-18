import contextlib
import gzip
import itertools
import json
import pickle
import sys
from importlib import import_module

import joblib
import monty
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from loguru import logger
from pandas._typing import FilePath


def gm_node(g1, g2, attrs: tuple = ("symbol",)) -> iso.GraphMatcher:
    return iso.GraphMatcher(g1, g2, node_match=lambda a, b: all(a[attr] == b[attr] for attr in attrs))


def gm_edge(g1, g2, attrs: tuple = ("bondtype",)) -> iso.GraphMatcher:
    return iso.GraphMatcher(g1, g2, edge_match=lambda a, b: all(a[attr] == b[attr] for attr in attrs))


def all_have_attributes(g: nx.Graph, attrs: tuple = ("symbol",), element: str = "node") -> bool:
    nedges = 0
    if element == "node":
        data = g.nodes.data(data=True)
    elif element == "edge":
        data = g.edges.data(data=True)
    else:
        raise ValueError("element must be 'node' or 'edge', got: {}".format(element))
    counter = dict()
    for entry in data:
        d = entry[-1]
        nedges += 1
        for attr in d:
            try:
                counter[attr] += 1
            except KeyError:
                counter[attr] = 1
    return all(counter[a] == nedges for a in attrs)


def multigraph_cycles(multigraph: nx.MultiGraph):
    """return a dict mapping edge lists of multigraph to cycles"""
    g = nx.Graph()
    for n, d in multigraph.nodes(data=True):
        g.add_node(n, **d)
    edge2k = dict()
    for u, v, k, d in multigraph.edges(data=True, keys=True):
        if k == 0:
            g.add_edge(u, v, **d, )
        e = frozenset((u, v))
        if e not in edge2k:
            edge2k[e] = [k]
        else:
            edge2k[e].append(k)

    cycles = list(nx.cycle_basis(g))
    edge_lists_to_cycles = {}
    for c in cycles:
        edge_choices = []
        for i in range(len(c)):
            head = c[i]
            if i == len(c) - 1:
                tail = c[0]
            else:
                tail = c[i + 1]
            this_edge = [(head, tail, k) for k in edge2k[frozenset((head, tail))]]
            edge_choices.append(this_edge)
        possible_edge_lists = list(itertools.product(*edge_choices))
        for el in possible_edge_lists:
            edge_lists_to_cycles[el] = c
    return edge_lists_to_cycles


def edge_hash(g: nx.MultiGraph):
    data = []
    for u, v, d in g.edges(data=True):
        e = (frozenset([(u,), (v,)]), d["voltage"], d["direction"])
        data.append(e)
    return hash(tuple(data))


def is_bond_visited(visited_vectors, visited_headtails, vector, n1, n2) -> bool:
    assert len(visited_vectors) == len(visited_headtails)
    if len(visited_vectors) == 0:
        return False
    cp = np.cross(visited_vectors, vector)
    eps = 1e-5
    cp[np.abs(cp) < eps] = 0
    zero_rows = np.where(~cp.any(axis=1))[0]
    if len(zero_rows) == 0:
        return False
    elif frozenset([n1, n2]) not in visited_headtails:
        return False
    return True


def is_3d_parallel(v1, v2, eps=1e-5):
    cp = np.cross(v1, v2)
    return np.allclose(cp, np.zeros(3), atol=eps)


def cached_import(module_path, class_name):
    # from https://docs.djangoproject.com/
    # Check whether module is loaded and fully initialized.
    if not (
            (module := sys.modules.get(module_path))
            and (spec := getattr(module, "__spec__", None))
            and getattr(spec, "_initializing", False) is False
    ):
        module = import_module(module_path)
    return getattr(module, class_name)


def import_string(dotted_path):
    """
    from https://docs.djangoproject.com/
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    try:
        return cached_import(module_path, class_name)
    except AttributeError as err:
        raise ImportError(
            'Module "%s" does not define a "%s" attribute/class'
            % (module_path, class_name)
        ) from err


def json_dump(o, fn: FilePath, gz=False):
    if gz:
        with gzip.open(fn, 'wt') as f:
            json.dump(o, f, cls=monty.json.MontyEncoder, indent=2)
    else:
        with open(fn, "w") as f:
            json.dump(o, f, cls=monty.json.MontyEncoder, indent=2)


def json_load(fn: FilePath, warning=False, disable_monty=False):
    if warning:
        logger.warning("loading file: {}".format(fn))
    if disable_monty:
        decoder = None
    else:
        decoder = monty.json.MontyDecoder

    if fn.endswith(".gz"):
        gz = True
    else:
        gz = False

    if gz:
        with gzip.open(fn, 'rt') as f:
            o = json.load(f, cls=decoder)
    else:
        with open(fn, "r") as f:
            o = json.load(f, cls=decoder)
    return o


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def pkl_dump(obj, fname: FilePath):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(fname: FilePath):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

import signal
from functools import wraps
def timeout(seconds, default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def signal_handler(signum, frame):
                raise TimeoutError("Timed out!")

            # Set up the signal handler for timeout
            signal.signal(signal.SIGALRM, signal_handler)

            # Set the initial alarm for the integer part of seconds
            signal.setitimer(signal.ITIMER_REAL, seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                return default
            finally:
                signal.alarm(0)

            return result

        return wrapper

    return decorator