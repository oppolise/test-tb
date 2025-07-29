# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

# pyre-unsafe
from typing import Any, Dict, List, Optional

from . import utils
from .profiler.node import OperatorNode

logger = utils.get_logger()


class Run:
    """ A profiler run. For visualization purpose only.
    May contain profiling results from multiple workers. E.g. distributed scenario.
    """

    def __init__(self, name, run_dir):
        self.name = name
        self.run_dir = run_dir
        self.profiles: Dict[str, RunProfile] = {}

    @property
    def workers(self):
        return sorted(list(self.profiles.keys()))

    def add_profile(self, profile: 'RunProfile'):
        self.profiles[profile.worker] = profile

    def get_profile(self, worker) -> 'RunProfile':
        if worker is None:
            raise ValueError('the worker parameter is mandatory')
        return self.profiles.get(worker, None)


class RunProfile:
    """ Cooked profiling result for a worker. For visualization purpose only.
    """

    def __init__(self, worker, span):
        self.worker = worker
        self.span = span
        self.tid2tree: Dict[int, OperatorNode] = None
        self.pl_tid2tree: Dict[int, OperatorNode] = None

    def get_operator_tree(self):
        if self.pl_tid2tree and any(self.pl_tid2tree.values()):
             # If pl_tid2tree is not empty, use it.
            root = next(iter(self.pl_tid2tree.values()))
        else:
            root = next(iter(self.tid2tree.values()))

        result = []

        def traverse_node(parent: List, node: OperatorNode):
            # A simplified dictionary representation of the node
            d = {
                'name': node.name,
                'duration': node.duration,
                'children': []
            }
            parent.append(d)
            for child in node.children:
                traverse_node(d['children'], child)

        # Start traversal from the root's children to skip the 'CallTreeRoot'
        for child in root.children:
            traverse_node(result, child)

        return result
