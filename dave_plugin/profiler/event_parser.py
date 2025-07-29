# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

# pyre-unsafe
import sys
from collections import defaultdict
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Tuple

from .. import utils
from .node import (CommunicationNode, DeviceNode, ModuleNode, OperatorNode, PLModuleNode, PLProfileNode,
                   ProfilerStepNode, RuntimeNode, create_operator_node)
from .op_tree import OpTreeBuilder
from .trace import BaseEvent, DurationEvent, EventTypes, KernelEvent, NcclOpNameSet, GlooOpNameSet

logger = utils.get_logger()

class NodeParserMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_data: Dict[int, CommunicationNode] = {}
        self.runtime_node_list: List[RuntimeNode] = []

    def parse_nodes(self, events: Iterable[BaseEvent]):
        tid2list: Dict[int, List[OperatorNode]] = defaultdict(list)
        pl_tid2list: Dict[int, List[PLProfileNode]] = defaultdict(list)
        tid2zero_rt_list: Dict[int, List[RuntimeNode]] = defaultdict(list)
        corrid_to_device: Dict[int, List[DeviceNode]] = defaultdict(list)
        corrid_to_runtime: Dict[int, RuntimeNode] = {}
        externalid_to_runtime: Dict[int, List[RuntimeNode]] = defaultdict(list)

        for event in events:
            if event.type == EventTypes.MEMORY:
                continue
            self._parse_node(
                event,
                corrid_to_device,
                corrid_to_runtime,
                externalid_to_runtime,
                tid2list,
                pl_tid2list,
                tid2zero_rt_list)

        # associate CUDA Runtimes with CPU events
        for op_list in tid2list.values():
            for op in op_list:
                runtime_nodes = externalid_to_runtime.pop(op.external_id, [])
                if runtime_nodes:
                    op.runtimes.extend(runtime_nodes)
        for ext_id in externalid_to_runtime:
            if ext_id != 0:
                logger.warning("{} Runtime with external id {} don't correlate to any operator!".format(
                    len(externalid_to_runtime[ext_id]), ext_id))

        staled_device_nodes = []
        for device_nodes in corrid_to_device.values():
            staled_device_nodes.extend([n for n in device_nodes if n.type == EventTypes.KERNEL])

        return tid2list, tid2zero_rt_list, staled_device_nodes, pl_tid2list

    def _parse_node(self,
                    event: DurationEvent,
                    corrid_to_device: Dict[int, List[DeviceNode]],
                    corrid_to_runtime: Dict[int, RuntimeNode],
                    externalid_to_runtime: Dict[int, List[RuntimeNode]],
                    tid2list: Dict[int, List[OperatorNode]],
                    pl_tid2list: Dict[int, List[PLProfileNode]],
                    tid2zero_rt_list: Dict[int, List[RuntimeNode]]):
        corrid = event.correlation_id
        tid = event.tid
        if event.type in [EventTypes.KERNEL, EventTypes.MEMCPY, EventTypes.MEMSET]:
            device_node = DeviceNode.create(event)
            if corrid in corrid_to_runtime:
                rt_node = corrid_to_runtime[corrid]
                if rt_node.device_nodes is None:
                    rt_node.device_nodes = []
                rt_node.device_nodes.append(device_node)
            else:
                corrid_to_device[corrid].append(device_node)
        elif event.type == EventTypes.RUNTIME:
            device_nodes = corrid_to_device.pop(corrid, None)
            rt_node = RuntimeNode.create(event, device_nodes)
            corrid_to_runtime[corrid] = rt_node
            externalid_to_runtime[rt_node.external_id].append(rt_node)
            if rt_node.external_id == 0:
                tid2zero_rt_list[tid].append(rt_node)
            self.runtime_node_list.append(rt_node)
        elif event.type in [EventTypes.PYTHON,
                            EventTypes.OPERATOR,
                            EventTypes.PL_MODULE,
                            EventTypes.PROFILER_STEP,
                            EventTypes.MODULE,
                            EventTypes.USER_ANNOTATION]:
            if event.type == EventTypes.PROFILER_STEP:
                op_node = ProfilerStepNode.create(event)
            elif event.type == EventTypes.MODULE:
                op_node = ModuleNode.create(event)
            elif event.type == EventTypes.PL_MODULE:
                op_node = PLModuleNode.create(event)
            else:
                op_node = create_operator_node(event)

            if op_node:
                tid2list[int(tid)].append(op_node)
        elif event.type == EventTypes.PL_PROFILE:
            op_node = PLProfileNode.create(event)
            pl_tid2list[int(tid)].append(op_node)


class EventParser(NodeParserMixin):
    def __init__(self):
        super().__init__()

    def parse(self, events: Iterable[BaseEvent], fwd_bwd_map: Dict[int, int]) -> Dict[int, List[OperatorNode]]:
        with utils.timing('EventParser: parse nodes'):
            tid2list, tid2zero_rt_list, staled_device_nodes, pl_tid2list = self.parse_nodes(events)

        with utils.timing('EventParser: build operator tree'):
            builder = OpTreeBuilder()
            tid2tree = builder.build_tree(tid2list, tid2zero_rt_list, staled_device_nodes, fwd_bwd_map=fwd_bwd_map)
            pl_tid2tree = builder.build_tree(pl_tid2list, {}, [], {})

        return tid2tree, pl_tid2tree
