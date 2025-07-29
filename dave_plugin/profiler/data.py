# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

# pyre-unsafe
import gzip
import io as sysio
import json
import re
import tempfile
from json.decoder import JSONDecodeError
from typing import Dict, List, Optional

from .. import io, utils
from . import trace
from .event_parser import EventParser
from .node import OperatorNode
from .trace import BaseEvent

logger = utils.get_logger()


class RunProfileData:
    def __init__(self, worker: str, span: str, trace_json: Dict):
        self.worker = worker
        self.span = span

        # metadatas
        self.is_pytorch_lightning = trace_json.get('Framework', None) == 'pytorch-lightning'
        self.data_schema_version = trace_json.get('schemaVersion', None)
        self.device_props = trace_json.get('deviceProperties', None)

        self.profiler_start_ts = float('inf')
        self.events: List[BaseEvent] = []

        trace_body = trace_json['traceEvents']
        fwd_bwd_events = []
        for data in trace_body:
            if data.get('cat') == 'fwdbwd':
                fwd_bwd_events.append(data)
            else:
                event = trace.create_event(data, self.is_pytorch_lightning)
                if event is not None:
                    self.profiler_start_ts = min(self.profiler_start_ts, event.ts)
                    self.events.append(event)

        self.events.sort(key=lambda e: e.ts)
        self.forward_backward_events = trace.create_association_events(fwd_bwd_events)

        self.trace_file_path: str = None
        self.tid2tree: Dict[int, OperatorNode] = None
        self.pl_tid2tree: Dict[int, OperatorNode] = None


    @staticmethod
    def parse(worker, span, path, cache_dir):
        trace_path, trace_json = RunProfileData._preprocess_file(path, cache_dir)

        profile = RunProfileData.from_json(worker, span, trace_json)
        profile.trace_file_path = trace_path
        return profile

    @staticmethod
    def from_json(worker, span, trace_json: Dict):
        profile = RunProfileData(worker, span, trace_json)
        with utils.timing('Data processing'):
            profile.process()
        return profile

    @staticmethod
    def _preprocess_file(trace_path, cache_dir):
        if not io.exists(trace_path):
            raise FileNotFoundError(trace_path)

        data = io.read(trace_path)
        if trace_path.endswith('.gz'):
            data = gzip.decompress(data)

        json_reencode = False
        try:
            trace_json = json.loads(data)
        except JSONDecodeError as e:
            # Kineto may export json file with non-ascii code. before this is fixed, use a workaround
            # to handle JSONDecodeError, re-encode it and save to a temp file
            try:
                trace_json = json.loads(data, strict=False)
            except JSONDecodeError:
                with sysio.StringIO() as fout:
                    str_data = data.decode('utf-8')
                    # only replace the N/A without surrounding double quote
                    fout.write(re.sub(r'(?<!")N/A(?!")', "\"N/A\"", str_data))
                    trace_json = json.loads(fout.getvalue())
                    logger.warning('Get JSONDecodeError: %s, Re-encode it to temp file' % e.msg)
                    json_reencode = True

        # work-around to remove the 'Record Window End' events to avoid the huge end timestamp
        event_list = trace_json['traceEvents']
        end_index = None
        start_index = None
        for i in reversed(range(len(event_list))):
            if event_list[i]['name'] == 'Record Window End':
                end_index = i
            elif event_list[i]['name'].startswith('Iteration Start:'):
                start_index = i
            if start_index is not None and end_index is not None:
                break

        if start_index is not None and end_index is not None:
            dur = event_list[end_index]['ts'] - event_list[start_index]['ts']
            if dur > 24 * 3600 * 1000:
                del trace_json['traceEvents'][end_index]
                json_reencode = True

        if json_reencode:
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json.gz', dir=cache_dir, delete=False)
            fp.close()
            with gzip.open(fp.name, mode='wt') as fzip:
                fzip.write(json.dumps(trace_json))
            trace_path = fp.name

        return trace_path, trace_json

    def process(self):
        with utils.timing('EventParser.parse'):
            parser = EventParser()
            self.tid2tree, self.pl_tid2tree = parser.parse(self.events, self.forward_backward_events)
