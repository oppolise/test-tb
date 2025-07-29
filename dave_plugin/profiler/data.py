# dave_plugin/profiler/data.py
import gzip
import json
from json.decoder import JSONDecodeError
import io as sysio
import re
from typing import Dict, List

from .. import io, utils
from . import trace
from .event_parser import EventParser
from .node import OperatorNode

logger = utils.get_logger()

class DaveProfileData:
    """
    A simplified version of RunProfileData, focusing only on getting the tid2tree.
    """
    def __init__(self, worker: str, span: str, trace_json: Dict):
        self.worker = worker
        self.span = span
        self.tid2tree: Dict[int, OperatorNode] = None
        self.pl_tid2tree: Dict[int, OperatorNode] = None

        # --- Core Logic from original RunProfileData.__init__ ---
        trace_body = trace_json.get('traceEvents', [])
        fwd_bwd_events = []
        events: List[trace.BaseEvent] = []
        is_pytorch_lightning = trace_json.get('Framework', None) == 'pytorch-lightning'

        for data in trace_body:
            if data.get('cat') == 'fwdbwd':
                fwd_bwd_events.append(data)
            else:
                event = trace.create_event(data, is_pytorch_lightning)
                if event:
                    events.append(event)

        events.sort(key=lambda e: e.ts)
        forward_backward_events = trace.create_association_events(fwd_bwd_events)
        # --- End of copied logic ---

        # Directly call the necessary parser
        self.process(events, forward_backward_events)

    @staticmethod
    def _preprocess_file(trace_path):
        """
        A simplified file preprocessor.
        """
        if not io.exists(trace_path):
            raise FileNotFoundError(trace_path)

        data = io.read(trace_path)
        if trace_path.endswith('.gz'):
            data = gzip.decompress(data)

        try:
            trace_json = json.loads(data)
        except JSONDecodeError:
            logger.warning("JSONDecodeError found. Trying to fix and re-parse...")
            try:
                str_data = data.decode('utf-8')
                fixed_str_data = re.sub(r'(?<!")N/A(?!")', "\"N/A\"", str_data)
                trace_json = json.loads(fixed_str_data)
            except Exception as e:
                logger.error(f"Failed to fix and re-parse JSON: {e}")
                return None
        return trace_json

    @classmethod
    def parse(cls, worker, span, path):
        """
        Parses a trace file to create a DaveProfileData instance.
        """
        trace_json = cls._preprocess_file(path)
        if trace_json:
            return cls(worker, span, trace_json)
        return None

    def process(self, events, fwd_bwd_map):
        """
        The core processing step. This is heavily simplified to only run
        the EventParser and get the operator tree.
        """
        logger.info("DaveProfileData: Starting EventParser...")
        with utils.timing('EventParser.parse'):
            parser = EventParser()
            self.tid2tree, self.pl_tid2tree = parser.parse(events, fwd_bwd_map)
        logger.info("DaveProfileData: EventParser finished.")
