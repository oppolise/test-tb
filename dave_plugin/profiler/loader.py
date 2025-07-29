# dave_plugin/profiler/loader.py
import bisect
import os
import sys
from collections import defaultdict
from typing import List, Tuple

from .. import consts, io, utils
from ..multiprocessing import Process, Queue
from ..run import Run, RunProfile # Note: We might simplify Run/RunProfile later
from .data import DaveProfileData # Import our new simplified data class

logger = utils.get_logger()

# A simplified RunProfile to hold just the tree
class DaveRunProfile(RunProfile):
    def __init__(self, worker, span):
        super().__init__(worker, span)
        self.tid2tree = None

class DaveLoader:
    def __init__(self, name, run_dir, caches: io.Cache):
        self.run_name = name
        self.run_dir = run_dir
        self.caches = caches
        self.queue = Queue()

    def load(self):
        workers = []
        spans_by_workers = defaultdict(list)
        for path in io.listdir(self.run_dir):
            if io.isdir(io.join(self.run_dir, path)):
                continue
            match = consts.WORKER_PATTERN.match(path)
            if not match:
                continue

            worker = match.group(1)
            span = match.group(2)
            if span is not None:
                span = span[1:]
                bisect.insort(spans_by_workers[worker], span)
            workers.append((worker, span, path))

        span_index_map = {}
        for worker, span_array in spans_by_workers.items():
            for i, span in enumerate(span_array, 1):
                span_index_map[(worker, span)] = i

        for worker, span, path in workers:
            span_index = None if span is None else span_index_map.get((worker, span))
            p = Process(target=self._process_data, args=(worker, span_index, path))
            p.start()
        logger.info(f'DaveLoader started {len(workers)} processing jobs')

        run = Run(self.run_name, self.run_dir)
        num_items = len(workers)
        while num_items > 0:
            item: DaveRunProfile = self.queue.get()
            num_items -= 1
            if item:
                logger.debug(f'Loaded profile for worker {item.worker} via mp.Queue')
                run.add_profile(item)

        # No distributed processing needed as we only care about the tree from each worker
        return run

    def _process_data(self, worker, span, path):
        try:
            logger.debug('Parse trace, run_dir=%s, worker=%s', self.run_dir, path)
            local_file = self.caches.get_remote_cache(io.join(self.run_dir, path))

            # Use our new DaveProfileData class
            data = DaveProfileData.parse(worker, span, local_file)

            if data and data.tid2tree:
                # Create a simplified profile object to send back
                profile = DaveRunProfile(worker, span)
                profile.tid2tree = data.tid2tree
                # We can also add pl_tid2tree if needed
                # profile.pl_tid2tree = data.pl_tid2tree

                logger.debug(f'Sending back profile for worker {worker} via mp.Queue')
                self.queue.put(profile)
            else:
                logger.warning(f"Failed to get tid2tree for worker {worker}")
                self.queue.put(None)

        except Exception as ex:
            logger.warning('Failed to parse profile data for Run %s on %s. Exception=%s',
                           self.run_name, worker, ex, exc_info=True)
            self.queue.put(None)
        logger.debug(f'Finishing process data for worker {worker}')
