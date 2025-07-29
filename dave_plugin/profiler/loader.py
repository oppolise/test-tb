# pyre-unsafe

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import bisect
import os
import sys
from collections import defaultdict
from typing import List, Tuple

from .. import consts, io, utils
# For simplicity, we will assume single process and not use the custom multiprocessing
# from ..multiprocessing import Process, Queue
from multiprocessing import Process, Queue
from ..run import Run, RunProfile
from .data import RunProfileData
from .run_generator import RunGenerator

logger = utils.get_logger()


class RunLoader:
    def __init__(self, name, run_dir, caches: io.Cache):
        self.run_name = name
        self.run_dir = run_dir
        self.caches = caches
        self.queue = Queue()

    def load(self):
        workers = []
        # Span processing is removed for simplicity.
        for path in io.listdir(self.run_dir):
            if io.isdir(io.join(self.run_dir, path)):
                continue
            match = consts.WORKER_PATTERN.match(path)
            if not match:
                continue

            worker = match.group(1)
            # span is ignored.
            workers.append((worker, None, path))

        for worker, span, path in workers:
            # Simplified: no more span_index
            p = Process(target=self._process_data, args=(worker, span, path))
            p.start()
        logger.info('started all processing')

        run = Run(self.run_name, self.run_dir)
        num_items = len(workers)
        while num_items > 0:
            profile: RunProfile = self.queue.get()
            num_items -= 1
            if profile is not None:
                logger.debug('Loaded profile via mp.Queue')
                run.add_profile(profile)

        # for no daemon process, no need to join them since it will automatically join
        return run

    def _process_data(self, worker, span, path):
        # pyre-fixme[21]: Could not find module `absl.logging`.
        import absl.logging
        absl.logging.use_absl_handler()

        try:
            logger.debug('Parse trace, run_dir=%s, worker=%s', self.run_dir, path)
            # Caching mechanism is kept, but can be simplified if only local files are used.
            local_file = self.caches.get_remote_cache(io.join(self.run_dir, path))
            data = RunProfileData.parse(worker, span, local_file, self.caches.cache_dir)
            if data.trace_file_path != local_file:
                self.caches.add_file(local_file, data.trace_file_path)

            generator = RunGenerator(worker, span, data)
            profile = generator.generate_run_profile()

            logger.debug('Sending back profile via mp.Queue')
            self.queue.put(profile)
        except KeyboardInterrupt:
            logger.warning('tb_plugin receive keyboard interrupt signal, process %d will exit' % (os.getpid()))
            sys.exit(1)
        except Exception as ex:
            logger.warning('Failed to parse profile data for Run %s on %s. Exception=%s',
                           self.run_name, worker, ex, exc_info=True)
            self.queue.put(None)
        logger.debug('finishing process data')
