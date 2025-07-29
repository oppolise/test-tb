# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

# pyre-unsafe
import atexit
import json
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from queue import Queue

import werkzeug
# pyre-fixme[21]: Could not find module `tensorboard.plugins`.
from tensorboard.plugins import base_plugin
from werkzeug import exceptions, wrappers

from . import consts, io, utils
from .profiler import RunLoader
from .run import Run

logger = utils.get_logger()


def decorate_headers(func):
    def wrapper(*args, **kwargs):
        headers = func(*args, **kwargs)
        headers.extend(DavePlugin.headers)
        return headers
    return wrapper

exceptions.HTTPException.get_headers = decorate_headers(exceptions.HTTPException.get_headers)


class DavePlugin(base_plugin.TBPlugin):
    """A simplified profiler plugin to display op tree."""

    plugin_name = 'dave'
    headers = [('X-Content-Type-Options', 'nosniff')]
    CONTENT_TYPE = 'application/json'

    def __init__(self, context: base_plugin.TBContext):
        """Instantiates DavePlugin."""
        super(DavePlugin, self).__init__(context)
        if not context.logdir and context.flags.logdir_spec:
            dirs = context.flags.logdir_spec.split(',')
            if len(dirs) > 1:
                logger.warning(f"Multiple directories are specified by --logdir_spec flag. DavePlugin will load the first one: \n {dirs[0]}")
            self.logdir = io.abspath(dirs[0].rstrip('/'))
        else:
            self.logdir = io.abspath(context.logdir.rstrip('/'))

        self._load_lock = threading.Lock()
        self._load_threads = []

        self._runs = OrderedDict()
        self._runs_lock = threading.Lock()

        self._temp_dir = tempfile.mkdtemp()
        self._cache = io.Cache(self._temp_dir)
        self._queue = Queue()

        monitor_runs = threading.Thread(target=self._monitor_runs, name='monitor_runs', daemon=True)
        monitor_runs.start()

        receive_runs = threading.Thread(target=self._receive_runs, name='receive_runs', daemon=True)
        receive_runs.start()

        def clean():
            logger.debug('starting cleanup...')
            self._cache.__exit__(*sys.exc_info())
            logger.debug('remove temporary cache directory %s' % self._temp_dir)
            shutil.rmtree(self._temp_dir)

        atexit.register(clean)

    def is_active(self):
        if self.is_loading:
            return True
        else:
            with self._runs_lock:
                return bool(self._runs)

    def get_plugin_apps(self):
        return {
            '/dave_plugin.js': self.static_file_route,
            '/index.html': self.static_file_route,
            '/runs': self.runs_route,
            '/workers': self.workers_route,
            '/op_tree': self.op_tree_route,
        }

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path='/dave_plugin.js', disable_reload=True)

    @wrappers.Request.application
    def runs_route(self, request: werkzeug.Request):
        with self._runs_lock:
            names = list(self._runs.keys())
        data = {
            'runs': names,
            'loading': self.is_loading
        }
        return self.respond_as_json(data)

    @wrappers.Request.application
    def workers_route(self, request: werkzeug.Request):
        name = request.args.get('run')
        self._validate(run=name)
        run = self._get_run(name)
        return self.respond_as_json(run.workers)

    @wrappers.Request.application
    def op_tree_route(self, request: werkzeug.Request):
        run_name = request.args.get('run')
        worker_name = request.args.get('worker')
        self._validate(run=run_name, worker=worker_name)
        run = self._get_run(run_name)
        profile = run.get_profile(worker_name)
        if profile is None:
            raise exceptions.NotFound(f"Profile not found for run '{run_name}' and worker '{worker_name}'")

        content = profile.get_operator_tree()
        return self.respond_as_json(content)

    @wrappers.Request.application
    def static_file_route(self, request: werkzeug.Request):
        filename = os.path.basename(request.path)
        extension = os.path.splitext(filename)[1]
        if extension == '.html':
            mimetype = 'text/html'
        elif extension == '.js':
            mimetype = 'application/javascript'
        else:
            mimetype = 'application/octet-stream'
        filepath = os.path.join(os.path.dirname(__file__), 'static', filename)
        try:
            with open(filepath, 'rb') as infile:
                contents = infile.read()
        except IOError:
            raise exceptions.NotFound('404 Not Found')
        return werkzeug.Response(
            contents, content_type=mimetype, headers=DavePlugin.headers
        )

    @staticmethod
    def respond_as_json(obj):
        content = json.dumps(obj)
        return werkzeug.Response(content, content_type=DavePlugin.CONTENT_TYPE, headers=DavePlugin.headers)

    @property
    def is_loading(self):
        with self._load_lock:
            return bool(self._load_threads)

    def _monitor_runs(self):
        logger.info('Monitor runs begin')
        touched = set()
        while True:
            try:
                run_dirs = self._get_run_dirs()
                for run_dir in run_dirs:
                    if run_dir not in touched:
                        touched.add(run_dir)
                        logger.info('Find run directory %s', run_dir)
                        t = threading.Thread(target=self._load_run, args=(run_dir,))
                        t.start()
                        with self._load_lock:
                            self._load_threads.append(t)
            except Exception as ex:
                logger.warning('Failed to scan runs. Exception=%s', ex, exc_info=True)
            time.sleep(consts.MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS)

    def _receive_runs(self):
        while True:
            run: Run = self._queue.get()
            if run is None:
                continue
            logger.info('Add run %s', run.name)
            with self._runs_lock:
                is_new = run.name not in self._runs
                self._runs[run.name] = run
                if is_new:
                    self._runs = OrderedDict(sorted(self._runs.items()))

    def _get_run_dirs(self):
        if not io.isdir(self.logdir):
            return
        for root, _, files in io.walk(self.logdir):
            for file in files:
                if utils.is_chrome_trace_file(file):
                    yield root
                    break

    def _load_run(self, run_dir):
        name = self._get_run_name(run_dir)
        try:
            logger.info('Load run %s', name)
            loader = RunLoader(name, run_dir, self._cache)
            run = loader.load()
            logger.info('Run %s loaded', name)
            self._queue.put(run)
        except Exception as ex:
            logger.warning('Failed to load run %s. Exception=%s', name, ex, exc_info=True)

        t = threading.current_thread()
        with self._load_lock:
            try:
                self._load_threads.remove(t)
            except ValueError:
                logger.warning('could not find the thread {}'.format(run_dir))

    def _get_run(self, name) -> Run:
        with self._runs_lock:
            run = self._runs.get(name, None)
        if run is None:
            raise exceptions.NotFound(f'could not find the run for {name}')
        return run

    def _get_run_name(self, run_dir):
        logdir = io.abspath(self.logdir)
        if run_dir == logdir:
            name = io.basename(run_dir)
        else:
            name = io.relpath(run_dir, logdir)
        return name

    def _validate(self, **kwargs):
        for name, v in kwargs.items():
            if v is None:
                raise exceptions.BadRequest(f'Must specify {name} in request url')
