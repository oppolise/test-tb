# dave_plugin/plugin.py
import os
from werkzeug import wrappers
from tensorboard.plugins import base_plugin

from . import consts, io, utils
from .profiler.loader import DaveLoader # Import our new loader
from .profiler.node import OperatorNode

logger = utils.get_logger()

# The "major" version of the plugin.
# A change in the major version indicates a backwards-incompatible change.
PLUGIN_VERSION = '2.0.0'

class DavePlugin(base_plugin.TBPlugin):
    """A simplified profiler plugin to demonstrate tid2tree display."""

    plugin_name = 'dave' # a. Set plugin name to 'dave'

    def __init__(self, context: base_plugin.TBContext):
        super().__init__(context)
        self.logdir = self._get_logdir(context)
        self.runs = {}
        self.cache = io.Cache()

    def get_plugin_apps(self):
        return {
            '/runs': self.runs_route,
            '/profile': self.profile_route,
            '/terminate': self.terminate_route,
        }

    def is_active(self):
        return bool(self.runs)

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path='/static/index.js')

    @wrappers.Request.application
    def runs_route(self, request):
        self.runs = self.runs_impl()
        run_names = sorted(self.runs.keys())
        return utils.jsonify(run_names)

    @wrappers.Request.application
    def profile_route(self, request):
        run = request.args.get('run')
        span = request.args.get('span')
        response = self.profile_impl(run, span)
        return utils.jsonify(response)

    @wrappers.Request.application
    def terminate_route(self, request):
        # A no-op for this simplified plugin.
        return utils.jsonify(True)

    def runs_impl(self):
        """
        Scans the log directory for runs and loads them using DaveLoader.
        """
        runs = {}
        if not self.logdir or not io.isdir(self.logdir):
            return runs

        for run_name in io.listdir(self.logdir):
            # b. Use DaveLoader
            loader = DaveLoader(run_name, os.path.join(self.logdir, run_name), self.cache)
            runs[run_name] = loader.load()

        return runs

    def _serialize_tree(self, node: OperatorNode):
        """

        Recursively serializes an OperatorNode tree into a simple dictionary format.
        """
        return {
            "name": node.name,
            "self_cpu_time_total": node.self_cpu_time_total,
            "cpu_time_total": node.cpu_time_total,
            "children": [self._serialize_tree(child) for child in node.children]
        }

    def profile_impl(self, run_name, span):
        """
        Handles profile data requests. Returns the tid2tree in a simple JSON format.
        """
        run = self.runs.get(run_name, None)
        if not run:
            return {"error": f"Run '{run_name}' not found."}

        # In this simple plugin, we just get the first available profile.
        # A more robust implementation would handle spans correctly.
        profile = run.get_profile(span=span)
        if not profile or not profile.tid2tree:
            return {"error": f"No profile data or tid2tree found for run '{run_name}'"}

        # c. Format tid2tree for the frontend
        formatted_tid2tree = {}
        for tid, tree_root in profile.tid2tree.items():
            formatted_tid2tree[tid] = self._serialize_tree(tree_root)

        return formatted_tid2tree

    def _get_logdir(self, context):
        if context.flags and 'logdir' in context.flags:
            return context.flags['logdir']
        elif context.logdir:
            return context.logdir
        return None
