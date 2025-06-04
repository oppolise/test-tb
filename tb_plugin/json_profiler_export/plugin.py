import json
import os
import tempfile
import shutil
import werkzeug
from werkzeug import wrappers

from tensorboard.plugins import base_plugin

from torch_tb_profiler import io, utils # For is_chrome_trace_file and get_logger

# Import the processor from the sibling module
from . import run_export 

logger = utils.get_logger()

class JsonProfilerExportPlugin(base_plugin.TBPlugin):
    plugin_name = 'json_export_profiler'
    # Define HTTP headers for all responses
    headers = [('X-Content-Type-Options', 'nosniff')]

    def __init__(self, context: base_plugin.TBContext):
        super().__init__(context)
        self._logdir = context.logdir
        # Create a temporary directory for caching processed trace files if needed by run_export
        # This directory will be cleaned up when TensorBoard shuts down.
        self._temp_cache_dir = tempfile.mkdtemp(prefix=f"{self.plugin_name}_cache_")
        logger.info(f"JsonProfilerExportPlugin initialized. Cache dir: {self._temp_cache_dir}")

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/index.js", disable_reload=True)

    def get_plugin_apps(self):
        # static/index.html will be served by static_file_route
        # static/index.js will be served by static_file_route
        # The download_json_export_route remains for data download.
        
        # Ensure static directory exists (should be handled by packaging)
        static_path = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(static_path):
            os.makedirs(static_path) # Fallback
            logger.warning(f"Static directory was missing and created at {static_path}. This should ideally be handled by packaging.")

        return {
            '/': self.static_file_route,  # Serves static/index.html
            '/index.html': self.static_file_route, # Serves static/index.html
            '/index.js': self.static_file_route,   # Serves static/index.js
            '/download_json_export': self.download_json_export_route,
            '/api/processed_info': self.processed_info_route,
            '/api/runs': self._serve_runs_api,  # New route for discovering runs
            '/api/workers': self._serve_workers_api, # New route for getting workers in a run
            '/api/data': self._serve_data_api, # New route for getting data for a specific worker
        }

    @wrappers.Request.application
    def _serve_data_api(self, request: werkzeug.Request):
        """Serves detailed profile data for a specific worker in a run."""
        run_name = request.args.get('run')
        worker_identifier = request.args.get('worker')

        if not run_name or not worker_identifier:
            logger.warning("/api/data: Missing 'run' or 'worker' query parameter.")
            error_response = json.dumps({"error": "Missing 'run' or 'worker' query parameter"})
            return werkzeug.Response(
                error_response,
                content_type='application/json',
                status=400, # Bad Request
                headers=self.headers
            )

        try:
            # self._temp_cache_dir is used as the cache_dir for parsing
            profile_data = run_export.get_profile_data_for_worker(
                self._logdir,
                run_name,
                worker_identifier,
                self._temp_cache_dir 
            )
            
            # get_profile_data_for_worker returns a dict, which might itself indicate an error
            # (e.g. {"status": "error", ...}). We'll return this dict as is.
            # If the data indicates an error, the client can interpret it.
            # The HTTP status code will be 200 OK if the function completed,
            # or 500 if an unexpected exception occurred in this handler.
            
            json_payload = json.dumps(profile_data)
            return werkzeug.Response(
                json_payload,
                content_type='application/json',
                headers=self.headers
            )
        except Exception as e:
            logger.error(f"Error in /api/data route for run '{run_name}', worker '{worker_identifier}': {e}", exc_info=True)
            error_response = json.dumps({
                "error": f"Failed to get profile data for run '{run_name}', worker '{worker_identifier}'.",
                "details": str(e)
            })
            return werkzeug.Response(
                error_response,
                content_type='application/json',
                status=500, # Internal Server Error
                headers=self.headers
            )

    @wrappers.Request.application
    def _serve_workers_api(self, request: werkzeug.Request):
        """Serves a list of worker identifiers for a given run."""
        run_name = request.args.get('run')
        if not run_name:
            logger.warning("/api/workers: Missing 'run' query parameter.")
            error_response = json.dumps({"error": "Missing 'run' query parameter"})
            return werkzeug.Response(
                error_response,
                content_type='application/json',
                status=400, # Bad Request
                headers=self.headers
            )

        try:
            worker_identifiers = run_export.get_workers_for_run(self._logdir, run_name)
            
            if not isinstance(worker_identifiers, list):
                logger.error(f"/api/workers: get_workers_for_run returned type {type(worker_identifiers)}, expected list. Forcing to empty list.")
                worker_identifiers = []
                # Or, alternatively, raise an internal error:
                # raise ValueError(f"get_workers_for_run returned unexpected type: {type(worker_identifiers)}")

            json_payload = json.dumps(worker_identifiers)
            return werkzeug.Response(
                json_payload,
                content_type='application/json',
                headers=self.headers
            )
        except Exception as e:
            logger.error(f"Error in /api/workers route for run '{run_name}': {e}", exc_info=True)
            error_response = json.dumps({
                "error": f"Failed to get workers for run '{run_name}'.",
                "details": str(e)
            })
            return werkzeug.Response(
                error_response,
                content_type='application/json',
                status=500,
                headers=self.headers
            )

    @wrappers.Request.application
    def _serve_runs_api(self, request: werkzeug.Request):
        """Serves a list of available runs."""
        try:
            runs = run_export.discover_runs(self._logdir)
            # Ensure runs is a list for valid JSON, even if empty or None from discover_runs
            if not isinstance(runs, list):
                logger.error(f"/api/runs: discover_runs returned type {type(runs)}, expected list. Forcing to empty list.")
                runs = [] 
                # Or, alternatively, raise an internal error:
                # raise ValueError(f"discover_runs returned unexpected type: {type(runs)}")

            json_payload = json.dumps(runs)
            return werkzeug.Response(
                json_payload,
                content_type='application/json',
                headers=self.headers
            )
        except Exception as e:
            logger.error(f"Error in /api/runs route: {e}", exc_info=True)
            error_response = json.dumps({
                "error": "Failed to discover runs.",
                "details": str(e)
            })
            return werkzeug.Response(
                error_response,
                content_type='application/json',
                status=500,
                headers=self.headers
            )

    @wrappers.Request.application
    def processed_info_route(self, request: werkzeug.Request):
        logger.info("API route /api/processed_info called.")
        try:
            data_dict = run_export.process_run_data(self._logdir, self._temp_cache_dir)
            # We already log the content of data_dict in process_run_data or can do so here if needed
            # For example: logger.debug(f"Data for API: {data_dict}")
            
            # Ensure all_recommendations is a list, even if empty, for valid JSON
            if "all_recommendations" not in data_dict:
                data_dict["all_recommendations"] = []
            if "errors" not in data_dict:
                data_dict["errors"] = []
            if "successfully_processed_files" not in data_dict:
                data_dict["successfully_processed_files"] = []
            # No need to jsonify workers if it's already a dict of dicts.
            # Ensure it exists for valid JSON.
            if "workers" not in data_dict:
                data_dict["workers"] = {}


            json_payload = json.dumps(data_dict) # No indent needed for API response usually
            
            return werkzeug.Response(
                json_payload,
                content_type='application/json',
                headers=self.headers # Use class-defined headers
            )
        except Exception as e:
            logger.error(f"Error in /api/processed_info route: {e}", exc_info=True)
            error_response = json.dumps({"error": str(e), "details": "Check server logs for more information."})
            return werkzeug.Response(
                error_response,
                content_type='application/json',
                status=500, # Internal Server Error
                headers=self.headers
            )

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process."""
        if not self._logdir or not os.path.exists(self._logdir):
            return False
        
        # Check for trace files to activate plugin
        for _root, _dirs, files in io.walk(self._logdir):
            for file_name in files:
                if utils.is_chrome_trace_file(file_name):
                    logger.info(f"JsonProfilerExportPlugin is active. Found: {file_name}")
                    return True
        logger.info("JsonProfilerExportPlugin is not active. No trace files found.")
        return False

    @wrappers.Request.application
    def download_json_export_route(self, request: werkzeug.Request):
        """Handles the /download_json_export route."""
        try:
            logger.info(f"Processing run data for logdir: {self._logdir}")
            data_dict = run_export.process_run_data(self._logdir, self._temp_cache_dir)
            
            json_payload = json.dumps(data_dict, indent=4)
            
            response_headers = list(self.headers) # Make a copy
            response_headers.append(('Content-Disposition', 'attachment; filename="profiler_export.json"'))
            
            return werkzeug.Response(
                json_payload,
                content_type='application/json',
                headers=response_headers
            )
        except Exception as e:
            logger.error(f"Error generating JSON export: {e}", exc_info=True)
            error_response = json.dumps({"error": str(e), "details": "Check TensorBoard logs for more information."})
            return werkzeug.Response(
                error_response,
                content_type='application/json',
                status=500, 
                headers=self.headers
            )

    @wrappers.Request.application
    def static_file_route(self, request: werkzeug.Request):
        full_request_path = request.path # e.g., /data/plugin/json_export_profiler/index.js
        logger.debug(f"static_file_route received full_request_path: {full_request_path}")

        if full_request_path.endswith('/index.js'):
            filename = 'index.js'
        elif full_request_path.endswith('/index.html') or full_request_path.endswith('/'):
            filename = 'index.html'
        else:
            logger.warning(f"static_file_route: path '{full_request_path}' did not end with /index.js, /index.html, or /. Trying basename.")
            basename = os.path.basename(full_request_path)
            if basename in ('index.html', 'index.js'): # Check if basename is one of our known files
                 filename = basename
            else:
                logger.error(f"Could not determine appropriate file for path: {full_request_path}. Basename: {basename}")
                return werkzeug.exceptions.NotFound("File not found or path not understood.")
        
        logger.info(f"Determined filename: '{filename}' for request_path: '{full_request_path}'")

        static_dir_root = os.path.join(os.path.dirname(__file__), 'static')
        filepath = os.path.join(static_dir_root, filename)

        # Security: Check that the resolved path is still within the static directory
        if not os.path.abspath(filepath).startswith(os.path.abspath(static_dir_root)):
            logger.warning(f"Attempt to access file outside static directory: {filepath} (derived from request path {request.path})")
            return werkzeug.exceptions.NotFound("File not found.")

        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            logger.warning(f"Static file not found: {filepath} (derived from request path {request.path})")
            if filename == 'index.html' and (full_request_path.endswith('/') or full_request_path.endswith('/index.html')):
                 logger.error(f"Main index.html not found at {filepath}. Plugin UI will not load.")
                 error_html = "<html><body><h1>Plugin Error</h1><p>Main UI file (index.html) is missing. Ensure plugin is installed correctly.</p></body></html>"
                 return werkzeug.Response(error_html, content_type='text/html', status=500, headers=self.headers)
            return werkzeug.exceptions.NotFound("File not found.")
        
        try:
            with open(filepath, 'rb') as f:
                contents = f.read()
        except IOError:
            logger.error(f"IOError reading static file: {filepath}", exc_info=True)
            return werkzeug.exceptions.InternalServerError("Error reading file.")

        mimetype = 'text/html' # Default
        if filename.endswith('.js'):
            mimetype = 'application/javascript'
        elif filename.endswith('.css'):
            mimetype = 'text/css'
        
        response_headers = list(self.headers)
        return werkzeug.Response(
            contents,
            content_type=mimetype,
            headers=response_headers
        )

    def on_reload(self):
        """Called when TensorBoard is reloaded."""
        # Cleanup old temp cache if it exists and create a new one
        if hasattr(self, '_temp_cache_dir') and os.path.exists(self._temp_cache_dir):
            shutil.rmtree(self._temp_cache_dir)
        self._temp_cache_dir = tempfile.mkdtemp(prefix=f"{self.plugin_name}_cache_")
        logger.info(f"JsonProfilerExportPlugin reloaded. New cache dir: {self._temp_cache_dir}")


    def on_shutdown(self):
        """Called when TensorBoard is shutting down."""
        if hasattr(self, '_temp_cache_dir') and os.path.exists(self._temp_cache_dir):
            logger.info(f"Cleaning up cache directory: {self._temp_cache_dir}")
            shutil.rmtree(self._temp_cache_dir)
        else:
            logger.info("No cache directory to clean up or already removed.")
