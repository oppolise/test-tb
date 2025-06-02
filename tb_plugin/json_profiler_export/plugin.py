import json
import os
import tempfile
import shutil
import werkzeug
from werkzeug import wrappers

from tensorboard.plugins import base_plugin

from tb_plugin.torch_tb_profiler import io, utils # For is_chrome_trace_file and get_logger

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


    def get_plugin_apps(self):
        # Serve a static HTML file for the plugin's frontend
        # We will create this file in a subsequent step
        static_path = os.path.join(os.path.dirname(__file__), 'static')
        
        # Ensure the static directory exists if we plan to serve files from it
        if not os.path.exists(static_path):
            os.makedirs(static_path)
        # Create a dummy index.html if it doesn't exist, to be replaced later
        dummy_index_html_path = os.path.join(static_path, "index.html")
        if not os.path.exists(dummy_index_html_path):
            with open(dummy_index_html_path, "w") as f:
                f.write("<html><body><p>Placeholder page. Real content pending.</p></body></html>")

        return {
            '/': self.static_file_route, # Serve index.html at the root of the plugin
            '/index.html': self.static_file_route, # Explicitly serve index.html
            '/download_json_export': self.download_json_export_route,
        }

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process."""
        if not self._logdir or not os.path.exists(self._logdir):
            return False
        
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
            
            response_headers = list(self.headers) # Make a copy to extend
            response_headers.append(('Content-Disposition', 'attachment; filename="profiler_export.json"'))
            
            return werkzeug.Response(
                json_payload,
                content_type='application/json',
                headers=response_headers
            )
        except Exception as e:
            logger.error(f"Error generating JSON export: {e}", exc_info=True)
            # Return a JSON error response
            error_response = json.dumps({"error": str(e), "details": "Check TensorBoard logs for more information."})
            return werkzeug.Response(
                error_response,
                content_type='application/json',
                status=500, # Internal Server Error
                headers=self.headers
            )

    @wrappers.Request.application
    def static_file_route(self, request: werkzeug.Request):
        """Serves static files (e.g., index.html)."""
        filename = os.path.basename(request.path) or 'index.html'
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        filepath = os.path.join(static_dir, filename)

        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            logger.warning(f"Static file not found: {filepath}")
            return werkzeug.exceptions.NotFound("File not found.")

        try:
            with open(filepath, 'rb') as f:
                contents = f.read()
        except IOError:
            logger.error(f"IOError reading static file: {filepath}", exc_info=True)
            return werkzeug.exceptions.InternalServerError("Error reading file.")

        mimetype = 'text/html' # Default for index.html
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
