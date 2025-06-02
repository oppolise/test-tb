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


    def get_plugin_apps(self):
        # Serve a static HTML file for the plugin's frontend
        # static/index.html will be read and templated by serve_index_html
        # Other static assets can be served by static_file_route if needed
        static_path = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(static_path):
             # This should ideally be handled by packaging, but as a fallback:
            os.makedirs(static_path)
            logger.warning(f"Static directory created at {static_path}. It should be part of the package.")
            # We expect index.html to be present in static dir due to packaging.
            # No longer creating a dummy index.html here.

        return {
            '/': self.serve_index_html,
            '/index.html': self.serve_index_html,
            '/download_json_export': self.download_json_export_route,
            # Example: '/static_assets/<path:filename>': self.static_file_route,
        }

    @wrappers.Request.application
    def serve_index_html(self, request: werkzeug.Request):
        try:
            data_dict = run_export.process_run_data(self._logdir, self._temp_cache_dir)
            successfully_processed_files = data_dict.get("successfully_processed_files", [])
            processing_errors = data_dict.get("errors", [])

            file_list_html = "<h3>Processed Files:</h3>"
            if successfully_processed_files:
                file_list_html += "<ul>"
                for filename in successfully_processed_files:
                    file_list_html += f"<li>{werkzeug.utils.escape(filename)}</li>"
                file_list_html += "</ul>"
            else:
                file_list_html += "<p>No trace files were successfully processed.</p>"

            if processing_errors:
                file_list_html += "<h3>Processing Errors:</h3><ul>"
                for error_msg in processing_errors:
                    file_list_html += f"<li><pre>{werkzeug.utils.escape(error_msg)}</pre></li>"
                file_list_html += "</ul>"


            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            index_html_path = os.path.join(static_dir, 'index.html')

            if not os.path.exists(index_html_path):
                logger.error(f"index.html not found at {index_html_path}")
                # Attempt to create a very basic index.html with placeholder if missing, though it should be packaged.
                # This is a fallback to prevent complete failure if packaging somehow misses the file.
                error_html_content = "<html><head><title>Error</title></head><body><h1>Plugin UI Error</h1><p>index.html is missing.</p> <!-- %PROCESSED_FILES_INFO% --> </body></html>"
                with open(index_html_path, 'w', encoding='utf-8') as f_err:
                    f_err.write(error_html_content)
                logger.warning(f"Created a fallback index.html at {index_html_path} as it was missing.")
                # return werkzeug.exceptions.InternalServerError("Plugin UI file (index.html) not found. It should be part of the package.")


            with open(index_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Replace placeholder with the list of files and errors
            html_content = html_content.replace("<!-- %PROCESSED_FILES_INFO% -->", file_list_html)

            return werkzeug.Response(
                html_content,
                content_type='text/html',
                headers=self.headers # Use class-defined headers
            )
        except Exception as e:
            logger.error(f"Error serving index.html: {e}", exc_info=True)
            # Fallback or simple error page
            error_html = f"<html><body><h1>Error</h1><p>Could not load plugin UI: {werkzeug.utils.escape(str(e))}</p></body></html>"
            return werkzeug.Response(
                error_html,
                content_type='text/html',
                status=500,
                headers=self.headers
            )

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
