import os
import gzip
import json
from collections import defaultdict

from tb_plugin.torch_tb_profiler import io, utils, consts
from tb_plugin.torch_tb_profiler.profiler.data import RunProfileData
from tb_plugin.torch_tb_profiler.profiler.run_generator import RunGenerator
from tb_plugin.torch_tb_profiler.run import RunProfile

logger = utils.get_logger()

def _get_trace_files(logdir):
    """Scans the logdir for PyTorch Profiler trace files."""
    trace_files = []
    for root, _, files in io.walk(logdir):
        for file in files:
            if utils.is_chrome_trace_file(file): # Checks for *.pt.trace.json or *.pt.trace.json.gz
                trace_files.append(io.join(root, file))
    return trace_files

def process_run_data(logdir: str, cache_dir: str):
    """
    Processes profiling data from a logdir and returns a dictionary
    containing data extracted from RunProfile objects.
    """
    processed_data = {
        "workers": {},
        "all_recommendations": [],
        "errors": []
    }

    # Create a cache directory if it doesn't exist (simplified from plugin's _cache)
    # In a real plugin, this cache would be managed more robustly.
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    trace_file_paths = _get_trace_files(logdir)

    if not trace_file_paths:
        logger.warning(f"No trace files found in {logdir}")
        processed_data["errors"].append(f"No trace files found in {logdir}")
        return processed_data

    # Group trace files by run directory (similar to how main plugin discovers runs)
    # For this exporter, we might simplify and assume one run or process all files.
    # Let's group them by their immediate parent directory to mimic "runs".
    runs = defaultdict(list)
    for tf_path in trace_file_paths:
        run_dir = os.path.dirname(tf_path)
        runs[run_dir].append(tf_path)

    # For now, let's process the first run found, or all files if structured flatly.
    # A more sophisticated approach would allow selecting a run if multiple exist.

    run_profiles = []

    for run_cpath, files_in_run in runs.items():
        run_name = os.path.basename(run_cpath) if run_cpath != logdir else os.path.basename(logdir)
        logger.info(f"Processing run: {run_name} from {run_cpath}")

        for i, file_path in enumerate(files_in_run):
            worker_name_match = consts.WORKER_PATTERN.match(os.path.basename(file_path))
            if worker_name_match:
                worker_name = worker_name_match.group(1)
                span_name = worker_name_match.group(2)
                if span_name:
                    span_name = span_name[1:] # remove leading dot
            else:
                # Fallback worker name if pattern doesn't match
                worker_name = f"worker_{i}"
                span_name = None

            logger.info(f"Parsing trace file: {file_path} for worker: {worker_name}, span: {span_name}")
            try:
                # RunProfileData.parse expects worker, span, path, cache_dir
                # The span here is the iteration/step, not the plugin's span concept.
                # For simplicity, if not part of filename, we'll use None or a default.
                profile_data = RunProfileData.parse(worker_name, span_name, file_path, cache_dir)

                generator = RunGenerator(worker_name, span_name, profile_data)
                run_profile = generator.generate_run_profile()
                run_profiles.append(run_profile)

                # Store some basic info
                processed_data["workers"][worker_name] = {
                    "trace_file": file_path,
                    "span": span_name,
                    "status": "parsed",
                    "overview": run_profile.overview, # Already a dict
                    "recommendations": profile_data.recommendations, # list of strings
                    "operation_pie_by_name": run_profile.operation_pie_by_name, # dict
                    "operation_table_by_name": run_profile.operation_table_by_name, # dict
                    "kernel_pie": run_profile.kernel_pie, # dict
                    "kernel_table": run_profile.kernel_table, # dict
                    "kernel_op_table": run_profile.kernel_op_table, # dict
                    "tc_pie": run_profile.tc_pie, # dict
                    # Memory data needs specific extraction
                    "memory_summary": None,
                    "memory_curve": None,
                    # Trace data might be too large to embed directly; consider path or summary
                    "trace_file_location": run_profile.trace_file_path
                }
                if profile_data.recommendations:
                    processed_data["all_recommendations"].extend(profile_data.recommendations)

                if run_profile.memory_snapshot:
                    try:
                        processed_data["workers"][worker_name]["memory_summary"] = run_profile.get_memory_stats()
                        processed_data["workers"][worker_name]["memory_curve"] = run_profile.get_memory_curve()
                        # get_memory_events might be too verbose for a summary JSON
                    except Exception as e:
                        logger.warning(f"Could not generate memory stats for {worker_name}: {e}")
                        processed_data["workers"][worker_name]["memory_summary"] = {"error": str(e)}

            except Exception as e:
                logger.error(f"Failed to process trace file {file_path}: {e}", exc_info=True)
                processed_data["errors"].append(f"Failed to process {file_path}: {str(e)}")
                if worker_name not in processed_data["workers"]:
                     processed_data["workers"][worker_name] = {"status": "error", "trace_file": file_path, "error_message": str(e)}
                else:
                    processed_data["workers"][worker_name]["status"] = "error"
                    processed_data["workers"][worker_name]["error_message"] = str(e)

    # Deduplicate recommendations
    if processed_data["all_recommendations"]:
        processed_data["all_recommendations"] = list(sorted(set(processed_data["all_recommendations"])))

    return processed_data
