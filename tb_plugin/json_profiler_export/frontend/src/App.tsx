import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [runs, setRuns] = useState<string[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [logdir, setLogdir] = useState<string>('[Loading logdir...]');
  const [workers, setWorkers] = useState<string[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [dagViewContent, setDagViewContent] = useState<string>('<p>Select a view (e.g., Dag) to see data.</p>');

  // Store the full JSON data for the currently selected run/worker context
  const [currentDetailData, setCurrentDetailData] = useState<any>(null);


  // Fetch list of runs on component mount
  useEffect(() => {
    fetch('../api/runs') // Assuming API is relative to where index.html is served from
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status} for /api/runs`);
        }
        return response.json();
      })
      .then(data => {
        setRuns(data || []);
        if (data && data.length > 0) {
          // For now, automatically select the first run to trigger data load for it
          // In future, user might click to select a run
          // setSelectedRun(data[0]);
          // For now, we still load general data from processed_info which uses the whole logdir
        }
      })
      .catch(error => {
        console.error("Error fetching runs:", error);
        setErrors(prevErrors => [...prevErrors, `Error fetching runs: ${error.message}`]);
      });

    // Fetch initial data (logdir, workers for the whole logdir, etc.)
    // This will be refactored later when a run is selected.
    // For now, it populates the initial view based on the whole logdir.
    fetch('../api/processed_info')
      .then(response => {
        if (!response.ok) {
           return response.text().then(text => {
                let errorDetails = text;
                try { const jsonError = JSON.parse(text); errorDetails = jsonError.error || jsonError.details || text; }
                catch (e) { /* Not JSON, use text as is */ }
                throw new Error(`HTTP error! status: ${response.status}, details: ${errorDetails} for /api/processed_info`);
            });
        }
        return response.json();
      })
      .then(data => {
        setCurrentDetailData(data); // Store all data from processed_info
        setLogdir(data.source_logdir || '[Logdir not available]');
        setWorkers(data.successfully_processed_files || []);
        if (data.errors && data.errors.length > 0) {
          setErrors(prevErrors => [...prevErrors, ...data.errors]);
        }
      })
      .catch(error => {
        console.error("Error fetching initial processed info:", error);
        setErrors(prevErrors => [...prevErrors, `Error fetching initial data: ${error.message}`]);
        setLogdir('[Error loading logdir]');
      });

  }, []); // Empty dependency array means this runs once on mount

  const handleDagViewClick = () => {
    document.querySelectorAll('.sidebar-section ul .view-item').forEach(li => li.classList.remove('active'));
    const dagViewLi = document.getElementById('viewDagPlaceholder');
    if (dagViewLi) dagViewLi.classList.add('active');

    if (currentDetailData) {
      try {
        const jsonString = JSON.stringify(currentDetailData, null, 2); // Pretty print
        const lines = jsonString.split('\n');
        const first10Lines = lines.slice(0, 10).join('\n');
        let content = first10Lines;
        if (lines.length > 10) {
          content += '\n... (and more lines, full data available in download)';
        }
        setDagViewContent(content);
      } catch (e: any) {
        setDagViewContent(`Error processing JSON data: ${e.message}`);
        console.error("Error stringifying or slicing JSON for Dag view:", e);
      }
    } else {
      setDagViewContent('Data not yet loaded or no run selected. Please wait or select a run.');
    }
  };

  const handleDownloadJson = () => {
      // The download link <a href="download_json_export" ...> handles this directly.
      // This function could be used if we wanted to trigger download via JS,
      // for example, after selecting a specific run and worker to get more granular data.
      // For now, the existing HTML link is sufficient for downloading the overall JSON.
      // To make it dynamic (e.g. download for selected run), this would need an API like:
      // `download_json_export?run=${selectedRun}`
      // And the backend `download_json_export_route` would need to handle this query param.
      // For now, let's just log that the button is conceptual for dynamic downloads.
      console.log("Download button clicked. Current static link will download data for the whole logdir.");
      window.location.href = '../download_json_export'; // Path relative to where index.html is served
  };


  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-section">
          <h2>Runs</h2>
          {/* Displaying the main logdir for now. Runs list will be interactive later. */}
          <div id="runNamePlaceholder" className="run-info">{logdir}</div>
          {runs.length > 0 ? (
            <ul className="run-list">
              {runs.map(runName => (
                <li key={runName} onClick={() => setSelectedRun(runName)}
                    className={selectedRun === runName ? 'active-run-item' : ''}>
                  {runName}
                </li>
              ))}
            </ul>
          ) : (
            <p className="run-info">{runs.length === 0 && !errors.find(e => e.includes("Error fetching runs")) ? "No sub-runs found." : ""}</p>
          )}
        </div>
        <div className="sidebar-section">
          <h2>Views</h2>
          <ul id="viewsListPlaceholder">
            <li id="viewDagPlaceholder" className="view-item" onClick={handleDagViewClick}>Dag</li>
          </ul>
        </div>
        <div className="sidebar-section">
          <h2>Workers</h2>
          {workers.length > 0 ? (
            <ul id="workersListPlaceholder" className="worker-list">
              {workers.map(workerName => (
                <li key={workerName}>{workerName}</li>
              ))}
            </ul>
          ) : (
            <p className="worker-info">{workers.length === 0 && !errors.find(e => e.includes("Error fetching initial data")) ? "No workers found." : ""}</p>
          )}
        </div>
        <div className="sidebar-section">
            {/* The button is now handled via React for potential future dynamic behavior */}
            <button id="downloadJsonButton" onClick={handleDownloadJson}>Download Processed JSON</button>
        </div>
      </div>
      <div className="main-content">
        <h1>Profiler Data View</h1>
        {errors.length > 0 && (
          <div id="errorsPlaceholder" className="error-display">
            <h3>Errors:</h3>
            <ul>
              {errors.map((error, index) => (
                <li key={index}><pre>{error}</pre></li>
              ))}
            </ul>
          </div>
        )}
        <div id="mainViewContentPlaceholder" style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
          {/* Dag view content is now set via state */}
          {dagViewContent}
        </div>
      </div>
    </div>
  );
}

export default App;
