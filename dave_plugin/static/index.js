document.addEventListener('DOMContentLoaded', () => {
    const runSelect = document.getElementById('run-select');
    const treeOutput = document.getElementById('tree-output');

    // Function to render the tree structure as indented text
    function renderTree(node, depth = 0) {
        let output = '';
        const indent = '  '.repeat(depth);

        // Display node information
        output += `${indent}- ${node.name} (Self CPU: ${node.self_cpu_time_total.toFixed(2)}us, Total CPU: ${node.cpu_time_total.toFixed(2)}us)\n`;

        // Recursively render children
        if (node.children && node.children.length > 0) {
            for (const child of node.children) {
                output += renderTree(child, depth + 1);
            }
        }
        return output;
    }

    // Function to fetch profile data for a selected run
    async function fetchProfileData(run) {
        if (!run) {
            treeOutput.textContent = 'Please select a run.';
            return;
        }
        treeOutput.textContent = `Loading data for ${run}...`;
        try {
            // The endpoint for our plugin's data
            const response = await fetch(`./data/plugin/dave/profile?run=${encodeURIComponent(run)}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            if (data.error) {
                treeOutput.textContent = `Error: ${data.error}`;
                return;
            }

            let fullTreeText = '';
            for (const tid in data) {
                fullTreeText += `--- Thread ID: ${tid} ---\n`;
                const rootNode = data[tid];
                fullTreeText += renderTree(rootNode);
                fullTreeText += '\n';
            }
            treeOutput.textContent = fullTreeText;

        } catch (error) {
            treeOutput.textContent = `Failed to fetch profile data: ${error}`;
            console.error('Fetch error:', error);
        }
    }

    // Function to fetch the list of available runs
    async function initialize() {
        try {
            const response = await fetch('./data/plugin/dave/runs');
            const runs = await response.json();

            if (runs && runs.length > 0) {
                runSelect.innerHTML = '<option value="">-- Select a Run --</option>';
                runs.forEach(run => {
                    const option = document.createElement('option');
                    option.value = run;
                    option.textContent = run;
                    runSelect.appendChild(option);
                });
            } else {
                runSelect.innerHTML = '<option value="">No runs found</option>';
                treeOutput.textContent = 'No profiler runs found in the log directory.';
            }
        } catch (error) {
            treeOutput.textContent = `Failed to fetch runs: ${error}`;
            console.error('Fetch runs error:', error);
        }
    }

    // Event listener for the run selection dropdown
    runSelect.addEventListener('change', (event) => {
        fetchProfileData(event.target.value);
    });

    // Initial load
    initialize();
});
