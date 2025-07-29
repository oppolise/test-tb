document.addEventListener('DOMContentLoaded', () => {
  const runSelect = document.getElementById('run-select');
  const workerSelect = document.getElementById('worker-select');
  const loadButton = document.getElementById('load-button');
  const opTreeContainer = document.getElementById('op-tree-container');
  const loadingMessage = document.getElementById('loading-message');

  // Fetches the list of available runs and populates the run dropdown.
  function fetchRuns() {
    fetch('./runs')
      .then(response => response.json())
      .then(data => {
        runSelect.innerHTML = '';
        data.runs.forEach(run => {
          const option = document.createElement('option');
          option.value = run;
          option.textContent = run;
          runSelect.appendChild(option);
        });
        // After fetching runs, fetch workers for the first run.
        if (data.runs.length > 0) {
          fetchWorkers(data.runs[0]);
        }
      })
      .catch(error => console.error('Error fetching runs:', error));
  }

  // Fetches the list of available workers for a given run.
  function fetchWorkers(run) {
    fetch(`./workers?run=${run}`)
      .then(response => response.json())
      .then(workers => {
        workerSelect.innerHTML = '';
        workers.forEach(worker => {
          const option = document.createElement('option');
          option.value = worker;
          option.textContent = worker;
          workerSelect.appendChild(option);
        });
      })
      .catch(error => console.error('Error fetching workers:', error));
  }

  // Renders the operator tree from the provided data.
  function renderTree(nodes, parentElement) {
    const ul = document.createElement('ul');
    nodes.forEach(node => {
      const li = document.createElement('li');

      const nameSpan = document.createElement('span');
      nameSpan.textContent = node.name;

      const detailsSpan = document.createElement('span');
      detailsSpan.className = 'node-details';
      detailsSpan.textContent = `(duration: ${node.duration} us)`;

      li.appendChild(nameSpan);
      li.appendChild(detailsSpan);

      if (node.children && node.children.length > 0) {
        renderTree(node.children, li);
      }
      ul.appendChild(li);
    });
    parentElement.appendChild(ul);
  }

  // Fetches and displays the operator tree.
  function loadOpTree() {
    const selectedRun = runSelect.value;
    const selectedWorker = workerSelect.value;
    if (!selectedRun || !selectedWorker) {
      opTreeContainer.innerHTML = 'Please select a run and a worker.';
      return;
    }

    loadingMessage.style.display = 'block';
    opTreeContainer.innerHTML = '';

    fetch(`./op_tree?run=${selectedRun}&worker=${selectedWorker}`)
      .then(response => response.json())
      .then(treeData => {
        loadingMessage.style.display = 'none';
        if (treeData && treeData.length > 0) {
          renderTree(treeData, opTreeContainer);
        } else {
          opTreeContainer.innerHTML = 'No operator tree data found.';
        }
      })
      .catch(error => {
        loadingMessage.style.display = 'none';
        opTreeContainer.innerHTML = `Error loading operator tree: ${error}`;
        console.error('Error fetching op tree:', error);
      });
  }

  // Event Listeners
  runSelect.addEventListener('change', () => {
    fetchWorkers(runSelect.value);
  });

  loadButton.addEventListener('click', loadOpTree);

  // Initial data load
  fetchRuns();
});
