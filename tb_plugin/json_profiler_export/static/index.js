export async function render() {
  console.log("JSON Profiler Export: static/index.js render() called, redirecting to index.html");
  // Ensure this redirect happens after the current execution context to avoid issues,
  // though direct assignment usually works.
  setTimeout(() => {
    document.location.href = 'index.html';
  }, 0);
}
