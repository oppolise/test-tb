import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Find the root element (should be in the HTML template used by HtmlWebpackPlugin)
const rootElement = document.getElementById('root');

if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} else {
  console.error('Failed to find the root element. React app could not be mounted.');
}
