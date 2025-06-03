const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: './src/index.tsx', // Entry point of the application
  output: {
    path: path.resolve(__dirname, '../static/built'), // Output to the plugin's static/built directory
    filename: 'bundle.js', // Output bundle file name
    publicPath: 'static/built/', // Public URL path for assets, relative to plugin root
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.jsx'], // Resolve these extensions
  },
  module: {
    rules: [
      {
        test: /\.(ts|tsx)$/, // Rule for .ts and .tsx files
        exclude: /node_modules/,
        use: 'ts-loader', // Use ts-loader for TypeScript files
      },
      {
        test: /\.css$/, // Rule for .css files (if you add any later)
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: './src/index.html', // Template HTML file
      filename: path.resolve(__dirname, '../static/index.html'), // Output final index.html to plugin's static dir
      inject: 'body', // Inject script tags at the end of the body
    }),
  ],
  devtool: 'source-map', // Enable source maps for debugging
};
