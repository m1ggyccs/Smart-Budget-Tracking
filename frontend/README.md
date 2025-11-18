# Smart Budget Tracking Frontend

React dashboard for visualizing budget predictions with multi-model analysis.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── MetricCard.jsx
│   │   ├── CategorySelector.jsx
│   │   ├── UnifiedForecastChart.jsx
│   │   ├── InputControls.jsx
│   │   └── ModelCards.jsx
│   ├── utils/               # Utility functions
│   │   ├── constants.js      # API URL and constants
│   │   └── formatters.js     # Formatting functions
│   ├── App.jsx               # Main dashboard component
│   ├── main.jsx              # React entry point
│   └── index.css             # Global styles
├── index.html
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## Environment Variables

Create a `.env` file to customize the API URL:

```
VITE_API_URL=http://localhost:8001/api
```

## Build

To build for production:

```bash
npm run build
```

The built files will be in the `dist/` directory.

