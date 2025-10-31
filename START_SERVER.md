# Starting the DDSP Neural Cello Server

## Quick Start

1. **Build the CSS file** (first time and when CSS changes):
   ```bash
   npm run build:css
   ```

2. **Start the Python server**:
   ```bash
   python ddsp_server_hybrid.py
   ```
   
   Or on Windows:
   ```cmd
   python ddsp_server_hybrid.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

## Troubleshooting

### Connection Refused Error

If you see `ERR_CONNECTION_REFUSED`, the server is not running. Make sure:

1. The Python server is started (see step 2 above)
2. You see the server startup message:
   ```
   ============================================================
   Hybrid DDSP Server Starting...
   ============================================================
   Server running on http://localhost:8000
   ```

### Tailwind CSS Warning

If you see a Tailwind CSS warning about using the CDN in production, make sure:

1. You've run `npm run build:css` to build the CSS file
2. The `public/styles.css` file exists
3. The HTML file uses `<link rel="stylesheet" href="/styles.css">` instead of the CDN

### Rebuilding CSS During Development

If you're modifying the HTML or CSS, you can:

- **Build once**: `npm run build:css`
- **Watch for changes**: `npm run watch:css` (runs in background and rebuilds on changes)

## Server Endpoints

- `GET /` - Frontend UI
- `GET /health` - Health check (returns server status and available models)
- `GET /api/training/status` - Training status
- `POST /api/upload-midi` - Upload MIDI file for synthesis
- `GET /api/download/<filename>` - Download generated audio

## Dependencies

Make sure you have:
- Python 3.8+ installed
- Node.js 18+ installed
- Required Python packages: `pip install -r requirements.txt`
- Required Node packages: `npm install`

