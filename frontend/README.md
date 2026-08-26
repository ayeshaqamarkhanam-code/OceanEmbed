# OceanEmbed — Frontend Prototype

Frontend prototype for:
**OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations**

## Current stack
- HTML5
- CSS3
- Vanilla JavaScript
- Leaflet.js
- OpenStreetMap
- Esri satellite imagery
- Nominatim geocoding API
- Google Fonts (Inter)
- Browser localStorage for prototype account/settings state

## Run locally
Open `index.html` in a browser, or serve the folder with a simple local server.

Example:
```bash
python -m http.server 8000
```
Then open `http://localhost:8000`.

## Backend integration
The current `runInference()` function is a frontend simulation. Replace its simulated `setTimeout(...)` section with a `fetch()` call to the team's backend inference endpoint.

The frontend currently expects the selected latitude/longitude and displays a depth-temperature profile. The backend can return the predicted profile and uncertainty values, then the frontend can pass those values to `refreshProfile()` / the rendering functions.

Authentication and SSO are also prototype-only and should be replaced with the team's real authentication/SSO flow before production use.

## Project structure
OceanEmbed/
├── index.html
├── css/
│   └── style.css
├── js/
│   └── app.js
├── assets/
│   ├── images/
│   └── icons/
└── README.md
