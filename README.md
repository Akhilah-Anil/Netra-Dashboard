# NETRA — Mission Telemetry Intelligence Platform

NETRA is an advanced, production-ready aerospace telemetry anomaly detection platform. It uses a custom Two-Pass LSTM Autoencoder architecture to detect real-time deviations in satellite and mission telemetry (e.g., Battery Voltage, Solar Panel Current, Thermal readings) and provides an AI Copilot for operational diagnostics.

## Architecture

The platform is fully containerized and consists of:
1. **FastAPI Backend**: A high-performance Python backend that handles model training, inference jobs (background tasks), and REST APIs.
2. **Detection Engine**: A hybrid Deep Learning and Rule-based engine (`detection_engine.py`) using TensorFlow/Keras.
3. **Mission Control Dashboard**: A native JavaScript & HTML frontend served directly by the FastAPI backend, utilizing Plotly.js for high-fidelity interactive telemetry waveforms.

## Getting Started

### Prerequisites
- Docker & Docker Compose
- (Optional) Python 3.11+ if running natively

### Running via Docker
The easiest way to launch the platform is via Docker Compose:

```bash
docker-compose up --build -d
```

This will spin up the backend and automatically mount your local `data/`, `saved_model/`, and `results/` directories so your training weights and CSV uploads are preserved across container restarts.

The dashboard will be available at: **http://localhost:8088**

## Training the Model

Before running inference, you must train the LSTM model. 
1. Open the dashboard.
2. Expand the **Model Training** tab on the left sidebar.
3. Upload your `segments.csv` and `train_dataset.csv`.
4. Click **Train Model**.

The backend will run the two-pass training sequence (Phase 1: Macro-Pattern learning, Phase 2: Error Distribution bounding) and save the weights to `/saved_model`.

## Telemetry Inference & Copilot

Once trained, use the **Telemetry Upload** zone on the left sidebar to drop a new telemetry CSV stream. 
- The system will process it and flag anomalies based on dynamic reconstruction error thresholds.
- Anomalies will appear in the **Active Anomalies** incident table.
- Open the **AI Copilot** (bottom left) and ask for a risk assessment or status report. The Copilot uses real-time context from your latest inference run.

## API Endpoints
- `GET /health` - System health check
- `GET /status` - Model training status
- `POST /train` - Initiates background training job
- `POST /test` - Initiates background inference job
- `GET /jobs/{job_id}` - Polls job progress
- `POST /api/chat` - Queries the AI Copilot

---
*Developed by Orbtrix Space*
