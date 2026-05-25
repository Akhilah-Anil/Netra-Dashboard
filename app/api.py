"""
api.py — NETRA LSTM Anomaly Detector REST API
"""
import os, json, uuid, shutil, threading, traceback, asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

MODEL_DIR   = Path(os.getenv('MODEL_DIR',   'saved_model'))
DATA_DIR    = Path(os.getenv('DATA_DIR',    'data'))
RESULTS_DIR = Path(os.getenv('RESULTS_DIR', 'results'))
for d in [MODEL_DIR, DATA_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_jobs: dict = {}
_jobs_lock  = threading.Lock()
_latest_inference: dict = {}

app = FastAPI(title='NETRA Anomaly Detector', version='2.0.0')
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def dashboard():
    return FileResponse("static/index.html")

@app.get('/health')
def health():
    return {'status': 'ok', 'time': datetime.utcnow().isoformat() + 'Z'}

@app.get('/status')
def status():
    meta_path = MODEL_DIR / 'metadata.json'
    if not meta_path.exists():
        return {'trained': False, 'message': 'No model found.'}
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        'trained'       : True,
        'trained_at'    : meta.get('trained_at'),
        'threshold'     : meta.get('threshold'),
        'train_f1'      : meta.get('train_metrics', {}).get('f1'),
        'epochs_trained': meta.get('epochs_trained'),
        'sep_ratio'     : meta.get('sep_ratio'),
        'model_dir'     : str(MODEL_DIR),
    }

# ── Helpers ───────────────────────────────────────────────────────────────────
def _new_job(kind: str) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            'job_id': job_id, 'kind': kind, 'status': 'running',
            'started_at': datetime.utcnow().isoformat() + 'Z',
            'finished_at': None, 'error': None, 'result': None,
        }
    return job_id

def _finish_job(job_id, result=None, error=None):
    with _jobs_lock:
        _jobs[job_id]['status']      = 'failed' if error else 'done'
        _jobs[job_id]['finished_at'] = datetime.utcnow().isoformat() + 'Z'
        _jobs[job_id]['error']       = error
        _jobs[job_id]['result']      = result

def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, 'wb') as f:
        shutil.copyfileobj(upload.file, f)
    return dest

def _classify_severity(err, thr):
    r = err / thr if thr > 0 else float('inf')
    if r <= 1.0:   return 'Normal'
    elif r <= 2.0: return 'Mild'
    elif r <= 4.0: return 'Moderate'
    else:          return 'Severe'

def _pick_threshold(stored: float, all_mse: np.ndarray,
                    override: Optional[float]) -> tuple:
    if override is not None:
        return override, 'user override'
    if len(all_mse) == 0:
        return stored, 'no data'
    p05 = float(np.percentile(all_mse, 5))
    p95 = float(np.percentile(all_mse, 95))
    if stored < p05:
        return p95, f'scale mismatch: stored={stored:.6f} < p05={p05:.6f}, using p95={p95:.6f}'
    elif stored > p95:
        p85 = float(np.percentile(all_mse, 85))
        return p85, f'stored={stored:.6f} > p95={p95:.6f}, using p85={p85:.6f}'
    return stored, 'stored threshold in range'

# ── Inference ─────────────────────────────────────────────────────────────────
def _run_inference(job_id: str, telemetry_path: str,
                   threshold_override: Optional[float]):
    try:
        import pickle, tensorflow as tf
        from preprocessing import process_telemetry_csv

        # Load model + scalers
        model = tf.keras.models.load_model(
            str(MODEL_DIR / 'lstm_autoencoder.keras'), compile=False)
        with open(MODEL_DIR / 'metadata.json') as f:
            metadata = json.load(f)

        stored_threshold = float(metadata.get('threshold', 0.000268))
        print(f'[CONFIG] threshold={stored_threshold}, max_len={metadata.get("max_len",100)}')

        # Load + preprocess CSV
        df = pd.read_csv(telemetry_path)
        print(f'[INFERENCE] Loaded CSV shape={df.shape}')

        channel_data = process_telemetry_csv(df)

        # Run model on each channel
        all_mse, segment_meta = [], []
        for ch_name, data in channel_data.items():
            X      = data['X']
            X_pred = model.predict(X, batch_size=32, verbose=0)
            mse    = np.mean(np.square(X - X_pred), axis=(1, 2))
            print(f'[MSE] {ch_name}: min={mse.min():.6f} p50={np.median(mse):.6f} '
                  f'p95={np.percentile(mse,95):.6f} max={mse.max():.6f}')
            for i, m in enumerate(mse):
                all_mse.append(float(m))
                segment_meta.append({
                    'channel': ch_name, 'segment_id': i,
                    'mse': float(m),
                    'timestamp': str(data['timestamps'][i][0]) if len(data['timestamps']) > i else '',
                })

        all_mse_arr = np.array(all_mse)
        print(f'[INFERENCE] Total windows={len(all_mse_arr)} '
              f'min={all_mse_arr.min():.6f} p95={np.percentile(all_mse_arr,95):.6f} '
              f'max={all_mse_arr.max():.6f} stored={stored_threshold:.6f}')

        threshold, reason = _pick_threshold(
            stored_threshold, all_mse_arr, threshold_override)
        print(f'[THRESHOLD] {threshold:.6f} — {reason}')

        # Flag anomalies
        anomalies = []
        result_rows = []
        for s in segment_meta:
            flag = s['mse'] > threshold
            sev  = _classify_severity(s['mse'], threshold)
            dev  = s['mse'] / threshold if threshold > 0 else 0
            result_rows.append({
                'channel': s['channel'], 'segment': s['segment_id'],
                'timestamp': s['timestamp'],
                'reconstruction_error': s['mse'],
                'predicted_anomaly': int(flag),
                'deviation_factor': round(dev, 4),
                'severity': sev,
            })
            if flag:
                anomalies.append({**s, 'severity': sev, 'deviation_factor': dev})

        n_flagged = len(anomalies)
        total     = len(segment_meta)
        print(f'[INFERENCE] Anomalies: {n_flagged}/{total} '
              f'({100*n_flagged/max(total,1):.1f}%)')

        # Save results
        run_id  = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        run_dir = RESULTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        result_df = pd.DataFrame(result_rows)
        result_df.to_csv(run_dir / 'all_segments.csv', index=False)
        result_df[result_df['predicted_anomaly']==1].to_csv(
            run_dir / 'anomalies.csv', index=False)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ['#E84C4C' if m > threshold else '#4C9BE8' for m in all_mse]
        ax.bar(range(len(all_mse)), all_mse, color=colors, width=1.0, alpha=0.7)
        ax.axhline(threshold, color='#2CA02C', linestyle='--',
                   label=f'Threshold={threshold:.6f}')
        ax.set_title('Reconstruction Error per Segment')
        ax.set_xlabel('Segment'); ax.set_ylabel('MSE')
        ax.legend(); ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(run_dir / 'test_error_plot.png', dpi=150)
        plt.close()

        # Update latest inference cache for copilot
        sev_counts = result_df['severity'].value_counts().to_dict()
        top5 = result_df.sort_values('deviation_factor', ascending=False).head(5)
        _latest_inference.clear()
        _latest_inference.update({
            'run_id': run_id, 'n_total': total, 'n_anomalies': n_flagged,
            'pct_anomalies': round(100.0 * n_flagged / max(total, 1), 2),
            'n_severe':   int(sev_counts.get('Severe', 0)),
            'n_moderate': int(sev_counts.get('Moderate', 0)),
            'n_mild':     int(sev_counts.get('Mild', 0)),
            'threshold':  float(threshold),
            'top_anomalies': [
                {'channel': r['channel'], 'segment': r['segment'],
                 'reconstruction_error': r['reconstruction_error'],
                 'deviation_factor': r['deviation_factor'],
                 'severity': r['severity']}
                for _, r in top5.iterrows()
            ],
        })

        _finish_job(job_id, result={
            'run_id': run_id,
            'total_segments': total,
            'flagged_anomalies': n_flagged,
            'threshold_used': float(threshold),
            'threshold_reason': reason,
            'mse_p95': float(np.percentile(all_mse_arr, 95)),
            'results_dir': str(run_dir),
            'files': ['all_segments.csv', 'anomalies.csv', 'test_error_plot.png'],
        })

    except Exception:
        _finish_job(job_id, error=traceback.format_exc())

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post('/test')
async def test(
    background_tasks: BackgroundTasks,
    telemetry_file: UploadFile = File(...),
    threshold: Optional[float] = None,
):
    if not (MODEL_DIR / 'metadata.json').exists():
        raise HTTPException(status_code=400, detail='No trained model. Run /train first.')
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    path   = str(_save_upload(telemetry_file, DATA_DIR / f'test_{run_id}_telemetry.csv'))
    job_id = _new_job('test')
    background_tasks.add_task(_run_inference, job_id, path, threshold)
    return {'job_id': job_id, 'status': 'running',
            'message': 'Inference started. Poll GET /jobs/{job_id}.'}

@app.get('/jobs/{job_id}')
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return job

@app.get('/jobs')
def list_jobs():
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j['started_at'], reverse=True)
    return {'count': len(jobs), 'jobs': jobs}

@app.get('/results/{run_id}/{filename}')
def download_result(run_id: str, filename: str):
    allowed = {'all_segments.csv','anomalies.csv','test_error_plot.png','test_metrics.json'}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail='File not allowed')
    path = RESULTS_DIR / run_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(str(path))

@app.get('/model/{filename}')
def download_model_artifact(filename: str):
    allowed = {'metadata.json','loss_curve.png','train_error_plot.png','threshold_sweep.png'}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail='File not allowed')
    path = MODEL_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(str(path))

@app.post('/api/chat')
async def copilot_chat(payload: dict):
    message  = payload.get('message', '')
    api_key  = os.getenv('ANTHROPIC_API_KEY', '')
    summary  = dict(_latest_inference) if _latest_inference else None

    if not summary:
        return JSONResponse({'reply': 'No inference results yet. Upload a CSV and run analysis first.', 'used_llm': False})

    top_lines = '\n'.join([
        f"  - Channel {a['channel']} seg {a['segment']}: "
        f"error={a['reconstruction_error']:.6f}, "
        f"deviation={a['deviation_factor']:.2f}x, severity={a['severity']}"
        for a in summary.get('top_anomalies', [])
    ]) or '  No anomalies'

    system_prompt = f"""You are NETRA, an AI assistant for satellite telemetry anomaly monitoring.

Current inference results:
- Total segments: {summary['n_total']}
- Anomalies: {summary['n_anomalies']} ({summary['pct_anomalies']}%)
- Severity: {summary['n_severe']} Severe, {summary['n_moderate']} Moderate, {summary['n_mild']} Mild
- Threshold: {summary.get('threshold')}

Top anomalies:
{top_lines}

Rules: Use only data above. Channel names are raw IDs. Be concise and technical."""

    if not api_key:
        m = message.lower()
        n = summary['n_anomalies']
        top = summary.get('top_anomalies', [])
        w = top[0] if top else None
        
        # Simulated intelligent responses based on keywords
        if any(x in m for x in ['hi', 'hello', 'hey', 'greetings']):
            reply = f"Greetings. I am actively monitoring NETRA telemetry streams. Currently tracking {n} anomalies across {summary['n_total']} segments. How can I assist with your mission analysis?"
        elif any(x in m for x in ['risk', 'danger', 'impact', 'avoid', 'consequence', 'happen']):
            reply = f"WARNING: Ignoring these {n} anomalies poses a critical risk to mission integrity. Potential consequences include cascading power subsystem failure, thermal runaway in the affected battery arrays, and significantly degraded satellite lifespan. Immediate investigation is advised."
        elif any(x in m for x in ['recommend', 'action', 'fix', 'do', 'mitigate']):
            reply = "Recommended Actions:\n1. Throttle power draw on the affected subsystems.\n2. Initiate secondary cooling loops if temperatures spike.\n3. Request a priority downlink for high-resolution telemetry.\n4. Isolate circuits showing the highest deviation."
        elif any(x in m for x in ['status', 'report', 'summary']):
            reply = f"Mission Status: Suboptimal. We have {n} anomalies detected ({summary['pct_anomalies']}% of payload). "
            if w:
                reply += f"The most critical deviation is on {w['channel']} with a {w['deviation_factor']:.1f}x deviation ({w['severity']} severity)."
            else:
                reply += "All systems within nominal thresholds."
        elif any(x in m for x in ['anomal', 'channel', 'worst', 'severe']):
            reply = f"{n} anomalies detected. " + (f"Worst deviation: {w['channel']} at {w['deviation_factor']:.1f}x ({w['severity']})." if w else "No severe anomalies currently tracked.")
        else:
            reply = f"Based on current telemetry, we are tracking {n} anomalies. Subsystem instability detected. Please specify if you need a risk assessment or recommended actions."
            
        return JSONResponse({'reply': reply, 'used_llm': False})

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp   = client.messages.create(
            model='claude-sonnet-4-5', max_tokens=600,
            system=system_prompt,
            messages=[{'role': 'user', 'content': message}])
        return JSONResponse({'reply': resp.content[0].text, 'used_llm': True})
    except Exception as e:
        return JSONResponse({'reply': f'LLM error: {e}', 'used_llm': False})
