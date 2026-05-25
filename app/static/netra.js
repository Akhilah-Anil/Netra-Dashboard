// app.js — NETRA shim
// All application logic lives in the inline <script> in index.html.
// This file previously redefined doInference, toggleCopilot, doTrain etc.
// which overwrote the working versions and broke nav switching + charts.

// Clock (only thing we keep here so it doesn't need to move)
function updateClock() {
  const el = document.getElementById('clock');
  if (!el) return;
  const now = new Date();
  const h = now.getUTCHours().toString().padStart(2,'0');
  const m = now.getUTCMinutes().toString().padStart(2,'0');
  const s = now.getUTCSeconds().toString().padStart(2,'0');
  el.textContent = `${h}:${m}:${s} UTC`;
}
setInterval(updateClock, 1000);
updateClock();

console.log('NETRA — Mission Telemetry Intelligence Platform Ready');
// File label update
function showFile(input, labelId) {
    const label = document.getElementById(labelId);
    if (input.files && input.files.length > 0) {
        label.innerText = input.files[0].name;
    } else {
        label.innerText = 'orion_demo_telemetry.csv';
    }
}
