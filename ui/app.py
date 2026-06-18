"""
NuanceIQ — Glassmorphism Dashboard
Run: streamlit run ui/app.py
"""

import os
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="NuanceIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stSidebar"], [data-testid="stDecoration"] {display: none !important;}
.block-container {padding: 0 !important; max-width: 100% !important;}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

def get_health():
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).json()
    except Exception:
        return None

def get_metrics():
    try:
        return requests.get(f"{API_BASE}/metrics", timeout=3).json()
    except Exception:
        return None

def predict_text(text: str):
    return requests.post(
        f"{API_BASE}/predict",
        json={"text": text},
        timeout=30
    ).json()

health = get_health()
metrics = get_metrics()

api_status = "online" if health and health.get("status") == "ok" else "offline"
model_status = "loaded" if health and health.get("model_loaded") else "offline"
redis_status = "connected" if health and health.get("redis_connected") else "offline"

accuracy = f"{metrics['accuracy']*100:.1f}%" if metrics else "—"
f1 = f"{metrics['f1']*100:.1f}%" if metrics else "—"
precision = f"{metrics['precision']*100:.1f}%" if metrics else "—"
recall = f"{metrics['recall']*100:.1f}%" if metrics else "—"

history_rows = ""
for h in st.session_state.history[:6]:
    label_class = "pos" if h["Label"] == "POSITIVE" else "neg"
    cached_html = '<span class="cache-hit">⚡ Hit</span>' if h.get("Cached") == "Hit" else "—"
    history_rows += f"""
    <tr>
        <td class="time-cell">{h['Time']}</td>
        <td class="text-cell">{h['Text']}</td>
        <td><span class="mini-badge {label_class}">{h['Label'].capitalize()}</span></td>
        <td class="conf-cell">{h['Confidence']}</td>
        <td>{cached_html}</td>
    </tr>
    """

if not history_rows:
    history_rows = """
    <tr>
        <td colspan="5" class="empty-row">No predictions yet — analyze some text above</td>
    </tr>
    """

HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NuanceIQ</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    background: #0a0812;
    overflow-x: hidden;
    color: #e2e0f0;
  }}

  /* ── Animated background ── */
  .bg {{
    position: fixed; inset: 0; z-index: 0;
    background: radial-gradient(ellipse at 20% 20%, rgba(83,74,183,0.35) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(139,92,246,0.25) 0%, transparent 50%),
                radial-gradient(ellipse at 60% 10%, rgba(168,85,247,0.2) 0%, transparent 40%),
                #0a0812;
  }}

  .orb {{
    position: fixed; border-radius: 50%; filter: blur(80px); z-index: 0; pointer-events: none;
  }}
  .orb1 {{
    width: 500px; height: 500px; top: -150px; left: -100px;
    background: rgba(83,74,183,0.25);
    animation: drift1 12s ease-in-out infinite alternate;
  }}
  .orb2 {{
    width: 400px; height: 400px; bottom: -100px; right: -80px;
    background: rgba(139,92,246,0.2);
    animation: drift2 15s ease-in-out infinite alternate;
  }}
  .orb3 {{
    width: 300px; height: 300px; top: 40%; left: 60%;
    background: rgba(168,85,247,0.15);
    animation: drift3 10s ease-in-out infinite alternate;
  }}

  @keyframes drift1 {{ from {{ transform: translate(0,0) scale(1); }} to {{ transform: translate(60px,40px) scale(1.1); }} }}
  @keyframes drift2 {{ from {{ transform: translate(0,0) scale(1); }} to {{ transform: translate(-40px,-60px) scale(1.15); }} }}
  @keyframes drift3 {{ from {{ transform: translate(0,0); }} to {{ transform: translate(-30px,50px); }} }}

  /* ── Layout ── */
  .app {{ position: relative; z-index: 1; padding: 24px; max-width: 1400px; margin: 0 auto; }}

  /* ── Glass mixin ── */
  .glass {{
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
  }}

  .glass-strong {{
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(32px) saturate(200%);
    -webkit-backdrop-filter: blur(32px) saturate(200%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
  }}

  /* ── Topbar ── */
  .topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; margin-bottom: 24px;
  }}

  .logo {{ display: flex; align-items: center; gap: 12px; }}

  .logo-icon {{
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, #534AB7, #8B5CF6);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 20px rgba(83,74,183,0.5), 0 0 0 1px rgba(255,255,255,0.1);
  }}

  .logo-text {{ font-size: 18px; font-weight: 600; color: #fff; }}
  .logo-sub {{ font-size: 11px; color: rgba(255,255,255,0.4); margin-top: 1px; }}

  .topbar-right {{ display: flex; align-items: center; gap: 10px; }}

  .status-pill {{
    display: flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 20px;
    background: rgba(99,153,34,0.12);
    border: 1px solid rgba(99,153,34,0.25);
    font-size: 11px; color: rgba(150,220,80,0.9);
  }}
  .status-pill.offline {{
    background: rgba(226,75,74,0.12);
    border-color: rgba(226,75,74,0.25);
    color: rgba(255,120,120,0.9);
  }}
  .status-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}

  .badge-model {{
    padding: 6px 14px; border-radius: 20px;
    background: rgba(83,74,183,0.2);
    border: 1px solid rgba(83,74,183,0.35);
    font-size: 11px; color: rgba(180,170,255,0.9);
  }}

  /* ── Stats row ── */
  .stats-row {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px;
  }}

  .stat-card {{
    padding: 18px 20px;
    position: relative; overflow: hidden;
    transform-style: preserve-3d;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    cursor: default;
  }}

  .stat-card::before {{
    content: '';
    position: absolute; inset: 0; border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, transparent 60%);
    pointer-events: none;
  }}

  .stat-card:hover {{
    transform: translateY(-4px) rotateX(2deg);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.1);
  }}

  .stat-label {{
    font-size: 10px; font-weight: 500; color: rgba(255,255,255,0.4);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
  }}

  .stat-value {{
    font-size: 28px; font-weight: 600;
    background: linear-gradient(135deg, #a78bfa, #c4b5fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }}

  /* ── Main grid ── */
  .main-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;
  }}

  /* ── Input panel ── */
  .panel {{ padding: 24px; }}
  .panel-title {{
    font-size: 11px; font-weight: 500; color: rgba(255,255,255,0.4);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 14px;
  }}

  textarea#sentiment-input {{
    width: 100%; min-height: 130px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 14px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 14px; color: #e2e0f0;
    resize: vertical; outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    line-height: 1.6;
  }}
  textarea#sentiment-input::placeholder {{ color: rgba(255,255,255,0.2); }}
  textarea#sentiment-input:focus {{
    border-color: rgba(139,92,246,0.5);
    box-shadow: 0 0 0 3px rgba(139,92,246,0.12);
  }}

  .analyze-btn {{
    margin-top: 14px; width: 100%;
    padding: 12px; border: none; border-radius: 12px;
    background: linear-gradient(135deg, #534AB7 0%, #7C3AED 100%);
    color: #fff; font-family: 'Inter', sans-serif;
    font-size: 14px; font-weight: 500; cursor: pointer;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(83,74,183,0.4);
    transition: transform 0.15s, box-shadow 0.15s;
  }}
  .analyze-btn::before {{
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, transparent 60%);
  }}
  .analyze-btn:hover {{
    transform: translateY(-1px);
    box-shadow: 0 8px 28px rgba(83,74,183,0.55);
  }}
  .analyze-btn:active {{ transform: translateY(0); }}

  /* ── Result panel ── */
  .result-empty {{
    height: 130px; display: flex; align-items: center; justify-content: center;
    border: 1px dashed rgba(255,255,255,0.1); border-radius: 14px;
    color: rgba(255,255,255,0.2); font-size: 13px;
  }}

  .result-content {{}}

  .result-header {{
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 20px;
  }}

  .label-badge {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 16px; border-radius: 24px;
    font-size: 14px; font-weight: 500;
  }}
  .label-badge.positive {{
    background: rgba(99,153,34,0.15);
    border: 1px solid rgba(99,153,34,0.3);
    color: #a3e060;
  }}
  .label-badge.negative {{
    background: rgba(226,75,74,0.15);
    border: 1px solid rgba(226,75,74,0.3);
    color: #ff9090;
  }}

  .conf-block {{ text-align: right; }}
  .conf-label {{ font-size: 10px; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
  .conf-value {{
    font-size: 40px; font-weight: 600; line-height: 1;
    background: linear-gradient(135deg, #a78bfa, #c4b5fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }}

  .bar-section {{ margin-bottom: 8px; }}
  .bar-meta {{
    display: flex; justify-content: space-between;
    font-size: 11px; color: rgba(255,255,255,0.35);
    margin-bottom: 5px;
  }}
  .bar-track {{
    height: 6px; border-radius: 3px;
    background: rgba(255,255,255,0.07); overflow: hidden;
  }}
  .bar-fill {{
    height: 100%; border-radius: 3px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
  }}
  .bar-fill.pos {{ background: linear-gradient(90deg, #639922, #a3e060); }}
  .bar-fill.neg {{ background: linear-gradient(90deg, #991b1b, #E24B4A); }}

  .result-footer {{
    margin-top: 16px; padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.07);
    display: flex; gap: 10px; flex-wrap: wrap;
  }}
  .meta-chip {{
    padding: 4px 10px; border-radius: 8px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 11px; color: rgba(255,255,255,0.4);
  }}

  /* ── Status panel ── */
  .status-grid {{
    display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-bottom: 16px;
  }}
  .status-item {{
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px; border-radius: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    font-size: 12px; color: rgba(255,255,255,0.5);
  }}
  .sdot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
  .sdot.on {{ background: #639922; box-shadow: 0 0 6px rgba(99,153,34,0.6); }}
  .sdot.off {{ background: #E24B4A; box-shadow: 0 0 6px rgba(226,75,74,0.6); }}

  /* ── History table ── */
  .history-panel {{ padding: 24px; }}

  table.htable {{
    width: 100%; border-collapse: collapse; font-size: 12px;
  }}
  .htable th {{
    text-align: left; padding: 8px 12px;
    font-size: 10px; font-weight: 500; color: rgba(255,255,255,0.3);
    text-transform: uppercase; letter-spacing: 0.06em;
    border-bottom: 1px solid rgba(255,255,255,0.07);
  }}
  .htable td {{
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.65);
    vertical-align: middle;
  }}
  .htable tr:last-child td {{ border-bottom: none; }}
  .htable tr:hover td {{ background: rgba(255,255,255,0.03); }}

  .time-cell {{ color: rgba(255,255,255,0.3) !important; font-size: 11px !important; }}
  .text-cell {{ max-width: 280px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .conf-cell {{ color: #a78bfa !important; font-weight: 500; }}

  .mini-badge {{
    display: inline-block; padding: 3px 10px; border-radius: 10px;
    font-size: 11px; font-weight: 500;
  }}
  .mini-badge.pos {{ background: rgba(99,153,34,0.15); color: #a3e060; }}
  .mini-badge.neg {{ background: rgba(226,75,74,0.15); color: #ff9090; }}

  .cache-hit {{ color: rgba(167,139,250,0.8); font-size: 11px; }}
  .empty-row {{ text-align: center; color: rgba(255,255,255,0.2) !important; padding: 24px !important; }}

  /* ── Loading spinner ── */
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .spinner {{
    width: 16px; height: 16px; border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.2);
    border-top-color: #a78bfa;
    animation: spin 0.7s linear infinite;
    display: inline-block; vertical-align: middle; margin-right: 8px;
  }}
</style>
</head>
<body>
<div class="bg"></div>
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

<div class="app">

  <!-- Topbar -->
  <div class="topbar glass">
    <div class="logo">
      <div class="logo-icon">🧠</div>
      <div>
        <div class="logo-text">NuanceIQ</div>
        <div class="logo-sub">Real-Time Sentiment Intelligence</div>
      </div>
    </div>
    <div class="topbar-right">
      <div class="status-pill {'offline' if api_status == 'offline' else ''}">
        <div class="status-dot"></div>
        API {api_status}
      </div>
      <div class="badge-model">DistilBERT · IMDB 55K</div>
    </div>
  </div>

  <!-- Stats row -->
  <div class="stats-row">
    <div class="stat-card glass">
      <div class="stat-label">Accuracy</div>
      <div class="stat-value">{accuracy}</div>
    </div>
    <div class="stat-card glass">
      <div class="stat-label">F1 Score</div>
      <div class="stat-value">{f1}</div>
    </div>
    <div class="stat-card glass">
      <div class="stat-label">Precision</div>
      <div class="stat-value">{precision}</div>
    </div>
    <div class="stat-card glass">
      <div class="stat-label">Recall</div>
      <div class="stat-value">{recall}</div>
    </div>
  </div>

  <!-- Main grid -->
  <div class="main-grid">

    <!-- Input panel -->
    <div class="glass panel">
      <div class="panel-title">Text input</div>
      <textarea id="sentiment-input" placeholder="Paste any text — a review, tweet, comment, or sentence..."></textarea>
      <button class="analyze-btn" onclick="analyze()">Analyze sentiment</button>
    </div>

    <!-- Result panel -->
    <div class="glass panel" id="result-panel">
      <div class="panel-title">Result</div>
      <div class="result-empty" id="result-empty">Run an analysis to see results here</div>
      <div class="result-content" id="result-content" style="display:none">
        <div class="result-header">
          <div>
            <div style="font-size:10px;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">Sentiment</div>
            <div class="label-badge" id="label-badge"></div>
          </div>
          <div class="conf-block">
            <div class="conf-label">Confidence</div>
            <div class="conf-value" id="conf-value"></div>
          </div>
        </div>
        <div class="bar-section">
          <div class="bar-meta"><span>Positive</span><span id="pos-pct"></span></div>
          <div class="bar-track"><div class="bar-fill pos" id="pos-bar" style="width:0%"></div></div>
        </div>
        <div class="bar-section" style="margin-top:10px">
          <div class="bar-meta"><span>Negative</span><span id="neg-pct"></span></div>
          <div class="bar-track"><div class="bar-fill neg" id="neg-bar" style="width:0%"></div></div>
        </div>
        <div class="result-footer">
          <div class="meta-chip" id="time-chip"></div>
          <div class="meta-chip" id="cache-chip"></div>
          <div class="meta-chip" id="proc-chip"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- System status + history -->
  <div class="glass" style="padding:24px;margin-bottom:20px">
    <div class="panel-title" style="margin-bottom:14px">System status</div>
    <div class="status-grid">
      <div class="status-item">
        <div class="sdot {'on' if api_status == 'online' else 'off'}"></div>
        API {api_status}
      </div>
      <div class="status-item">
        <div class="sdot {'on' if model_status == 'loaded' else 'off'}"></div>
        Model {model_status}
      </div>
      <div class="status-item">
        <div class="sdot {'on' if redis_status == 'connected' else 'off'}"></div>
        Redis {redis_status}
      </div>
    </div>
  </div>

  <!-- History -->
  <div class="glass history-panel">
    <div class="panel-title" style="margin-bottom:14px">Recent predictions</div>
    <table class="htable">
      <thead>
        <tr>
          <th>Time</th>
          <th>Text</th>
          <th>Label</th>
          <th>Confidence</th>
          <th>Cache</th>
        </tr>
      </thead>
      <tbody id="history-body">
        {history_rows}
      </tbody>
    </table>
  </div>

</div>

<script>
const API_BASE = "{API_BASE}";

async function analyze() {{
  const text = document.getElementById('sentiment-input').value.trim();
  if (!text) {{ alert('Enter some text first.'); return; }}

  const btn = document.querySelector('.analyze-btn');
  btn.innerHTML = '<span class="spinner"></span>Analyzing...';
  btn.disabled = true;

  try {{
    const res = await fetch(API_BASE + '/predict', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ text }})
    }});

    if (!res.ok) throw new Error('API error ' + res.status);
    const d = await res.json();

    const isPos = d.label === 'POSITIVE';
    const badge = document.getElementById('label-badge');
    badge.className = 'label-badge ' + (isPos ? 'positive' : 'negative');
    badge.innerHTML = (isPos ? '😊' : '😞') + ' ' + (isPos ? 'Positive' : 'Negative');

    document.getElementById('conf-value').textContent = (d.confidence * 100).toFixed(1) + '%';
    document.getElementById('pos-pct').textContent = (d.positive_score * 100).toFixed(1) + '%';
    document.getElementById('neg-pct').textContent = (d.negative_score * 100).toFixed(1) + '%';
    document.getElementById('pos-bar').style.width = (d.positive_score * 100).toFixed(1) + '%';
    document.getElementById('neg-bar').style.width = (d.negative_score * 100).toFixed(1) + '%';

    const now = new Date().toLocaleTimeString('en-IN', {{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
    document.getElementById('time-chip').textContent = now;
    document.getElementById('cache-chip').textContent = d.cached ? '⚡ Cache hit' : 'Fresh inference';
    document.getElementById('proc-chip').textContent = (d.processing_time_ms || 0).toFixed(1) + 'ms';

    document.getElementById('result-empty').style.display = 'none';
    document.getElementById('result-content').style.display = 'block';

    window.parent.postMessage({{
      type: 'nuanceiq_result',
      label: d.label,
      confidence: (d.confidence * 100).toFixed(1) + '%',
      text: text.slice(0, 70) + (text.length > 70 ? '…' : ''),
      time: now,
      cached: d.cached ? 'Hit' : '—'
    }}, '*');

  }} catch(e) {{
    alert('Error: ' + e.message + '\\nMake sure the API is running.');
  }} finally {{
    btn.innerHTML = 'Analyze sentiment';
    btn.disabled = false;
  }}
}}

document.getElementById('sentiment-input').addEventListener('keydown', function(e) {{
  if (e.ctrlKey && e.key === 'Enter') analyze();
}});
</script>
</body>
</html>
"""

components.html(HTML, height=1050, scrolling=True)

# Listen for result from iframe via query params workaround
st.markdown("""
<script>
window.addEventListener('message', function(e) {
  if (e.data && e.data.type === 'nuanceiq_result') {
    const params = new URLSearchParams(window.location.search);
    // History is managed server-side on next rerun
  }
});
</script>
""", unsafe_allow_html=True)