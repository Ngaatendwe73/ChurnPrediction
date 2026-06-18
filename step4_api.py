"""
BANK CHURN PREDICTION PROJECT
Zimbabwe Banking/MicroFinance — Industrial Attachment

Step 4: Flask REST API & Live Dashboard
========================================
Reads:  models/churn_model_bundle.pkl

Run:    python step4_api.py
Open:   http://127.0.0.1:5000
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, 'models', 'churn_model_bundle.pkl')

# ── Load model bundle ─────────────────────────────────────────────────────────
bundle    = joblib.load(BUNDLE_PATH)
model     = bundle['model']
scaler    = bundle['scaler']
FEATURES  = bundle['features']
encoders  = bundle['encoders']
MODEL_NAME = bundle['best_model_name']

# ── Risk band helper ──────────────────────────────────────────────────────────
def get_risk_band(prob: float) -> dict:
    if prob < 0.30:
        return {'band': 'Low',      'color': '#1D9E75',
                'action': 'No immediate action needed.'}
    elif prob < 0.50:
        return {'band': 'Medium',   'color': '#BA7517',
                'action': 'Schedule a retention check-in call.'}
    elif prob < 0.70:
        return {'band': 'High',     'color': '#D85A30',
                'action': 'Priority outreach — offer loyalty incentive.'}
    else:
        return {'band': 'Critical', 'color': '#A32D2D',
                'action': 'Immediate intervention required.'}


# ── Feature engineering (mirrors step2_eda.py) ────────────────────────────────
def build_features(data: dict) -> tuple:
    balance  = float(data.get('Balance', 0))
    salary   = float(data.get('EstimatedSalary', 1))
    active   = int(data.get('IsActiveMember', 0))
    products = int(data.get('NumOfProducts', 1))
    has_cr   = int(data.get('HasCrCard', 0))
    tenure   = int(data.get('Tenure', 0))
    currency = data.get('Account_Currency', 'USD')

    # RFM
    recency  = 1 if active else 3
    freq     = min(products, 4)
    if balance <= 0:         mon = 1
    elif balance <= 50000:   mon = 2
    elif balance <= 100000:  mon = 3
    elif balance <= 200000:  mon = 4
    else:                    mon = 5
    rfm = round(recency * 0.40 + freq * 0.35 + mon * 0.25, 3)

    has_zero    = 1 if balance == 0 else 0
    bal_sal     = round(balance / (salary + 1), 4)
    mobile_risk = (1 - active) + has_zero + (1 if products == 1 else 0)
    digital     = min(active * 2 + has_cr + (1 if products >= 2 else 0), 4)
    loyalty     = round(tenure * 1.0 + products * 0.5 + active * 2.0, 2)
    infl_f      = 5.0 if currency == 'ZWL' else 1.05
    real_bal    = round(balance / infl_f, 2)

    computed = {
        'CreditScore':                  float(data.get('CreditScore', 600)),
        'Age':                          float(data.get('Age', 35)),
        'Tenure':                       float(tenure),
        'Balance':                      float(balance),
        'NumOfProducts':                float(products),
        'HasCrCard':                    float(has_cr),
        'IsActiveMember':               float(active),
        'EstimatedSalary':              float(salary),
        'RFM_Score':                    rfm,
        'Has_Zero_Balance':             float(has_zero),
        'Balance_to_Salary_Ratio':      bal_sal,
        'Mobile_Money_Risk':            float(mobile_risk),
        'Digital_Adoption_Score':       float(digital),
        'Loyalty_Score':                loyalty,
        'Real_Balance_After_Inflation': real_bal,
        'Geography': float(encoders['Geography'].transform(
            [data.get('Geography', 'France')])[0]),
        'Gender': float(encoders['Gender'].transform(
            [data.get('Gender', 'Male')])[0]),
        'Account_Currency': float(encoders['Account_Currency'].transform(
            [currency])[0]),
    }

    X = np.array([[computed[f] for f in FEATURES]])
    return scaler.transform(X), computed


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════════════════════
DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Churn Risk Dashboard — ZimBank</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f0f4f8;
    color: #1a1a2e;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: linear-gradient(135deg, #0f4c81, #1a6fb5);
    color: #fff;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
  }
  header .logo { font-size: 2rem; }
  header h1   { font-size: 1.2rem; font-weight: 700; }
  header p    { font-size: 0.78rem; opacity: 0.75; margin-top: 2px; }
  #clock      { margin-left: auto; font-size: 0.82rem; opacity: 0.85;
                font-family: monospace; }

  /* ── Layout ── */
  .page    { max-width: 1140px; margin: 28px auto; padding: 0 18px; }
  .grid    { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }

  /* ── Cards ── */
  .card {
    background: #fff;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
  }
  .card h2 {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f4c81;
    border-bottom: 2px solid #e4eef8;
    padding-bottom: 8px;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }

  /* ── Form ── */
  .row2   { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  label   { display: block; font-size: 0.78rem; color: #555;
            font-weight: 600; margin-bottom: 4px; margin-top: 12px; }
  input, select {
    width: 100%; padding: 8px 10px;
    border: 1px solid #cdd5df; border-radius: 7px;
    font-size: 0.88rem; color: #1a1a2e;
    transition: border-color .2s;
  }
  input:focus, select:focus {
    outline: none; border-color: #0f4c81;
    box-shadow: 0 0 0 3px rgba(15,76,129,0.12);
  }
  .btn {
    width: 100%; margin-top: 20px; padding: 12px;
    background: linear-gradient(135deg, #0f4c81, #1a6fb5);
    color: #fff; border: none; border-radius: 8px;
    font-size: 1rem; font-weight: 700; cursor: pointer;
    letter-spacing: 0.3px; transition: opacity .2s;
  }
  .btn:hover { opacity: 0.88; }
  .btn:active { opacity: 0.75; }

  /* ── Result panel ── */
  #result { display: none; }

  .prob-wrap {
    display: flex; flex-direction: column;
    align-items: center; margin-bottom: 18px;
  }
  .prob-ring {
    width: 110px; height: 110px; border-radius: 50%;
    border: 6px solid #ccc;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    margin-bottom: 12px;
    transition: border-color .4s;
  }
  .prob-num   { font-size: 1.7rem; font-weight: 800; }
  .prob-label { font-size: 0.7rem; opacity: 0.75; margin-top: 1px; }

  .badge {
    display: inline-block; padding: 5px 18px;
    border-radius: 20px; color: #fff;
    font-size: 0.82rem; font-weight: 700;
    margin-bottom: 14px;
  }
  .action-box {
    background: #f4f7fb; border-left: 4px solid #0f4c81;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px; font-size: 0.84rem;
    color: #333; line-height: 1.55; width: 100%;
    text-align: left;
  }

  /* ── Feature table ── */
  .feat-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.8rem; margin-top: 6px;
  }
  .feat-table th {
    text-align: left; color: #888; font-weight: 600;
    padding: 5px 4px; border-bottom: 1px solid #eee;
    font-size: 0.75rem; text-transform: uppercase;
  }
  .feat-table td { padding: 5px 4px; border-bottom: 1px solid #f2f2f2; }
  .feat-table tr:last-child td { border: none; }
  .zim-label { color: #D85A30; font-weight: 600; }
  .bar-wrap  { background: #eee; border-radius: 4px; height: 5px; }
  .bar-fill  { height: 5px; border-radius: 4px; transition: width .5s; }

  /* ── History table ── */
  #history-section { margin-top: 20px; }
  .hist-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  .hist-table th {
    background: #0f4c81; color: #fff;
    padding: 8px 10px; text-align: left; font-size: 0.78rem;
  }
  .hist-table td { padding: 7px 10px; border-bottom: 1px solid #eee; }
  .hist-table tr:hover td { background: #f0f6ff; }
  .dot {
    display: inline-block; width: 9px; height: 9px;
    border-radius: 50%; margin-right: 5px; vertical-align: middle;
  }

  /* ── API docs ── */
  .api-block {
    background: #1e2d3d; color: #a8d8a8;
    border-radius: 8px; padding: 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.76rem; margin-top: 10px;
    overflow-x: auto; white-space: pre; line-height: 1.6;
  }
</style>
</head>
<body>

<header>
  <div class="logo">🏦</div>
  <div>
    <h1>Customer Churn Risk Dashboard</h1>
    <p>Zimbabwe Banking &amp; MicroFinance &nbsp;·&nbsp;
       ML Model: {{ model_name }} &nbsp;·&nbsp; 18 features</p>
  </div>
  <span id="clock"></span>
</header>

<div class="page">
  <div class="grid">

    <!-- ── INPUT FORM ── -->
    <div class="card">
      <h2>📋 Customer Profile</h2>

      <div class="row2">
        <div>
          <label>Credit Score</label>
          <input id="CreditScore" type="number" value="620" min="300" max="850">
        </div>
        <div>
          <label>Age</label>
          <input id="Age" type="number" value="38" min="18" max="90">
        </div>
      </div>

      <div class="row2">
        <div>
          <label>Tenure (years with bank)</label>
          <input id="Tenure" type="number" value="3" min="0" max="15">
        </div>
        <div>
          <label>Account Balance (USD)</label>
          <input id="Balance" type="number" value="45000" min="0">
        </div>
      </div>

      <div class="row2">
        <div>
          <label>Number of Products</label>
          <select id="NumOfProducts">
            <option value="1">1</option>
            <option value="2" selected>2</option>
            <option value="3">3</option>
            <option value="4">4</option>
          </select>
        </div>
        <div>
          <label>Estimated Annual Salary</label>
          <input id="EstimatedSalary" type="number" value="35000" min="0">
        </div>
      </div>

      <div class="row2">
        <div>
          <label>Has Credit Card</label>
          <select id="HasCrCard">
            <option value="1">Yes</option>
            <option value="0">No</option>
          </select>
        </div>
        <div>
          <label>Active Member</label>
          <select id="IsActiveMember">
            <option value="1" selected>Yes</option>
            <option value="0">No</option>
          </select>
        </div>
      </div>

      <div class="row2">
        <div>
          <label>Region</label>
          <select id="Geography">
            <option value="France">Harare / Bulawayo CBD</option>
            <option value="Germany">Secondary City</option>
            <option value="Spain">Rural / Growth Point</option>
          </select>
        </div>
        <div>
          <label>Gender</label>
          <select id="Gender">
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
        </div>
      </div>

      <div class="row2">
        <div>
          <label>Account Currency (Zimbabwe)</label>
          <select id="Account_Currency">
            <option value="USD" selected>USD</option>
            <option value="ZWL">ZWL (Zimbabwe Dollar)</option>
          </select>
        </div>
      </div>

      <button class="btn" onclick="predict()">
        ⚡ Predict Churn Risk
      </button>
    </div>

    <!-- ── RESULT PANEL ── -->
    <div class="card" id="result">
      <h2>📊 Prediction Result</h2>

      <div class="prob-wrap">
        <div class="prob-ring" id="prob-ring">
          <span class="prob-num"  id="prob-num">--</span>
          <span class="prob-label">churn probability</span>
        </div>
        <span class="badge" id="risk-badge">--</span>
        <div class="action-box" id="action-box">--</div>
      </div>

      <h2 style="margin-top:14px">🇿🇼 Zimbabwe Feature Breakdown</h2>
      <table class="feat-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Value</th>
            <th style="width:80px">Signal</th>
          </tr>
        </thead>
        <tbody id="feat-body"></tbody>
      </table>
    </div>

  </div><!-- /grid -->

  <!-- ── RECENT PREDICTIONS ── -->
  <div class="card" id="history-section" style="display:none; margin-top:20px">
    <h2>🕐 Recent Predictions (this session)</h2>
    <table class="hist-table">
      <thead>
        <tr>
          <th>#</th><th>Age</th><th>Balance</th><th>Currency</th>
          <th>Active</th><th>Mobile Risk</th>
          <th>Churn Prob</th><th>Band</th>
        </tr>
      </thead>
      <tbody id="hist-body"></tbody>
    </table>
  </div>

  <!-- ── API DOCS ── -->
  <div class="card" style="margin-top:20px">
    <h2>🔌 REST API Reference</h2>
    <p style="font-size:0.83rem;color:#555;margin-bottom:8px">
      Send a <strong>POST</strong> request to
      <code style="background:#eef2f7;padding:2px 7px;border-radius:4px;
                   font-size:0.82rem">/predict</code>
      with a JSON body. Returns churn probability, risk band, and action.
    </p>
    <div class="api-block">POST /predict   HTTP/1.1
Content-Type: application/json

{
  "CreditScore": 620,   "Age": 38,         "Tenure": 3,
  "Balance": 45000,     "NumOfProducts": 2, "HasCrCard": 1,
  "IsActiveMember": 1,  "EstimatedSalary": 35000,
  "Geography": "France","Gender": "Male",
  "Account_Currency": "USD"
}

── Response ──────────────────────────────────────────────
{
  "churn_probability": 0.3352,
  "risk_band": "Medium",
  "action": "Schedule a retention check-in call.",
  "features_used": { ... all 18 computed features ... }
}</div>

    <p style="font-size:0.83rem;color:#555;margin:14px 0 8px">
      <strong>Batch endpoint</strong> — score many customers at once,
      sorted highest risk first:
    </p>
    <div class="api-block">POST /predict/batch   HTTP/1.1
Content-Type: application/json

[ { "customer_id": "ZB001", ... }, { "customer_id": "ZB002", ... } ]

── Response ──────────────────────────────────────────────
{
  "total": 2,
  "predictions": [
    { "customer_id": "ZB002", "churn_probability": 0.54, "risk_band": "High" },
    { "customer_id": "ZB001", "churn_probability": 0.26, "risk_band": "Low"  }
  ]
}</div>
  </div>

</div><!-- /page -->

<script>
  // ── Live clock (Harare time) ──────────────────────────────────────────────
  function tick() {
    document.getElementById('clock').textContent =
      new Date().toLocaleString('en-ZW', { timeZone: 'Africa/Harare' });
  }
  tick(); setInterval(tick, 1000);

  // ── Session history ───────────────────────────────────────────────────────
  const history = [];
  let   counter = 1;

  // ── Predict ───────────────────────────────────────────────────────────────
  async function predict() {
    const btn = document.querySelector('.btn');
    btn.textContent = '⏳ Predicting...';
    btn.disabled = true;

    const payload = {
      CreditScore:     +document.getElementById('CreditScore').value,
      Age:             +document.getElementById('Age').value,
      Tenure:          +document.getElementById('Tenure').value,
      Balance:         +document.getElementById('Balance').value,
      NumOfProducts:   +document.getElementById('NumOfProducts').value,
      HasCrCard:       +document.getElementById('HasCrCard').value,
      IsActiveMember:  +document.getElementById('IsActiveMember').value,
      EstimatedSalary: +document.getElementById('EstimatedSalary').value,
      Geography:        document.getElementById('Geography').value,
      Gender:           document.getElementById('Gender').value,
      Account_Currency: document.getElementById('Account_Currency').value,
    };

    try {
      const res  = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.error) { alert('Error: ' + data.error); return; }
      renderResult(data, payload);
    } catch (e) {
      alert('Could not connect to API: ' + e);
    } finally {
      btn.textContent = '⚡ Predict Churn Risk';
      btn.disabled = false;
    }
  }

  // ── Render result ─────────────────────────────────────────────────────────
  function renderResult(data, payload) {
    const pct   = Math.round(data.churn_probability * 100);
    const color = data.risk_color;
    const fv    = data.features_used;

    // Show panel
    document.getElementById('result').style.display = 'block';

    // Probability ring
    const ring = document.getElementById('prob-ring');
    ring.style.borderColor = color;
    ring.style.color       = color;
    document.getElementById('prob-num').textContent = pct + '%';

    // Badge + action
    const badge = document.getElementById('risk-badge');
    badge.textContent      = data.risk_band + ' Risk';
    badge.style.background = color;
    document.getElementById('action-box').textContent = '💡 ' + data.action;

    // Feature breakdown
    const zimKeys = ['Mobile_Money', 'Digital', 'RFM', 'Account', 'Real_Balance', 'Loyalty'];
    const feats = [
      { k: 'Mobile_Money_Risk',            label: 'Mobile money risk',           max: 3      },
      { k: 'Digital_Adoption_Score',        label: 'Digital adoption score',      max: 4      },
      { k: 'RFM_Score',                     label: 'RFM score',                   max: 4      },
      { k: 'Has_Zero_Balance',              label: 'Zero balance flag',           max: 1      },
      { k: 'Loyalty_Score',                 label: 'Loyalty score',               max: 12     },
      { k: 'Real_Balance_After_Inflation',  label: 'Real balance (inflation adj)',max: 250000 },
      { k: 'Age',                           label: 'Age',                         max: 90     },
      { k: 'Balance',                       label: 'Balance (USD)',               max: 250000 },
      { k: 'Tenure',                        label: 'Tenure (years)',              max: 10     },
      { k: 'CreditScore',                   label: 'Credit score',                max: 850    },
    ];

    const tbody = document.getElementById('feat-body');
    tbody.innerHTML = '';
    feats.forEach(f => {
      const val  = fv[f.k] !== undefined ? fv[f.k] : '—';
      const pct2 = f.max ? Math.min(100, Math.round(val / f.max * 100)) : 0;
      const isZim = zimKeys.some(z => f.k.includes(z));
      const barColor = isZim ? '#D85A30' : '#0f4c81';
      tbody.innerHTML += `
        <tr>
          <td class="${isZim ? 'zim-label' : ''}">
            ${f.label}${isZim ? ' ★' : ''}
          </td>
          <td><strong>${typeof val === 'number' ? val.toLocaleString() : val}</strong></td>
          <td>
            <div class="bar-wrap">
              <div class="bar-fill"
                   style="width:${pct2}%;background:${barColor}"></div>
            </div>
          </td>
        </tr>`;
    });

    // History
    history.unshift({
      n: counter++, ...payload,
      mobile_risk: fv.Mobile_Money_Risk,
      prob: pct, band: data.risk_band, color,
    });
    document.getElementById('history-section').style.display = 'block';
    document.getElementById('hist-body').innerHTML =
      history.slice(0, 10).map(h => `
        <tr>
          <td>${h.n}</td>
          <td>${h.Age}</td>
          <td>$${Number(h.Balance).toLocaleString()}</td>
          <td>${h.Account_Currency}</td>
          <td>${h.IsActiveMember ? 'Yes' : 'No'}</td>
          <td>${h.mobile_risk}/3</td>
          <td><strong>${h.prob}%</strong></td>
          <td>
            <span class="dot" style="background:${h.color}"></span>
            ${h.band}
          </td>
        </tr>`).join('');
  }
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)


@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD, model_name=MODEL_NAME)


@app.route('/health')
def health():
    return jsonify({
        'status':   'ok',
        'model':    MODEL_NAME,
        'features': len(FEATURES),
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data       = request.get_json(force=True)
        X, computed = build_features(data)
        prob       = float(model.predict_proba(X)[0][1])
        band       = get_risk_band(prob)

        return jsonify({
            'churn_probability': round(prob, 4),
            'risk_band':         band['band'],
            'risk_color':        band['color'],
            'action':            band['action'],
            'features_used':     {k: round(v, 4) if isinstance(v, float)
                                  else v for k, v in computed.items()},
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        customers = request.get_json(force=True)
        if not isinstance(customers, list):
            return jsonify({'error': 'Expected a JSON array'}), 400

        predictions = []
        for i, cust in enumerate(customers):
            X, _ = build_features(cust)
            prob  = float(model.predict_proba(X)[0][1])
            band  = get_risk_band(prob)
            predictions.append({
                'index':             i,
                'customer_id':       cust.get('customer_id', f'CUST_{i+1:04d}'),
                'churn_probability': round(prob, 4),
                'risk_band':         band['band'],
                'action':            band['action'],
            })

        predictions.sort(key=lambda x: x['churn_probability'], reverse=True)
        return jsonify({'total': len(predictions), 'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("=" * 55)
    print("  ZimBank Churn Prediction API — RUNNING")
    print("=" * 55)
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Features  : {len(FEATURES)}")
    print()
    print("  Dashboard : http://127.0.0.1:5000")
    print("  Health    : http://127.0.0.1:5000/health")
    print("  Predict   : POST http://127.0.0.1:5000/predict")
    print("  Batch     : POST http://127.0.0.1:5000/predict/batch")
    print("=" * 55)
    print("  Press CTRL+C to stop the server")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)