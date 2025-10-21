# web_dashboard.py

from flask import Flask, render_template_string, jsonify
import json
from datetime import datetime
from database_manager import get_dashboard_data, setup_database

app = Flask(__name__)

# --- Jinja Filters ---
def from_json(value):
    """Custom Jinja filter to safely load JSON strings."""
    try:
        return json.loads(value)
    except:
        return {}

def to_pretty_json(value):
    """Custom Jinja filter to pretty-print JSON."""
    return json.dumps(value, indent=2)

def format_sentiment(score):
    if score is None: return "N/A"
    if score > 0.3: return f"😊 Bullish ({score:.2f})"
    if score < -0.3: return f"😞 Bearish ({score:.2f})"
    return f"😐 Neutral ({score:.2f})"

app.jinja_env.filters['from_json'] = from_json
app.jinja_env.filters['to_pretty_json'] = to_pretty_json
app.jinja_env.filters['format_sentiment'] = format_sentiment


# --- Complex HTML Template ---
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Trading Bot Dashboard</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background-color: #f0f2f5; color: #333; }
        .container { max-width: 1600px; margin: 20px auto; padding: 0 15px; }
        h1, h2 { color: #333; }
        h1 { border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { margin-top: 30px; border-left: 4px solid #007bff; padding-left: 10px; font-size: 1.5em; }
        .card { background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 25px; }
        .grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }
        th, td { padding: 12px 10px; text-align: left; border-bottom: 1px solid #dee2e6; vertical-align: top; }
        th { background-color: #f8f9fa; font-weight: 600; color: #495057; }
        .status-active, .side-long, .pnl-positive { color: #28a745; font-weight: bold; }
        .status-flat { color: #6c757d; font-weight: bold; }
        .side-short, .pnl-negative { color: #dc3545; font-weight: bold; }
        .pnl-zero { color: #6c757d; }
        .mono-box { background-color: #e9ecef; padding: 10px; border-radius: 4px; white-space: pre-wrap; font-family: monospace; font-size: 0.8em; max-height: 200px; overflow-y: auto; }
        details { cursor: pointer; }
        summary { font-weight: bold; color: #007bff; margin-bottom: 5px; }
        .log-box { max-height: 400px; overflow-y: scroll; background-color: #f8f9fa; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 0.8em; }
        .log-entry { padding: 4px 0; border-bottom: 1px dotted #ccc; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Trading Bot Dashboard</h1>
        <p style="color: #6c757d;">Last Updated: {{ last_updated }} (Auto-refresh in 10s)</p>
        
        <!-- Account Vitals -->
        <div class="card">
            <h2>Account Vitals</h2>
            <div class="grid-container">
                {# CRITICAL FIX: Use .get() to provide default values if data.vitals is empty #}
                <div><strong>Total Equity:</strong> <span class="pnl-positive">{{ "{:,.2f}".format(data.vitals.get('total_equity', 0)) }} USDT</span></div>
                <div><strong>Available Margin:</strong> {{ "{:,.2f}".format(data.vitals.get('available_margin', 0)) }} USDT</div>
                {% set ts = data.vitals.get('timestamp') %}
                <div><strong>Last Update:</strong> {{ ts.split('T')[1].split('.')[0] if ts else 'N/A' }} UTC</div>
            </div>
        </div>
        
        <!-- Bot State -->
        <div class="card">
            <h2>Symbol States</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 8%;">Symbol</th>
                        <th style="width: 10%;">Status & Side</th>
                        <th style="width: 15%;">Position Details</th>
                        <th style="width: 17%;">Market & Analysis</th>
                        <th style="width: 20%;">Active Triggers</th>
                        <th style="width: 30%;">Last AI Response</th>
                    </tr>
                </thead>
                <tbody>
                    {% for state in data.state %}
                    <tr>
                        <td><strong>{{ state.symbol }}</strong></td>
                        <td>
                            <div class="status-{{ 'active' if state.is_in_position else 'flat' }}">{{ 'ACTIVE' if state.is_in_position else 'FLAT' }}</div>
                            {% if state.side %}
                            <div class="side-{{ state.side | lower }}">{{ state.side }}</div>
                            {% endif %}
                        </td>
                        <td>
                            {% if state.is_in_position %}
                                <div><strong>Entry:</strong> {{ "{:,.4f}".format(state.entry_price) }}</div>
                                <div><strong>Qty:</strong> {{ "{:,.4f}".format(state.quantity) }}</div>
                                <div><strong>PNL:</strong> 
                                    <span class="pnl-{{ 'positive' if state.unrealized_pnl > 0 else 'negative' if state.unrealized_pnl < 0 else 'zero' }}">
                                        {{ "{:,.2f}".format(state.unrealized_pnl) }} USDT
                                    </span>
                                </div>
                            {% else %}
                                -
                            {% endif %}
                        </td>
                        <td>
                            <div><strong>Market Price:</strong> {{ "{:,.4f}".format(state.last_known_price) }}</div>
                            <div><strong>Sentiment:</strong> {{ state.last_sentiment_score | format_sentiment }}</div>
                            <div>
                                <strong>Context:</strong>
                                <div class="mono-box" style="font-size: 0.7em; max-height: 80px;">{{ state.market_context | from_json | to_pretty_json }}</div>
                            </div>
                        </td>
                        <td>
                            <div class="mono-box">{{ state.active_triggers | from_json | to_pretty_json }}</div>
                        </td>
                        <td>
                            <details>
                                <summary>Show/Hide Full Response</summary>
                                <div class="mono-box">{{ state.last_ai_response }}</div>
                            </details>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Recent Trades & Logs -->
        <div class="grid-container">
            <div class="card">
                <h2>Recent Trades</h2>
                <div style="max-height: 400px; overflow-y: auto;">
                    <table>
                        <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>PNL (USDT)</th><th>Reasoning</th></tr></thead>
                        <tbody>
                        {% for trade in data.trades %}
                            <tr>
                                <td>{{ trade.timestamp.split('T')[1].split('.')[0] }}</td>
                                <td>{{ trade.symbol }}</td>
                                <td class="side-{{ trade.side | lower }}">{{ trade.side }}</td>
                                <td class="pnl-{{ 'positive' if trade.pnl > 0 else 'negative' }}">{{ "{:,.2f}".format(trade.pnl) }}</td>
                                <td>{{ trade.reasoning | truncate(30) }}</td>
                            </tr>
                        {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="card">
                <h2>Recent Logs</h2>
                <div class="log-box">
                    {% for log in data.logs %}
                    <div class="log-entry">
                        <span style="color: #999;">[{{ log.timestamp.split('T')[1].split('.')[0] }}] [{{ log.symbol }}]</span> {{ log.message }}
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Renders the main dashboard page."""
    data = get_dashboard_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template_string(DASHBOARD_TEMPLATE, data=data, last_updated=now)

@app.route('/api/data')
def api_data():
    """API endpoint for fetching raw data."""
    return jsonify(get_dashboard_data())

def start_dashboard():
    """Function to start the Flask server."""
    setup_database() 
    print("\n🌐 Starting Web Dashboard at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)