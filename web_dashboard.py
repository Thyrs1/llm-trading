# web_dashboard.py

from flask import Flask, render_template_string, jsonify
import json
import threading
import time
from datetime import datetime
from database_manager import get_dashboard_data, setup_database

app = Flask(__name__)

# --- Jinja Filters ---
def from_json(value):
    """Custom Jinja filter to safely load JSON strings."""
    try:
        return json.loads(value)
    except:
        return {"error": "Invalid JSON"}

def to_pretty_json(value):
    """Custom Jinja filter to pretty-print JSON."""
    return json.dumps(value, indent=2)

app.jinja_env.filters['from_json'] = from_json
app.jinja_env.filters['to_pretty_json'] = to_pretty_json

# --- Complex HTML Template ---
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Trading Bot Dashboard</title>
    <meta http-equiv="refresh" content="10"> <!-- Auto-refresh every 10 seconds -->
    <style>
        /* General Styles */
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f8f9fa; color: #343a40; }
        .container { max-width: 1400px; margin: 20px auto; padding: 0 15px; }
        h1 { color: #007bff; border-bottom: 3px solid #007bff; padding-bottom: 10px; margin-bottom: 20px; }
        h2 { color: #495057; margin-top: 30px; border-left: 5px solid #007bff; padding-left: 10px; }
        
        /* Card Layout */
        .card { background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); padding: 20px; margin-bottom: 20px; }
        .grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }

        /* Table Styles */
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e9ecef; }
        th { background-color: #007bff; color: white; font-weight: 600; text-transform: uppercase; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #e2f0ff; }

        /* Status & PNL Indicators */
        .status-active { color: #28a745; font-weight: bold; }
        .status-flat { color: #6c757d; font-weight: bold; }
        .pnl-positive { color: #28a745; font-weight: bold; }
        .pnl-negative { color: #dc3545; font-weight: bold; }
        .pnl-zero { color: #6c757d; }
        .side-long { color: #28a745; font-weight: bold; }
        .side-short { color: #dc3545; font-weight: bold; }

        /* Context Box */
        .context-box { 
            background-color: #f1f1f1; 
            padding: 10px; 
            border-radius: 4px; 
            white-space: pre-wrap; 
            font-family: monospace; 
            font-size: 0.75em; 
            max-height: 150px; 
            overflow-y: auto;
            border: 1px solid #ddd;
        }

        /* Log Box */
        .log-box {
            max-height: 300px; 
            overflow-y: scroll; 
            background-color: #f1f1f1; 
            padding: 15px; 
            border-radius: 4px; 
            border: 1px solid #ddd;
        }
        .log-entry { font-size: 0.85em; padding: 5px 0; border-bottom: 1px dotted #ccc; }
        .log-entry:last-child { border-bottom: none; }
        .log-timestamp { color: #999; margin-right: 10px; }
        .log-symbol { font-weight: bold; margin-right: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Trading Bot Dashboard</h1>
        <p style="color: #6c757d;">Last Updated: {{ last_updated }} (Auto-refresh in 10s)</p>

        <!-- Current Bot State & Positions -->
        <div class="card">
            <h2>Current Bot State & Positions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Status</th>
                        <th>Side</th>
                        <th>Entry Price</th>
                        <th>Quantity</th>
                        <th>Unrealized PNL</th>
                        <th>Last Analysis Time</th>
                        <th>Market Context</th>
                    </tr>
                </thead>
                <tbody>
                    {% for state in data.state %}
                    <tr>
                        <td>{{ state.symbol }}</td>
                        <td class="{{ 'status-active' if state.is_in_position else 'status-flat' }}">
                            {{ 'ACTIVE' if state.is_in_position else 'FLAT' }}
                        </td>
                        <td class="side-{{ state.side | lower }}">{{ state.side if state.side else '-' }}</td>
                        <td>{{ "{:,.4f}".format(state.entry_price) if state.entry_price else '-' }}</td>
                        <td>{{ "{:,.4f}".format(state.quantity) if state.quantity else '-' }}</td>
                        <td class="pnl-{{ 'positive' if state.unrealized_pnl > 0 else 'negative' if state.unrealized_pnl < 0 else 'pnl-zero' }}">
                            {{ "{:,.2f}".format(state.unrealized_pnl) if state.unrealized_pnl is not none else '-' }} USDT
                        </td>
                        <td>{{ state.last_analysis_time.split('T')[1].split('.')[0] if state.last_analysis_time else '-' }}</td>
                        <td>
                            <div class="context-box">
                                {{ state.market_context | from_json | to_pretty_json }}
                            </div>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Recent Trades -->
        <div class="card">
            <h2>Recent Trades (Last 50)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time (UTC)</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Quantity</th>
                        <th>PNL (USDT)</th>
                        <th>PNL (%)</th>
                        <th>Reasoning</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in data.trades %}
                    <tr>
                        <td>{{ trade.timestamp.split('T')[1].split('.')[0] }}</td>
                        <td>{{ trade.symbol }}</td>
                        <td class="side-{{ trade.side | lower }}">{{ trade.side }}</td>
                        <td>{{ "{:,.4f}".format(trade.entry_price) }}</td>
                        <td>{{ "{:,.4f}".format(trade.exit_price) }}</td>
                        <td>{{ "{:,.4f}".format(trade.quantity) }}</td>
                        <td class="pnl-{{ 'positive' if trade.pnl > 0 else 'negative' }}">
                            {{ "{:,.2f}".format(trade.pnl) }}
                        </td>
                        <td class="pnl-{{ 'positive' if trade.pnl_pct > 0 else 'negative' }}">
                            {{ "{:,.2f}".format(trade.pnl_pct * 100) }}%
                        </td>
                        <td>{{ trade.reasoning }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Recent Logs -->
        <div class="card">
            <h2>Recent Logs (Last 20)</h2>
            <div class="log-box">
                {% for log in data.logs %}
                <div class="log-entry">
                    <span class="log-timestamp">[{{ log.timestamp.split('T')[1].split('.')[0] }}]</span>
                    <span class="log-symbol">[{{ log.symbol }}]</span>
                    {{ log.message }}
                </div>
                {% endfor %}
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
    # Note: host='0.0.0.0' makes it accessible externally (e.g., from WSL to Windows browser)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)