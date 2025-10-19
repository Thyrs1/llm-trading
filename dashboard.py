# dashboard.py

from flask import Flask, render_template, jsonify
import json
import os
import logging

# Disable default Flask logging for cleaner console output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
STATUS_FILE = 'status.json'

@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/status')
def get_status():
    """Provides the status.json file content as an API endpoint."""
    if not os.path.exists(STATUS_FILE):
        return jsonify({"error": "Status file not found. Is the bot running?"}), 404
    
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Failed to read or parse status file: {e}"}), 500

if __name__ == '__main__':
    print("="*40)
    print("📈 Gemini Trading Bot Dashboard 📈")
    print("Visit http://127.0.0.1:5000")
    print("Ensure trading_bot.py is running in another terminal.")
    print("="*40)
    app.run(host='0.0.0.0', port=5000, debug=False)