REM Run the python scripts in sequence simultaneously
start cmd /k "python trading_bot.py"
start cmd /k "python dashboard.py"
start cmd /k "python watcher.py"
pause