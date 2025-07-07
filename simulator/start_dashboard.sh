#!/bin/bash
# WolfXE Trading Dashboard Launcher - Dashboard Only

echo "Starting WolfXE Trading Dashboard..."

# Activate environment
source ../wolfxe_env/bin/activate

# Check if Flask is available
python -c "import flask" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Using Flask-based dashboard..."
    python streaming/trading_stream.py &
    STREAM_PID=$!
    DASHBOARD_URL="http://localhost:9900"
else
    echo "Flask not found, using simple dashboard..."
    python streaming/simple_dashboard.py --symbol S_AAPL --port 8000 &
    STREAM_PID=$!
    DASHBOARD_URL="http://localhost:8000"
fi

# Wait for service to start
sleep 3

echo "Dashboard running at: $DASHBOARD_URL"
echo "Use the dashboard interface to start/stop simulations manually"
echo "Press Ctrl+C to stop dashboard"

# Wait for interrupt - NO AUTO-SIMULATION
trap "kill $STREAM_PID; exit" INT
wait

