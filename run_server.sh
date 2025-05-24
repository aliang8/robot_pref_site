#!/bin/bash

# Function to cleanup processes
cleanup_processes() {
    echo "Cleaning up old processes..."
    
    # Kill any Python processes running app.py
    pkill -f "python.*app.py" || true
    
    # Find and kill any process using port 8000
    pid=$(lsof -ti:8000)
    if [ ! -z "$pid" ]; then
        echo "Killing process $pid using port 8000"
        kill -9 $pid || true
    fi
    
    # Wait a moment to ensure ports are freed
    sleep 2
}

# Run cleanup on script start
cleanup_processes

# Activate conda environment if it exists
if [ -f "/scr/aliang80/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/scr/aliang80/miniconda3/etc/profile.d/conda.sh"
    conda activate robot_pref
fi

# Set environment variables
export FLASK_APP=public/app.py
export FLASK_ENV=development
export PYTHONUNBUFFERED=1  # This ensures Python output is sent straight to terminal

# Create necessary directories
mkdir -p logs
mkdir -p temp_videos
mkdir -p data

# Function to start the server
start_server() {
    echo "Starting Flask server..."
    python public/app.py 2>&1 | tee -a logs/server.log
}

# Cleanup on script exit
trap cleanup_processes EXIT

# Start the server in a loop
while true; do
    start_server
    echo "Server crashed or stopped. Cleaning up and restarting in 5 seconds..."
    cleanup_processes
    sleep 5
done 