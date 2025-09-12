#!/bin/bash
#
# This script provides a "one-click" way to build and run the mineral classifier demo.
# It handles building the Docker container, pulling DVC data, and opening the app.
#

# --- Configuration ---
# Port for the Streamlit app
APP_PORT=8501
# URL for the Streamlit app
APP_URL="http://localhost:${APP_PORT}"

# --- Helper Functions ---
# Function to print a formatted message
function log() {
  echo
  echo "--- $1 ---"
}

# Function to check if a command exists
function command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# --- Main Script ---
log "Starting Mineral Classifier Demo Setup"

# 1. Check for prerequisites (Docker, DVC)
if ! command_exists docker || ! command_exists dvc; then
  echo "Error: Prerequisites not met."
  echo "Please ensure both 'docker' and 'dvc' are installed and available in your PATH."
  exit 1
fi

# 2. Pull data with DVC
log "Pulling data with DVC..."
dvc pull -f
if [ $? -ne 0 ]; then
  echo "Error: DVC failed to pull data. Please check your DVC setup."
  exit 1
fi

# 3. Build and run Docker container in the background
log "Building and starting the Docker container..."
docker-compose up --build -d
if [ $? -ne 0 ]; then
  echo "Error: Docker Compose failed to start. Please check your Docker setup."
  exit 1
fi

# 4. Wait for the app to be ready
log "Waiting for the Streamlit app to launch..."
sleep 5 # Give the server a moment to start

# 5. Open the application in the default browser
log "Opening the app in your browser at ${APP_URL}"
# Cross-platform way to open a URL
if command_exists xdg-open; then
  xdg-open "${APP_URL}" # Linux
elif command_exists open; then
  open "${APP_URL}" # macOS
elif command_exists start; then
  start "" "${APP_URL}" # Windows
else
  echo "Could not automatically open the browser. Please navigate to ${APP_URL} manually."
fi

log "Demo is running!"
echo "To stop the application and shut down the container, run the following command:"
echo "docker-compose down"
echo

