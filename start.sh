#!/bin/bash

# DDSP Neural Cello - Startup Script
# embracingearth.space - Premium AI Audio Synthesis

echo "🎻 DDSP Neural Cello - Enterprise Audio Synthesis"
echo "embracingearth.space - Premium AI Audio Technology"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.11+ and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    echo "Please install Node.js 18+ and try again."
    exit 1
fi

# Check if training data exists
if [ ! -d "neural_cello_training_samples" ]; then
    echo "⚠️  Training samples not found."
    echo "Please ensure 'neural_cello_training_samples' directory exists."
    echo "Continuing with fallback synthesis..."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models output logs

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Build frontend
echo "🏗️  Building frontend..."
npm run build

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting DDSP Neural Cello..."
echo ""
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start backend in background
echo "Starting backend server..."
python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "Starting frontend server..."
npm start &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down DDSP Neural Cello..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Shutdown complete!"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# Wait for processes
wait







