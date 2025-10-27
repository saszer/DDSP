@echo off
REM DDSP Neural Cello - Startup Script for Windows
REM embracingearth.space - Premium AI Audio Synthesis

echo 🎻 DDSP Neural Cello - Enterprise Audio Synthesis
echo embracingearth.space - Premium AI Audio Technology
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed.
    echo Please install Python 3.11+ and try again.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is required but not installed.
    echo Please install Node.js 18+ and try again.
    pause
    exit /b 1
)

REM Check if training data exists
if not exist "neural_cello_training_samples" (
    echo ⚠️  Training samples not found.
    echo Please ensure 'neural_cello_training_samples' directory exists.
    echo Continuing with fallback synthesis...
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "models" mkdir models
if not exist "output" mkdir output
if not exist "logs" mkdir logs

REM Install Python dependencies
echo 🐍 Installing Python dependencies...
pip install -r requirements.txt

REM Install Node.js dependencies
echo 📦 Installing Node.js dependencies...
npm install

REM Build frontend
echo 🏗️  Building frontend...
npm run build

echo.
echo ✅ Setup complete!
echo.
echo 🚀 Starting DDSP Neural Cello...
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3000
echo.
echo Press Ctrl+C to stop
echo.

REM Start backend in background
echo Starting backend server...
start /b python main.py

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting frontend server...
start /b npm start

echo.
echo ✅ DDSP Neural Cello is running!
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3000
echo.
echo Press any key to stop all services...
pause >nul

REM Kill background processes
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1

echo.
echo 🛑 DDSP Neural Cello stopped!
pause







