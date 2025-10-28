#!/bin/bash
# Post-deploy script for Fly.io
# embracingearth.space

echo "🚀 Post-deploy script running..."

# Ensure directories exist
mkdir -p /app/models
mkdir -p /app/output
mkdir -p /app/public

# Set permissions
chmod -R 755 /app/models
chmod -R 755 /app/output

# Restart the application
echo "✅ Deployment complete, restarting application..."
flyctl apps restart ddsp

echo "✅ Post-deploy complete!"

