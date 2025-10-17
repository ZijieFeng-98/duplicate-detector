#!/bin/bash
# Auto-run Streamlit Duplicate Detector

cd "/Users/zijiefeng/Desktop/Guo's lab/APP/Streamlit_Duplicate_Detector"

# Activate virtual environment
source venv/bin/activate

# Kill any existing streamlit processes
pkill -f "streamlit run streamlit_app.py" 2>/dev/null

echo "🚀 Starting Streamlit Duplicate Detector..."
echo "📍 URL: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop"
echo ""

# Run streamlit and auto-restart if it crashes
while true; do
    streamlit run streamlit_app.py --server.port 8501 --server.headless true
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ]; then
        # Clean exit (0) or Ctrl+C (130)
        echo "👋 App stopped cleanly"
        break
    else
        # Crashed - restart after 3 seconds
        echo "⚠️  App crashed (exit code $EXIT_CODE)"
        echo "🔄 Restarting in 3 seconds..."
        sleep 3
    fi
done


