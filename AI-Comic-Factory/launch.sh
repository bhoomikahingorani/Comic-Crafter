#!/bin/bash

# AI Comic Factory Launch Script
# This script helps launch the AI Comic Factory application

echo "🎨 AI Comic Factory Launch Script"
echo "================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please run this script from the AI-Comic-Factory directory."
    exit 1
fi

echo "✅ Python 3 found"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/installed.flag" ]; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch venv/installed.flag
        echo "✅ Requirements installed successfully"
    else
        echo "❌ Failed to install requirements"
        exit 1
    fi
fi

# Check if Ollama is running
echo "🔍 Checking Ollama status..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running"
    echo "💡 Please start Ollama in another terminal: ollama serve"
    echo "💡 And make sure Llama 3 is installed: ollama pull llama3"
    echo ""
    echo "Would you like to continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

echo "🚀 Starting AI Comic Factory..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Launch the Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
