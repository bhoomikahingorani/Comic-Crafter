#!/usr/bin/env python3
"""
Setup script for AI Comic Factory
This script helps verify the installation and setup of the AI Comic Factory application.
"""

import subprocess
import sys
import os
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10 or higher"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.10 or higher")
        return False

def check_cuda():
    """Check if CUDA is available"""
    print("Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("❌ CUDA is not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed - cannot check CUDA")
        return False

def check_ollama():
    """Check if Ollama is running and has Llama 3 model"""
    print("Checking Ollama status...")
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            
            # Check if Llama 3 model is available
            tags = response.json()
            models = [model['name'] for model in tags.get('models', [])]
            
            if any('llama3' in model for model in models):
                print("✅ Llama 3 model is available")
                return True
            else:
                print("❌ Llama 3 model not found")
                print("Run: ollama pull llama3")
                return False
        else:
            print("❌ Ollama is not responding")
            return False
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running")
        print("Please start Ollama with: ollama serve")
        return False

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_disk_space():
    """Check available disk space"""
    print("Checking disk space...")
    try:
        statvfs = os.statvfs('.')
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        free_gb = free_bytes / (1024**3)
        
        if free_gb >= 20:
            print(f"✅ Sufficient disk space: {free_gb:.1f} GB available")
            return True
        else:
            print(f"❌ Insufficient disk space: {free_gb:.1f} GB available")
            print("At least 20GB free space is recommended for model downloads")
            return False
    except:
        print("❌ Could not check disk space")
        return False

def main():
    """Main setup verification function"""
    print("=" * 50)
    print("AI Comic Factory Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Disk Space", check_disk_space),
        ("Python Requirements", install_requirements),
        ("CUDA Support", check_cuda),
        ("Ollama & Llama 3", check_ollama),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 20)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to run the AI Comic Factory!")
        print("\nTo start the application:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Run the app: streamlit run app.py")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above before running the app.")
        print("\nCommon solutions:")
        print("- Install CUDA drivers and toolkit")
        print("- Install Ollama and pull llama3 model")
        print("- Free up disk space")
        print("- Install Python 3.10+")

if __name__ == "__main__":
    main()
