#!/usr/bin/env python3
"""
Development setup script for AI Comic Factory
This script sets up a development environment with all necessary dependencies.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_system():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 10:
        print(f"✅ Python {python_version.major}.{python_version.minor} is compatible")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} is not compatible")
        print("Please install Python 3.10 or higher")
        return False
    
    # Check OS
    os_name = platform.system()
    print(f"📱 Operating System: {os_name}")
    
    # Check if we have enough disk space (rough estimate)
    try:
        statvfs = os.statvfs('.')
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        free_gb = free_bytes / (1024**3)
        if free_gb >= 20:
            print(f"✅ Sufficient disk space: {free_gb:.1f} GB")
        else:
            print(f"⚠️  Limited disk space: {free_gb:.1f} GB (20GB+ recommended)")
    except:
        print("⚠️  Could not check disk space")
    
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command(
        f"{sys.executable} -m venv venv",
        "Creating virtual environment"
    )

def install_dependencies():
    """Install Python dependencies"""
    # Determine the correct pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    commands = [
        (f"{pip_path} install --upgrade pip", "Upgrading pip"),
        (f"{pip_path} install -r requirements.txt", "Installing dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def check_gpu_setup():
    """Check GPU and CUDA setup"""
    print("\n🔍 Checking GPU setup...")
    
    try:
        # Try to import torch and check CUDA
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("❌ CUDA is not available")
            print("Please install CUDA toolkit and compatible PyTorch version")
            return False
    except ImportError:
        print("❌ PyTorch not installed - cannot check CUDA")
        return False

def create_sample_configs():
    """Create sample configuration files"""
    print("\n📝 Creating sample configuration files...")
    
    # Create a sample .env file
    env_file = Path(".env.example")
    if not env_file.exists():
        env_content = """# AI Comic Factory Environment Variables
# Copy this file to .env and modify as needed

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=120

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Model Configuration
SDXL_BASE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
SDXL_REFINER_MODEL=stabilityai/stable-diffusion-xl-refiner-1.0

# Generation Settings
DEFAULT_INFERENCE_STEPS=30
DEFAULT_GUIDANCE_SCALE=7.5
"""
        env_file.write_text(env_content)
        print("✅ Created .env.example file")
    
    return True

def main():
    """Main setup function"""
    print("🎨 AI Comic Factory - Development Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run this script from the AI-Comic-Factory directory.")
        sys.exit(1)
    
    steps = [
        ("System Requirements", check_system),
        ("Virtual Environment", setup_virtual_environment),
        ("Dependencies", install_dependencies),
        ("GPU Setup", check_gpu_setup),
        ("Sample Configs", create_sample_configs),
    ]
    
    results = []
    for name, func in steps:
        print(f"\n{'='*20} {name} {'='*20}")
        result = func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("DEVELOPMENT SETUP SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Development environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull Llama 3 model: ollama pull llama3")
        print("3. Run the app: ./launch.sh or streamlit run app.py")
    else:
        print("\n⚠️  Some steps failed. Please address the issues above.")
        print("\nTroubleshooting:")
        print("- Ensure Python 3.10+ is installed")
        print("- Install CUDA toolkit for GPU support")
        print("- Check internet connection for model downloads")
        print("- Ensure sufficient disk space (20GB+)")

if __name__ == "__main__":
    main()
