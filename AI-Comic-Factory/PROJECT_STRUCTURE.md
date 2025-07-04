# AI Comic Factory - Project Structure

```
AI-Comic-Factory/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── setup.py              # Setup verification script
├── dev_setup.py          # Development setup script
├── launch.sh             # Launch script for Unix/Linux/macOS
├── .gitignore            # Git ignore file
├── .env.example          # Environment variables template
└── README.md             # Documentation
```

## File Descriptions

### Core Application Files

- **app.py**: Main Streamlit application implementing the 5-step AI Comic Factory workflow
- **config.py**: Centralized configuration settings for the application
- **utils.py**: Helper functions for system checks, image processing, and error handling

### Setup and Configuration

- **requirements.txt**: Python package dependencies
- **setup.py**: Verification script to check system requirements
- **dev_setup.py**: Development environment setup script
- **launch.sh**: Convenient launch script for Unix-based systems
- **.env.example**: Template for environment variables

### Documentation

- **README.md**: Comprehensive documentation and setup instructions
- **PROJECT_STRUCTURE.md**: This file describing the project organization

## Key Features Implemented

1. **5-Step Workflow**: Exactly as specified in the requirements
2. **Modular Design**: Separated concerns into config, utils, and main app
3. **Error Handling**: Comprehensive error handling and user feedback
4. **System Checks**: Real-time system status monitoring
5. **Easy Setup**: Multiple setup scripts for different use cases
6. **Professional Structure**: Following Python best practices

## Usage

1. Run setup verification: `python setup.py`
2. Launch application: `./launch.sh` or `streamlit run app.py`
3. Access at: `http://localhost:8501`

## Dependencies

The application requires:
- Python 3.10+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- Ollama with Llama 3 model
- All Python packages listed in requirements.txt
