# AI Comic Factory

A powerful Streamlit application that transforms your ideas into stunning comic book panels using AI. The app combines the storytelling capabilities of Llama 3 with the artistic prowess of Stable Diffusion XL to create complete comic panels with artwork and dialogue.

## Features

- **AI-Powered Storytelling**: Uses Llama 3 to generate panel descriptions and dialogue
- **High-Quality Artwork**: Stable Diffusion XL (Base + Refiner) creates stunning comic-style images
- **Interactive Interface**: Clean, user-friendly Streamlit interface
- **Customizable Settings**: Adjustable inference steps and guidance scale
- **Speech Bubbles**: Automatically adds dialogue to generated images
- **Download Support**: Save your comic panels as PNG files

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended: 12GB+ for optimal performance)
- **CUDA**: CUDA-compatible GPU with drivers installed
- **RAM**: At least 16GB system RAM
- **Storage**: 20GB+ free space for model downloads

### Software Requirements

- Python 3.10 or higher
- CUDA toolkit installed
- Ollama running locally

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AI-Comic-Factory
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and Setup Ollama

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull the Llama 3 model:
   ```bash
   ollama pull llama3
   ```
3. Start Ollama server:
   ```bash
   ollama serve
   ```

### 4. GPU Setup

Ensure you have CUDA installed and your GPU is detected:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Running the Application

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

### Using the Interface

1. **Enter Your Idea**: Type your comic concept in the text area
2. **Adjust Settings**: Use the sidebar to modify:
   - Inference Steps (20-50): Higher = better quality, slower generation
   - Guidance Scale (5.0-15.0): Higher = more adherence to prompt
3. **Generate Panel**: Click "Generate Panel" to start the AI workflow
4. **Download**: Save your finished comic panel as a PNG file

### Example Ideas

- "A robot detective solves a case in a rainy, neon-lit city"
- "A superhero soaring through clouds at sunset"
- "A wizard casting a spell in an ancient library"
- "A space explorer discovering alien ruins"
- "A detective finding clues in a dark alley"

## Technical Workflow

The application follows a precise 5-step process:

1. **The Spark**: User input and settings configuration
2. **The Script**: Llama 3 generates panel description and dialogue
3. **The Prompt Forge**: Combines LLM output with artistic keywords
4. **The Canvas**: Stable Diffusion XL creates the artwork
5. **The Final Panel**: Adds dialogue and assembles the final comic

## Troubleshooting

### Common Issues

#### "CUDA Not Available"
- Ensure NVIDIA drivers are installed
- Install CUDA toolkit compatible with your PyTorch version
- Check GPU compatibility

#### "Ollama Not Available"
- Verify Ollama is installed and running
- Check that port 11434 is not blocked
- Ensure Llama 3 model is downloaded

#### "Out of Memory" Error
- Reduce inference steps
- Close other GPU-intensive applications
- Consider using a smaller model variant

#### Model Loading Issues
- Ensure sufficient disk space (20GB+)
- Check internet connection for initial downloads
- Verify Hugging Face access for model downloads

### Performance Tips

- **First Run**: Models will download automatically (may take 10-30 minutes)
- **Memory Management**: Models are cached after first load
- **Quality vs Speed**: Lower inference steps = faster generation
- **GPU Utilization**: Monitor GPU memory usage during generation

## System Status

The application displays real-time system status in the sidebar:
- ✅ CUDA Available: GPU is detected and ready
- ✅ Ollama Running: Local LLM server is accessible
- GPU Information: Shows your graphics card details

## File Structure

```
AI-Comic-Factory/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify system requirements
3. Ensure all dependencies are installed
4. Check GPU and CUDA setup

## Acknowledgments

- **Llama 3**: Meta's large language model for storytelling
- **Stable Diffusion XL**: Stability AI's image generation model
- **Streamlit**: For the web application framework
- **Diffusers**: Hugging Face's diffusion models library
