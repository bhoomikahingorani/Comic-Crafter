# Comic Book Generator

This repository contains two separate applications for generating comic book stories and visualizations:

1. **ComicCrafter** - An edge-deployed application that generates comic stories using local LLM models
2. **Comic-AI** - A cloud-based application that creates complete comic strips with AI-generated images

## Overview

Both applications help users create comic book stories based on prompts, but they take different approaches:

- **ComicCrafter** runs entirely on your local machine using the Mistral model, making it suitable for offline use and edge deployment
- **Comic-AI** leverages cloud APIs (Groq and Together) to generate more sophisticated stories with accompanying AI-generated images
- **Demo Video Link** https://www.loom.com/share/63cdee7d7f29482b87b78234163a71f0?sid=795a38fd-9d38-4d2c-a799-82193751ea0b

## Applications

### 1. ComicCrafter (Edge Deployment)

ComicCrafter generates comic stories using a locally-deployed Mistral model, structured with panels, narration, and dialogue.

**Features:**
- Generates comic story with structured panels
- Parse and display comic story elements (narration, dialogue, thoughts)
- Select from multiple genres (Superhero, Science Fiction, Fantasy, etc.)
- Control story length via token settings
- Runs entirely locally - no API needed

**Files:**
- `app.py` - Main Streamlit application
- `story_generator1.py` or `story_generator2.py` - Story generation and parsing logic

**Output Structure**
* INTRODUCTION:
- Panel 1: [Scene Description]
Character Dialogues \
Narration Boxes
* STORYLINE:
- Panel 2: [Scene Description] \
...
* CLIMAX:
- Panel 3: [Scene Description] \
...
* MORAL:
Panel 4: [Scene Description] \
...

### 2. Comic-AI (API-based)

Comic-AI creates complete comic strips with both story and images using cloud-based AI services (Groq and Together).

**Features:**
- Generate structured comic stories with detailed character descriptions
- Create AI-generated images for each panel
- Choose from different art styles (Manga, Comic Book, Noir, Watercolor, Cartoon)
- Select from multiple genres
- Save and review generated comics in history
- Custom-styled UI with dark theme

**Files:**
- `Multi_Art_Generator.py` - Complete application

## Setup Instructions

### Prerequisites

- Python 3.8+
- For ComicCrafter (Edge): Mistral model file
- For Comic-AI (API): Groq and Together API keys

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/comic-book-generator.git
cd comic-book-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root (for Comic-AI):
```
GROQ_API_KEY=your_groq_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
```

### Running ComicCrafter (Edge Deployment)

1. Installations:
   - Download the Mistral-Nemo-Instruct-2407-Q3_K_L.gguf model and place it in your preferred directory
   - Download Visual Studios build tools
   - Follow the instructions below:
```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# For Windows:
venv\Scripts\activate
# For Unix/MacOS:
source venv/bin/activate

# Install required packages
pip install streamlit
pip install llama-cpp-python
pip install regex  # for re module

```

2. Update the model path in `app.py`:
```python
model_path = r"path/to/your/Mistral-Nemo-Instruct-2407-Q3_K_L.gguf"
```

3. Run the application:
```bash
streamlit run app.py
```

### Running Comic-AI (API Version)

1. Ensure you have valid API keys in your `.env` file

2. Run the application:
```bash
streamlit run Multi_Art_Generator.py
```

## Usage

### ComicCrafter

1. Enter your story concept in the text area
2. Select a genre from the dropdown menu
3. Adjust the maximum length if desired
4. Click "Generate Comic Story"
5. Explore the generated story with its panels, narration, and dialogue

### Comic-AI

1. Enter your story prompt
2. Select a genre and art style
3. Click "Generate Comic"
4. View the generated character sheet and story panels
5. Generate images for individual panels or all at once
6. Save your comic to history
7. View your saved comics in the History tab

## Technical Details

### ComicCrafter

- Uses `llama-cpp-python` for local model inference
- Implements regex-based parsing to structure the comic story
- Provides debugging tools for panel extraction
- Displays structured content with expandable panels

### Comic-AI (Multi_Art_geberator)

- Uses Groq API (with llama3-70b-8192 model) for story generation
- Uses Together API for image generation (with FLUX.1-schnell-Free model)
- Implements regex-based parsing for character and panel extraction
- Features a custom-styled UI with dark theme
- Stores comic history in session state

### Comic-AI (Manga_generator)

- Uses Groq API (with llama3-70b-8192 model) for story generation
- Uses Together API for image generation (with FLUX.1-schnell-Free model)
- Implements regex-based parsing for character and panel extraction
- Features a custom-styled UI with dark theme
- Stores comic history in session state
- Editing capabilities for generated stories *(Extra)*

## Environment Variables

For Comic-AI, the following environment variables are required in your `.env` file:

- `GROQ_API_KEY`: Your API key for the Groq service
- `TOGETHER_API_KEY`: Your API key for the Together AI service

## Future Improvements

- Export comics as PDF or image sequence
- Add more art styles and genre options
- Implement user accounts for persistent storage
- Add audio narration options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
