# AI Comic Factory Configuration

# Ollama API Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 120  # seconds

# Stable Diffusion Configuration
SDXL_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Default Generation Settings
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
MIN_INFERENCE_STEPS = 20
MAX_INFERENCE_STEPS = 50
MIN_GUIDANCE_SCALE = 5.0
MAX_GUIDANCE_SCALE = 15.0

# Image Settings
IMAGE_OUTPUT_FORMAT = "PNG"
IMAGE_QUALITY = 95

# UI Configuration
APP_TITLE = "AI Comic Factory"
APP_ICON = "📚"
SIDEBAR_EXPANDED = True

# Text and Font Settings
FONT_FALLBACKS = [
    "Arial.ttf",
    "/System/Library/Fonts/Arial.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf"  # Windows
]

# Artistic Style Keywords
ARTISTIC_KEYWORDS = "style of Moebius and Frank Miller, dramatic shadows, intricate detail, masterpiece, 4k"

# Negative Prompt for Image Generation
NEGATIVE_PROMPT = "ugly, tiling, poorly drawn, out of frame, disfigured, body out of frame, bad anatomy, watermark, signature, low contrast, distorted face"

# System Requirements
MIN_VRAM_GB = 8
RECOMMENDED_VRAM_GB = 12
MIN_DISK_SPACE_GB = 20

# GPU Memory Management
MIN_FREE_VRAM_MB = 4000  # Minimum free VRAM needed for Stable Diffusion
OLLAMA_KEEP_ALIVE_UNLOAD = 0  # Time to keep Ollama models loaded (0 = unload immediately)
ENABLE_GPU_MEMORY_MONITORING = True
