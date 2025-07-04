"""
Utility functions for AI Comic Factory
"""

import streamlit as st
import requests
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import io
import config

def check_system_requirements():
    """
    Check if the system meets the minimum requirements.
    Returns a dictionary with status information.
    """
    status = {
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": 0,
        "ollama_running": False,
        "llama3_available": False
    }
    
    # Check CUDA
    if torch.cuda.is_available():
        status["cuda_available"] = True
        status["gpu_name"] = torch.cuda.get_device_name(0)
        
        # Estimate VRAM (this is approximate)
        try:
            status["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            status["vram_gb"] = 0
    
    # Check Ollama
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            status["ollama_running"] = True
            
            # Check for Llama 3 model
            tags = response.json()
            models = [model['name'] for model in tags.get('models', [])]
            status["llama3_available"] = any('llama3' in model for model in models)
    except:
        pass
    
    return status

def display_system_status():
    """
    Display system status in the Streamlit sidebar.
    """
    st.sidebar.markdown("### System Status")
    status = check_system_requirements()
    
    # CUDA Status
    if status["cuda_available"]:
        st.sidebar.success("✅ CUDA Available")
        if status["gpu_name"]:
            st.sidebar.info(f"GPU: {status['gpu_name']}")
        if status["vram_gb"] > 0:
            vram_status = "✅" if status["vram_gb"] >= config.MIN_VRAM_GB else "⚠️"
            st.sidebar.info(f"{vram_status} VRAM: {status['vram_gb']:.1f} GB")
            
            # Show current GPU memory usage
            memory_info = get_gpu_memory_info()
            if memory_info["available"]:
                st.sidebar.info(f"💾 GPU Memory: {memory_info['used_mb']:.0f}MB / {memory_info['total_mb']:.0f}MB ({memory_info['usage_percent']:.1f}%)")
                
                # Warning if memory usage is high
                if memory_info["usage_percent"] > 80:
                    st.sidebar.warning("⚠️ High GPU memory usage detected")
    else:
        st.sidebar.error("❌ CUDA Not Available")
    
    # Ollama Status
    if status["ollama_running"]:
        st.sidebar.success("✅ Ollama Running")
        if status["llama3_available"]:
            st.sidebar.success("✅ Llama 3 Available")
        else:
            st.sidebar.warning("⚠️ Llama 3 Not Found")
    else:
        st.sidebar.error("❌ Ollama Not Available")
    
    return status

def get_font(size=16):
    """
    Get the best available font for text rendering.
    
    Args:
        size (int): Font size
        
    Returns:
        PIL.ImageFont: Font object
    """
    for font_path in config.FONT_FALLBACKS:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    
    # Fallback to default font
    return ImageFont.load_default()

def validate_json_response(response_text):
    """
    Validate and parse JSON response from Llama 3.
    
    Args:
        response_text (str): Raw response text
        
    Returns:
        dict or None: Parsed JSON data or None if invalid
    """
    try:
        data = json.loads(response_text)
        
        # Validate required fields
        if not isinstance(data, dict):
            return None
        
        if "panel_description" not in data or "dialogue" not in data:
            return None
        
        # Ensure both fields are strings and not empty
        if not isinstance(data["panel_description"], str) or not data["panel_description"].strip():
            return None
        
        if not isinstance(data["dialogue"], str) or not data["dialogue"].strip():
            return None
        
        return data
    
    except json.JSONDecodeError:
        return None

def validate_comic_json_response(response_text):
    """
    Validate and parse JSON response from Llama 3 for 5-panel comic.
    
    Args:
        response_text (str): Raw response text
        
    Returns:
        dict or None: Parsed JSON data or None if invalid
    """
    try:
        data = json.loads(response_text)
        
        # Validate required fields
        if not isinstance(data, dict):
            return None
        
        if "comic_style" not in data or "panels" not in data:
            return None
        
        # Validate comic_style
        if not isinstance(data["comic_style"], str) or not data["comic_style"].strip():
            return None
        
        # Validate panels array
        panels = data["panels"]
        if not isinstance(panels, list) or len(panels) != 5:
            return None
        
        # Validate each panel
        for panel in panels:
            if not isinstance(panel, dict):
                return None
            
            if "panel_description" not in panel or "dialogue" not in panel:
                return None
            
            # Ensure both fields are strings and not empty
            if not isinstance(panel["panel_description"], str) or not panel["panel_description"].strip():
                return None
            
            if not isinstance(panel["dialogue"], str) or not panel["dialogue"].strip():
                return None
            
            # Check token limit for panel_description (approximately 12 words = ~77 tokens)
            description_words = len(panel["panel_description"].split())
            if description_words > 15:  # Allow some flexibility
                return None
        
        return data
    
    except json.JSONDecodeError:
        return None

def create_download_link(image, filename="comic_panel.png"):
    """
    Create a download link for the generated image.
    
    Args:
        image (PIL.Image): The image to download
        filename (str): Filename for download
        
    Returns:
        bytes: Image data for download
    """
    img_buffer = io.BytesIO()
    image.save(img_buffer, format=config.IMAGE_OUTPUT_FORMAT, quality=config.IMAGE_QUALITY)
    img_buffer.seek(0)
    return img_buffer.getvalue()

def calculate_text_position(img_size, text_size, position="top"):
    """
    Calculate optimal text position for speech bubble.
    
    Args:
        img_size (tuple): (width, height) of image
        text_size (tuple): (width, height) of text
        position (str): "top", "bottom", "center"
        
    Returns:
        tuple: (x, y) position for text
    """
    img_width, img_height = img_size
    text_width, text_height = text_size
    
    padding = 20
    
    # Calculate x position (center horizontally)
    x = (img_width - text_width) // 2
    
    # Calculate y position based on preference
    if position == "top":
        y = padding
    elif position == "bottom":
        y = img_height - text_height - padding
    else:  # center
        y = (img_height - text_height) // 2
    
    return max(0, x), max(0, y)

def format_error_message(error):
    """
    Format error messages for user display.
    
    Args:
        error (Exception): The error object
        
    Returns:
        str: Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Common error translations
    if "CUDA out of memory" in error_msg:
        return "GPU memory is full. Try reducing the inference steps or closing other GPU applications."
    elif "Connection refused" in error_msg:
        return "Cannot connect to Ollama. Please make sure Ollama is running on localhost:11434."
    elif "No such file or directory" in error_msg:
        return "Model files not found. Please check your installation."
    elif "JSON" in error_type:
        return "Invalid response format from AI model. Please try again."
    else:
        return f"An error occurred: {error_msg}"

def estimate_generation_time(steps, has_refiner=True, num_panels=5):
    """
    Estimate generation time based on steps, settings, and number of panels.
    
    Args:
        steps (int): Number of inference steps
        has_refiner (bool): Whether refiner model is used
        num_panels (int): Number of panels to generate
        
    Returns:
        str: Estimated time range
    """
    # These are rough estimates and will vary based on hardware
    base_time = steps * 0.5  # ~0.5 seconds per step for base model
    refiner_time = steps * 0.3 if has_refiner else 0  # ~0.3 seconds per step for refiner
    
    # Time per panel
    time_per_panel = base_time + refiner_time
    
    # Total time for all panels
    total_time = time_per_panel * num_panels
    
    if total_time < 60:
        return f"~{total_time:.0f} seconds"
    elif total_time < 300:  # 5 minutes
        minutes = total_time / 60
        return f"~{minutes:.1f} minutes"
    else:
        minutes = total_time / 60
        return f"~{minutes:.0f} minutes"

def get_gpu_memory_info():
    """
    Get detailed GPU memory information.
    
    Returns:
        dict: GPU memory information
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "used_mb": 0,
            "total_mb": 0,
            "free_mb": 0,
            "usage_percent": 0
        }
    
    used = torch.cuda.memory_allocated() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    free = total - used
    usage_percent = (used / total) * 100
    
    return {
        "available": True,
        "used_mb": used,
        "total_mb": total,
        "free_mb": free,
        "usage_percent": usage_percent
    }

def check_gpu_memory_available(required_mb=8000):
    """
    Check if enough GPU memory is available.
    
    Args:
        required_mb (int): Required memory in MB
        
    Returns:
        bool: True if enough memory is available
    """
    memory_info = get_gpu_memory_info()
    
    if not memory_info["available"]:
        return False
    
    return memory_info["free_mb"] >= required_mb
