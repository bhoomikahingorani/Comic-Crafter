"""
AI Comic Factory - A Streamlit Application
Generates comic book panels using Llama 3 for storytelling and Stable Diffusion XL for artwork.
"""

import streamlit as st
import requests
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import DiffusionPipeline
import io
import os
import gc
import config
import utils


# Configure page layout
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded" if config.SIDEBAR_EXPANDED else "collapsed"
)

# Cache the model loading to prevent reloading on every run
@st.cache_resource
def load_stable_diffusion_models():
    """
    Load Stable Diffusion XL base and refiner models.
    Returns: tuple of (base_model, refiner_model)
    """
    try:
        # Clear GPU cache before loading models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load base model
        base = DiffusionPipeline.from_pretrained(
            config.SDXL_BASE_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        base.to("cuda")

        # Load refiner model
        refiner = DiffusionPipeline.from_pretrained(
            config.SDXL_REFINER_MODEL,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.to("cuda")
        
        return base, refiner
    except Exception as e:
        st.error(f"Error loading Stable Diffusion models: {str(e)}")
        return None, None


def call_llama3_api(user_idea):
    """
    Call the Llama 3 API to generate 5 comic panels with descriptions and dialogue.
    
    Args:
        user_idea (str): The user's comic idea
        
    Returns:
        dict: JSON response containing comic_style, panels array with panel_description and dialogue
    """
    prompt = f"""You are a creative comic book writer. Based on the following idea, create a 5-panel comic story. Your response MUST be a single JSON object with the following structure:

{{
  "comic_style": "A brief description of the overall visual style and tone (max 15 words)",
  "panels": [
    {{
      "panel_description": "Vivid visual scene description (max 12 words for tokenizer limit)",
      "dialogue": "Character dialogue or narration (max 8 words)"
    }},
    // ... 4 more panels
  ]
}}

IMPORTANT: Keep panel descriptions under 12 words each to stay within the 77-token limit for image generation.

Idea: "{user_idea}"

JSON Response:"""

    api_url = f"{config.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=config.OLLAMA_TIMEOUT)
        response.raise_for_status()
        
        # Parse the response
        response_data = response.json()
        llm_response = response_data.get("response", "")
        
        # Use utility function to validate JSON response
        comic_data = utils.validate_comic_json_response(llm_response)
        
        if comic_data is None:
            st.error("Invalid response format from Llama 3. Please try again.")
            return None
        
        return comic_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Llama 3 API: {utils.format_error_message(e)}")
        return None


def forge_master_prompt(panel_description, comic_style):
    """
    Create the master prompt for Stable Diffusion by combining the panel description
    with the consistent comic style and artistic keywords.
    
    Args:
        panel_description (str): The panel description from Llama 3
        comic_style (str): The consistent comic style for all panels
        
    Returns:
        str: The master prompt for Stable Diffusion
    """
    master_prompt = f"cinematic comic book art, {comic_style}, {panel_description}, {config.ARTISTIC_KEYWORDS}"
    return master_prompt


def generate_comic_image(master_prompt, steps, cfg_scale):
    """
    Generate comic image using Stable Diffusion XL base and refiner models.
    
    Args:
        master_prompt (str): The master prompt for image generation
        steps (int): Number of inference steps
        cfg_scale (float): Guidance scale for generation
        
    Returns:
        PIL.Image: The generated comic image
    """
    # Clear GPU memory before loading models
    clear_gpu_memory()
    
    # Load models
    base, refiner = load_stable_diffusion_models()
    
    if base is None or refiner is None:
        st.error("Failed to load Stable Diffusion models")
        return None
    
    # Define negative prompt
    negative_prompt = config.NEGATIVE_PROMPT
    
    try:
        # Generate with base model
        with st.spinner("Generating base image..."):
            # Show GPU memory usage
            used_mem, total_mem = get_gpu_memory_usage()
            st.sidebar.info(f"GPU Memory (Base): {used_mem:.0f}MB / {total_mem:.0f}MB")
            
            latent_image = base(
                prompt=master_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                output_type="latent"
            ).images[0]
            
            # Clear cache between base and refiner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Refine with refiner model
        with st.spinner("Refining image..."):
            # Show GPU memory usage
            used_mem, total_mem = get_gpu_memory_usage()
            st.sidebar.info(f"GPU Memory (Refiner): {used_mem:.0f}MB / {total_mem:.0f}MB")
            
            final_image = refiner(
                prompt=master_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                image=latent_image
            ).images[0]
            
            # Clear cache after generation
            clear_gpu_memory()
        
        return final_image
    
    except Exception as e:
        # Clean up GPU memory on error
        clear_gpu_memory()
        st.error(f"Error generating image: {utils.format_error_message(e)}")
        return None


def add_dialogue_to_image(image, dialogue):
    """
    Add dialogue text to the comic image in a speech bubble style.
    
    Args:
        image (PIL.Image): The comic image
        dialogue (str): The dialogue text to add
        
    Returns:
        PIL.Image: The image with dialogue added
    """
    # Create a copy of the image to draw on
    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    # Get image dimensions
    img_width, img_height = img_with_text.size
    
    # Try to use a better font, fallback to default if not available
    try:
        font_size = max(16, img_width // 40)  # Scale font size with image
        font = utils.get_font(font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate text dimensions
    text_bbox = draw.textbbox((0, 0), dialogue, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate speech bubble dimensions and position
    bubble_padding = 20
    bubble_width = text_width + 2 * bubble_padding
    bubble_height = text_height + 2 * bubble_padding
    
    # Position bubble at top center of image
    bubble_x = (img_width - bubble_width) // 2
    bubble_y = 20
    
    # Draw speech bubble background (white with black border)
    bubble_coords = [
        bubble_x, bubble_y,
        bubble_x + bubble_width, bubble_y + bubble_height
    ]
    
    # Draw white background
    draw.rectangle(bubble_coords, fill="white", outline="black", width=3)
    
    # Draw text
    text_x = bubble_x + bubble_padding
    text_y = bubble_y + bubble_padding
    draw.text((text_x, text_y), dialogue, fill="black", font=font)
    
    return img_with_text


def clear_gpu_memory():
    """
    Clear GPU memory by forcing garbage collection and emptying CUDA cache.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def get_gpu_memory_usage():
    """
    Get current GPU memory usage in MB.
    Returns: tuple of (used_memory_mb, total_memory_mb)
    """
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return used, total
    return 0, 0

def unload_ollama_models():
    """
    Attempt to unload Ollama models from GPU memory.
    This sends a request to Ollama to unload all models.
    """
    try:
        # Send request to unload all models
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": "unload",
                "keep_alive": 0  # This tells Ollama to unload the model immediately
            },
            timeout=10
        )
        
        # Clear GPU cache after attempting to unload
        clear_gpu_memory()
        
        return True
    except Exception as e:
        st.warning(f"Could not unload Ollama models: {str(e)}")
        # Still try to clear GPU cache
        clear_gpu_memory()
        return False


def generate_comic_panels(comic_data, steps, cfg_scale):
    """
    Generate all 5 comic panels using Stable Diffusion XL.
    
    Args:
        comic_data (dict): Comic data with style and panels
        steps (int): Number of inference steps
        cfg_scale (float): Guidance scale for generation
        
    Returns:
        list: List of PIL Images for each panel
    """
    panels = []
    comic_style = comic_data.get("comic_style", "")
    panel_data = comic_data.get("panels", [])
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Load models once for all panels
    base, refiner = load_stable_diffusion_models()
    
    if base is None or refiner is None:
        st.error("Failed to load Stable Diffusion models")
        return None
    
    # Generate each panel
    for i, panel in enumerate(panel_data):
        panel_description = panel.get("panel_description", "")
        
        # Create master prompt with consistent style
        master_prompt = forge_master_prompt(panel_description, comic_style)
        
        # Show progress
        with st.spinner(f"Generating panel {i+1}/5: {panel_description}"):
            # Show GPU memory usage
            used_mem, total_mem = get_gpu_memory_usage()
            st.sidebar.info(f"Panel {i+1} GPU Memory: {used_mem:.0f}MB / {total_mem:.0f}MB")
            
            try:
                # Generate with base model
                latent_image = base(
                    prompt=master_prompt,
                    negative_prompt=config.NEGATIVE_PROMPT,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    output_type="latent"
                ).images[0]
                
                # Clear cache between base and refiner
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Refine with refiner model
                final_image = refiner(
                    prompt=master_prompt,
                    negative_prompt=config.NEGATIVE_PROMPT,
                    num_inference_steps=steps,
                    image=latent_image
                ).images[0]
                
                panels.append(final_image)
                
                # Clear cache after each panel
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                st.error(f"Error generating panel {i+1}: {utils.format_error_message(e)}")
                # Continue with other panels even if one fails
                panels.append(None)
    
    # Final cleanup
    clear_gpu_memory()
    
    return panels


def create_comic_layout(panels, dialogues):
    """
    Create a comic book layout with 5 panels arranged in a grid.
    
    Args:
        panels (list): List of PIL Images
        dialogues (list): List of dialogue strings
        
    Returns:
        PIL.Image: The complete comic layout
    """
    if len(panels) != 5 or len(dialogues) != 5:
        st.error("Expected 5 panels and 5 dialogues")
        return None
    
    # Filter out None panels
    valid_panels = [(panel, dialogue) for panel, dialogue in zip(panels, dialogues) if panel is not None]
    
    if not valid_panels:
        st.error("No valid panels generated")
        return None
    
    # If we have less than 5 panels, adjust layout
    num_panels = len(valid_panels)
    
    # Define layout dimensions
    panel_width = 512
    panel_height = 512
    margin = 20
    
    # Calculate layout based on number of panels
    if num_panels <= 2:
        cols, rows = num_panels, 1
    elif num_panels <= 4:
        cols, rows = 2, 2
    else:
        cols, rows = 3, 2  # 3x2 grid for 5 panels
    
    # Calculate canvas size
    canvas_width = cols * panel_width + (cols + 1) * margin
    canvas_height = rows * panel_height + (rows + 1) * margin
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    # Add panels to canvas
    for i, (panel, dialogue) in enumerate(valid_panels):
        # Calculate position
        col = i % cols
        row = i // cols
        
        x = margin + col * (panel_width + margin)
        y = margin + row * (panel_height + margin)
        
        # Resize panel to fit
        panel_resized = panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        
        # Add dialogue to panel
        panel_with_dialogue = add_dialogue_to_image(panel_resized, dialogue)
        
        # Paste panel onto canvas
        canvas.paste(panel_with_dialogue, (x, y))
    
    return canvas


def main():
    """
    Main Streamlit application function.
    """
    # Step 1: The Spark (User Input & Settings UI)
    st.title(config.APP_TITLE)
    st.markdown("Transform your ideas into stunning **5-panel comic stories** with consistent style and narrative flow!")
    
    # Main input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_idea = st.text_area(
            "Enter your comic story idea:",
            placeholder="A robot detective solves a case in a rainy, neon-lit city. The story should have suspense, action, and a twist ending...",
            height=150
        )
        
        generate_button = st.button("Generate 5-Panel Comic", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### Example Story Ideas:")
        st.markdown("""
        - A superhero's origin story with a moral dilemma
        - A wizard's quest to save their village from darkness
        - A space explorer's discovery of ancient alien secrets
        - A detective's investigation with an unexpected twist
        - A time traveler's attempt to fix a historical mistake
        """)
        
        st.markdown("### Features:")
        st.markdown("""
        - **5 Sequential Panels** with narrative flow
        - **Consistent Visual Style** across all panels
        - **Token-Optimized Prompts** (under 77 tokens)
        - **Professional Layout** with speech bubbles
        """)
    
    # Sidebar: Control Panel
    st.sidebar.header("Image Generation Settings")
    steps = st.sidebar.slider("Inference Steps", 
                             config.MIN_INFERENCE_STEPS, 
                             config.MAX_INFERENCE_STEPS, 
                             config.DEFAULT_INFERENCE_STEPS)
    cfg_scale = st.sidebar.slider("Guidance Scale (CFG)", 
                                 config.MIN_GUIDANCE_SCALE, 
                                 config.MAX_GUIDANCE_SCALE, 
                                 config.DEFAULT_GUIDANCE_SCALE)
    
    # Add generation time estimate for 5 panels
    estimated_time = utils.estimate_generation_time(steps, has_refiner=True, num_panels=5)
    st.sidebar.info(f"Estimated generation time (5 panels): {estimated_time}")
    st.sidebar.warning("⚠️ 5-panel generation requires significant GPU memory and time")
    
    # Display system status
    st.sidebar.markdown("---")
    system_status = utils.display_system_status()
    
    # Main workflow execution
    if generate_button:
        if not user_idea.strip():
            st.error("Please enter a comic idea!")
            return
        
        # Check system requirements before starting
        if not system_status["cuda_available"]:
            st.error("CUDA is not available. Please ensure you have a CUDA-compatible GPU and drivers installed.")
            return
        
        if not system_status["ollama_running"]:
            st.error("Ollama is not running. Please start Ollama with 'ollama serve' and try again.")
            return
        
        if not system_status["llama3_available"]:
            st.error("Llama 3 model is not available. Please run 'ollama pull llama3' and try again.")
            return
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 2: The Script (Llama 3 Writes)
            status_text.text("Step 1/4: Generating 5-panel comic story with Llama 3...")
            progress_bar.progress(0.25)
            
            # Display GPU memory usage before Ollama
            used_mem, total_mem = get_gpu_memory_usage()
            st.sidebar.info(f"GPU Memory: {used_mem:.0f}MB / {total_mem:.0f}MB")
            
            with st.spinner("Llama 3 is writing your 5-panel comic story..."):
                comic_data = call_llama3_api(user_idea)
            
            if comic_data is None:
                st.error("Failed to generate comic story. Please check if Ollama is running.")
                return
            
            comic_style = comic_data.get("comic_style", "")
            panels_data = comic_data.get("panels", [])
            
            if not comic_style or not panels_data or len(panels_data) != 5:
                st.error("Invalid response from Llama 3. Expected 5 panels with comic style.")
                return
            
            # Display the generated story
            st.success("Comic story generated!")
            with st.expander("View Generated Story", expanded=True):
                st.markdown(f"**Comic Style:** {comic_style}")
                st.markdown("**Panels:**")
                for i, panel in enumerate(panels_data):
                    st.markdown(f"**Panel {i+1}:** {panel.get('panel_description', '')}")
                    st.markdown(f"**Dialogue:** {panel.get('dialogue', '')}")
                    st.markdown("---")
            
            # Step 3: The Prompt Forge (Backend Logic)
            status_text.text("Step 2/4: Forging master prompts for all panels...")
            progress_bar.progress(0.50)
            
            # Show master prompts for each panel
            with st.expander("View Master Prompts"):
                for i, panel in enumerate(panels_data):
                    master_prompt = forge_master_prompt(panel.get('panel_description', ''), comic_style)
                    st.markdown(f"**Panel {i+1}:**")
                    st.code(master_prompt)
            
            # CRITICAL: Unload Ollama models before loading Stable Diffusion
            status_text.text("Step 2.5/4: Clearing GPU memory for image generation...")
            with st.spinner("Preparing GPU for image generation..."):
                # First, unload Ollama models
                unload_ollama_models()
                
                # Check if we have enough free GPU memory
                memory_info = utils.get_gpu_memory_info()
                if memory_info["available"]:
                    required_memory = config.MIN_FREE_VRAM_MB
                    if memory_info["free_mb"] < required_memory:
                        st.error(f"Insufficient GPU memory. Need {required_memory}MB free, but only {memory_info['free_mb']:.0f}MB available.")
                        st.error("Try closing other GPU applications or reducing inference steps.")
                        return
                
                # Display GPU memory after clearing
                used_mem, total_mem = get_gpu_memory_usage()
                st.sidebar.success(f"GPU Memory Cleared: {used_mem:.0f}MB / {total_mem:.0f}MB")
            
            # Step 4: The Canvas (Stable Diffusion XL Creates)
            status_text.text("Step 3/4: Generating 5 comic panels...")
            progress_bar.progress(0.75)
            
            comic_panels = generate_comic_panels(comic_data, steps, cfg_scale)
            
            if comic_panels is None or not any(comic_panels):
                st.error("Failed to generate comic panels. Please check your GPU setup.")
                return
            
            # Step 5: The Final Comic (Assembly)
            status_text.text("Step 4/4: Assembling final comic layout...")
            progress_bar.progress(1.0)
            
            # Extract dialogues for layout
            dialogues = [panel.get('dialogue', '') for panel in panels_data]
            
            final_comic = create_comic_layout(comic_panels, dialogues)
            
            if final_comic is None:
                st.error("Failed to create comic layout.")
                return
            
            # Display the final result
            status_text.text("✅ 5-panel comic generated successfully!")
            st.success("Your comic is ready!")
            
            # Display the final comic
            st.image(final_comic, caption="Your AI-Generated 5-Panel Comic", use_column_width=True)
            
            # Display individual panels
            st.markdown("### Individual Panels:")
            cols = st.columns(3)
            for i, (panel, dialogue) in enumerate(zip(comic_panels, dialogues)):
                if panel is not None:
                    with cols[i % 3]:
                        st.image(panel, caption=f"Panel {i+1}: {dialogue}", use_column_width=True)
            
            # Option to download the comic
            img_data = utils.create_download_link(final_comic)
            
            st.download_button(
                label="Download Complete Comic",
                data=img_data,
                file_name="5_panel_comic.png",
                mime="image/png",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your system setup and try again.")
            # Clean up GPU memory on error
            clear_gpu_memory()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>AI Comic Factory - Powered by Llama 3 & Stable Diffusion XL</p>
        <p>🎨 Generates 5-panel comics with consistent style and narrative flow</p>
        <p>📝 Token-optimized prompts (under 77 tokens) for best image quality</p>
        <p>Make sure Ollama is running locally on port 11434</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
