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
    Call the Llama 3 API to generate panel description and dialogue.
    
    Args:
        user_idea (str): The user's comic idea
        
    Returns:
        dict: JSON response containing panel_description and dialogue
    """
    prompt = f"""You are a creative comic book writer. Based on the following idea, generate a single panel for a comic. Your response MUST be a single JSON object containing two keys: 'panel_description' and 'dialogue'. The description should be a vivid, detailed visual scene. The dialogue should be a single, impactful line for the character.

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
        comic_data = utils.validate_json_response(llm_response)
        
        if comic_data is None:
            st.error("Invalid response format from Llama 3. Please try again.")
            return None
        
        return comic_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Llama 3 API: {utils.format_error_message(e)}")
        return None


def forge_master_prompt(panel_description):
    """
    Create the master prompt for Stable Diffusion by combining the LLM's description
    with predefined artistic keywords.
    
    Args:
        panel_description (str): The panel description from Llama 3
        
    Returns:
        str: The master prompt for Stable Diffusion
    """
    master_prompt = f"cinematic comic book art, {panel_description}, {config.ARTISTIC_KEYWORDS}"
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
            latent_image = base(
                prompt=master_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                output_type="latent"
            ).images[0]
        
        # Refine with refiner model
        with st.spinner("Refining image..."):
            final_image = refiner(
                prompt=master_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                image=latent_image
            ).images[0]
        
        return final_image
    
    except Exception as e:
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


def main():
    """
    Main Streamlit application function.
    """
    # Step 1: The Spark (User Input & Settings UI)
    st.title(config.APP_TITLE)
    st.markdown("Transform your ideas into stunning comic book panels!")
    
    # Main input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_idea = st.text_area(
            "Enter your comic idea:",
            placeholder="A robot detective solves a case in a rainy, neon-lit city...",
            height=150
        )
        
        generate_button = st.button("Generate Panel", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### Example Ideas:")
        st.markdown("""
        - A superhero soaring through clouds at sunset
        - A wizard casting a spell in an ancient library
        - A space explorer discovering alien ruins
        - A detective finding clues in a dark alley
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
    
    # Add generation time estimate
    estimated_time = utils.estimate_generation_time(steps, has_refiner=True)
    st.sidebar.info(f"Estimated generation time: {estimated_time}")
    
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
            status_text.text("Step 1/4: Generating story with Llama 3...")
            progress_bar.progress(0.25)
            
            with st.spinner("Llama 3 is writing your comic script..."):
                comic_data = call_llama3_api(user_idea)
            
            if comic_data is None:
                st.error("Failed to generate comic script. Please check if Ollama is running.")
                return
            
            panel_description = comic_data.get("panel_description", "")
            dialogue = comic_data.get("dialogue", "")
            
            if not panel_description or not dialogue:
                st.error("Invalid response from Llama 3. Please try again.")
                return
            
            # Display the generated script
            st.success("Comic script generated!")
            with st.expander("View Generated Script", expanded=True):
                st.markdown(f"**Panel Description:** {panel_description}")
                st.markdown(f"**Dialogue:** {dialogue}")
            
            # Step 3: The Prompt Forge (Backend Logic)
            status_text.text("Step 2/4: Forging master prompt...")
            progress_bar.progress(0.50)
            
            master_prompt = forge_master_prompt(panel_description)
            
            with st.expander("View Master Prompt"):
                st.code(master_prompt)
            
            # Step 4: The Canvas (Stable Diffusion XL Creates)
            status_text.text("Step 3/4: Generating artwork...")
            progress_bar.progress(0.75)
            
            comic_image = generate_comic_image(master_prompt, steps, cfg_scale)
            
            if comic_image is None:
                st.error("Failed to generate comic image. Please check your GPU setup.")
                return
            
            # Step 5: The Final Panel (Assembly)
            status_text.text("Step 4/4: Adding dialogue and finalizing...")
            progress_bar.progress(1.0)
            
            final_panel = add_dialogue_to_image(comic_image, dialogue)
            
            # Display the final result
            status_text.text("✅ Comic panel generated successfully!")
            st.success("Your comic panel is ready!")
            
            # Display the final comic panel
            st.image(final_panel, caption="Your AI-Generated Comic Panel", use_column_width=True)
            
            # Option to download the image
            img_data = utils.create_download_link(final_panel)
            
            st.download_button(
                label="Download Comic Panel",
                data=img_data,
                file_name="comic_panel.png",
                mime="image/png",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your system setup and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>AI Comic Factory - Powered by Llama 3 & Stable Diffusion XL</p>
        <p>Make sure Ollama is running locally on port 11434</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
