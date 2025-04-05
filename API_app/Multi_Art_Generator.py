import streamlit as st
from langchain_groq import ChatGroq
from together import Together
from dotenv import load_dotenv
import os
import re
import json

# Load environment variables
load_dotenv()

# Check for API keys
groq_api_key = os.getenv("GROQ_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

if not groq_api_key or not together_api_key:
    st.error("Please set GROQ_API_KEY and TOGETHER_API_KEY in your .env file")
    st.stop()

# Set environment variables
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["TOGETHER_API_KEY"] = together_api_key

# Initialize models
llm = ChatGroq(
    model_name="llama3-70b-8192",  # Use an available model
    temperature=0,
    max_tokens=2000  # Increased for more detailed character descriptions and story
)
together_client = Together(api_key=os.environ["TOGETHER_API_KEY"])

# Art style definitions with detailed prompts
ART_STYLES = {
    "Manga": {
        "description": "Japanese manga style with clean lines and expressive characters",
        "prompt": "professional Japanese manga panel style, monochrome, clean line art, dynamic poses, expressive facial features, screentone shading, speech bubbles, high contrast, detailed backgrounds, professional inking technique"
    },
    "Comic Book": {
        "description": "American comic book style with bold colors and dynamic action",
        "prompt": "professional American comic book style, vibrant colors, bold outlines, dynamic action, superhero aesthetic, detailed backgrounds, ink and color, cross-hatching, professional comic art"
    },
    "Noir": {
        "description": "High contrast black and white with dramatic shadows",
        "prompt": "professional noir comic style, high contrast black and white, dramatic shadows, moody atmosphere, cinematic framing, stark lighting, heavy inks, Sin City style, dark and gritty, silhouettes"
    },
    "Watercolor": {
        "description": "Soft watercolor aesthetic with artistic brushwork",
        "prompt": "professional watercolor comic style, soft colors, flowing textures, painterly style, artistic brushstrokes, gentle transitions, muted palette, impressionistic, dreamy quality"
    },
    "Cartoon": {
        "description": "Animated cartoon style with bold lines and colors",
        "prompt": "professional cartoon style, bold outlines, bright colors, simplified forms, animated aesthetic, clean design, expressive faces, modern animation style, playful"
    }
}

def generate_structured_comic(prompt, genre, art_style):
    """Generate a structured comic with clear character sheets and panel descriptions"""
    messages = [
        ("system", f"""You are a professional comic storytelling AI specializing in {art_style} style. 
        
        Create a {genre} story with detailed character descriptions and 6 panels. Structure your response in this exact format:

        **CHARACTER SHEET:**
        
        [CHARACTER NAME]:
        - Name: [Full name]
        - Age: [Age]
        - Physical description: [Detailed physical appearance including height, build, hair, eyes, clothing, and distinctive features]
        - Personality traits: [Key personality characteristics]
        
        [REPEAT FOR EACH MAIN CHARACTER - include 2-3 characters]
        
        **COMIC PANELS:**
        
        Panel 1:
        - Description: [Detailed visual description of what's happening]
        - [Character]: [Dialogue with emotion indicators like *whispers* or *shouts*]
        
        [REPEAT FOR EACH PANEL - create exactly 6 panels]
        
        Important:
        - Ensure consistent character descriptions that will allow an AI to draw them consistently across panels
        - Include distinctive visual traits that can be maintained across panels
        - Make dialogue emotionally expressive with clear indicators
        - Describe action and movement clearly
        - Structure panels for dramatic pacing
        - Format strictly as above - no additional text"""),
        ("human", prompt)
    ]
    response = llm.invoke(messages)
    return response.content

def parse_character_sheet(text):
    """Parse character details into a structured format"""
    characters = []
    
    # Find the character sheet section
    character_sheet_match = re.search(r"\*\*CHARACTER SHEET:\*\*(.*?)(?=\*\*COMIC PANELS:|$)", text, re.DOTALL)
    
    if not character_sheet_match:
        return characters
    
    character_sheet = character_sheet_match.group(1).strip()
    
    # Split into individual characters
    character_blocks = re.split(r'\n\s*\n', character_sheet)
    
    for block in character_blocks:
        if not block.strip():
            continue
            
        character = {}
        
        # Extract character name from header
        name_match = re.search(r'^([^:]+):', block.strip(), re.MULTILINE)
        if name_match:
            character["full_name"] = name_match.group(1).strip()
        
        # Extract details
        name_match = re.search(r'- Name: (.*?)(?=\n|$)', block, re.MULTILINE)
        if name_match:
            character["name"] = name_match.group(1).strip()
            
        age_match = re.search(r'- Age: (.*?)(?=\n|$)', block, re.MULTILINE)
        if age_match:
            character["age"] = age_match.group(1).strip()
            
        physical_match = re.search(r'- Physical description: (.*?)(?=\n- |$)', block, re.DOTALL)
        if physical_match:
            character["physical"] = physical_match.group(1).strip()
            
        personality_match = re.search(r'- Personality traits: (.*?)(?=\n|$)', block, re.DOTALL)
        if personality_match:
            character["personality"] = personality_match.group(1).strip()
            
        if character:
            characters.append(character)
    
    return characters

def parse_panels(text):
    """Parse comic panels into a structured format"""
    panels = []
    
    # Find the panels section
    panels_match = re.search(r"\*\*COMIC PANELS:\*\*(.*?)$", text, re.DOTALL)
    
    if not panels_match:
        return panels
    
    panels_text = panels_match.group(1).strip()
    
    # Split into individual panels
    panel_blocks = re.split(r'Panel \d+:', panels_text)
    
    for i, block in enumerate(panel_blocks[1:], 1):  # Skip the first empty block
        if not block.strip():
            continue
            
        panel = {
            "number": i,
            "description": "",
            "dialogues": []
        }
        
        # Extract description
        desc_match = re.search(r'- Description: (.*?)(?=\n- |$)', block, re.DOTALL)
        if desc_match:
            panel["description"] = desc_match.group(1).strip()
        
        # Extract dialogues
        dialogue_matches = re.finditer(r'- ([^:]+): (.*?)(?=\n- |$)', block, re.DOTALL)
        for match in dialogue_matches:
            panel["dialogues"].append({
                "character": match.group(1).strip(),
                "text": match.group(2).strip()
            })
            
        panels.append(panel)
    
    return panels

def generate_panel_image_prompt(panel, characters, art_style):
    """Create a detailed prompt for image generation"""
    # Start with basic panel description
    prompt = f"{panel['description']}"
    
    # Add dialogue information
    if panel["dialogues"]:
        dialogue_text = "; ".join([f"{d['character']} saying '{d['text']}'" for d in panel["dialogues"]])
        prompt += f". With dialogue: {dialogue_text}"
    
    # Add character descriptions for consistency
    char_descriptions = []
    for character in characters:
        # Extract the character name that appears in this panel
        panel_chars = [d["character"] for d in panel["dialogues"]]
        desc_text = panel["description"].lower()
        
        # Check if character appears in this panel
        if any(character["name"].lower() in char.lower() for char in panel_chars) or character["name"].lower() in desc_text:
            char_desc = f"{character['name']}: {character['physical']}"
            char_descriptions.append(char_desc)
    
    if char_descriptions:
        prompt += f". Characters: {'; '.join(char_descriptions)}"
    
    # Add style-specific prompt details
    prompt += f". {ART_STYLES[art_style]['prompt']}"
    
    # Ensure the prompt isn't too long
    if len(prompt) > 950:
        prompt = prompt[:947] + "..."
        
    return prompt

def generate_image_for_panel(panel, characters, art_style):
    """Generate image using Together API"""
    try:
        prompt = generate_panel_image_prompt(panel, characters, art_style)
        
        response = together_client.images.generate(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell-Free",
            steps=4,
            height=512,
            width=512
        )
        
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        return None

# Set page config with dark theme
st.set_page_config(
    page_title="Comic-AI", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Apply the dark theme
st.markdown("""
<style>
    /* Dark theme */
    .main {
        background-color: #121212;
        color: #f0f0f0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    .stButton>button {
        background-color: #8954A8;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #9e68b9;
    }
    .stSelectbox>div>div {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    /* Comic panel styling */
    .comic-panel {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .panel-title {
        font-weight: bold;
        color: #8954A8;
        margin-bottom: 5px;
    }
    .panel-text {
        color: #f0f0f0;
        margin-bottom: 10px;
        font-family: 'Courier New', monospace;
    }
    /* Character sheet styling */
    .character-sheet {
        background-color: #2a2a2a;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 3px solid #8954A8;
    }
    .character-name {
        font-weight: bold;
        color: #8954A8;
        font-size: 1.2em;
        margin-bottom: 10px;
    }
    .character-detail {
        margin-bottom: 5px;
        color: #f0f0f0;
    }
    .main-header {
        font-size: 2.5rem;
        color: #8954A8;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .book-icon {
        font-size: 2.5rem;
        margin-right: 0.5rem;
    }
    /* For tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2a2a;
        color: #f0f0f0;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #8954A8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "comic_data" not in st.session_state:
    st.session_state.comic_data = None
if "characters" not in st.session_state:
    st.session_state.characters = []
if "panels" not in st.session_state:
    st.session_state.panels = []
if "panel_images" not in st.session_state:
    st.session_state.panel_images = {}
if "comic_history" not in st.session_state:
    st.session_state.comic_history = []
if "current_style" not in st.session_state:
    st.session_state.current_style = "Manga"

# Header
st.markdown('<div class="main-header"><span class="book-icon">📖</span> COMIC - AI</div>', unsafe_allow_html=True)

# Main UI
tab1, tab2 = st.tabs(["Create", "History"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### Story Prompt")
        story_prompt = st.text_area(
            "Story Prompt", 
            height=100,
            label_visibility="collapsed",
            placeholder="Enter your comic story idea here..."
        )
    
    with col2:
        st.markdown("#### Genre")
        genre = st.selectbox(
            "Genre", 
            ["Action", "Fantasy", "Sci-Fi", "Horror", "Mystery", "Romance", "Comedy", "Thriller"],
            label_visibility="collapsed"
        )
        
        st.markdown("#### Art Style")
        art_style = st.selectbox(
            "Art Style",
            list(ART_STYLES.keys()),
            index=list(ART_STYLES.keys()).index(st.session_state.current_style),
            label_visibility="collapsed",
            format_func=lambda x: f"{x} - {ART_STYLES[x]['description']}"
        )
        st.session_state.current_style = art_style
        
        # Generate button
        if st.button("Generate Comic", use_container_width=True):
            with st.spinner("Creating your comic story..."):
                try:
                    # Generate the comic content
                    comic_data = generate_structured_comic(story_prompt, genre, art_style)
                    
                    # Parse the structured data
                    characters = parse_character_sheet(comic_data)
                    panels = parse_panels(comic_data)
                    
                    # Save to session state
                    st.session_state.comic_data = comic_data
                    st.session_state.characters = characters
                    st.session_state.panels = panels
                    st.session_state.panel_images = {}
                    
                    st.success("Comic story created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating comic: {str(e)}")
    
    # Display generated comic if available
    if st.session_state.comic_data and st.session_state.characters and st.session_state.panels:
        # Character Sheet Section
        st.markdown("### Character Sheet")
        
        # Display characters in a grid
        cols = st.columns(len(st.session_state.characters))
        for i, character in enumerate(st.session_state.characters):
            with cols[i]:
                st.markdown(f"""
                <div class="character-sheet">
                    <div class="character-name">{character.get('full_name', character.get('name', 'Character'))}</div>
                    <div class="character-detail"><strong>Name:</strong> {character.get('name', 'Unknown')}</div>
                    <div class="character-detail"><strong>Age:</strong> {character.get('age', 'Unknown')}</div>
                    <div class="character-detail"><strong>Physical:</strong> {character.get('physical', 'No description')}</div>
                    <div class="character-detail"><strong>Personality:</strong> {character.get('personality', 'No description')}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Comic Panels Section
        st.markdown("### Story Panels")
        
        # Create rows of 3 panels each
        panel_rows = [st.session_state.panels[i:i+3] for i in range(0, len(st.session_state.panels), 3)]
        
        for row_idx, row_panels in enumerate(panel_rows):
            cols = st.columns(len(row_panels))
            
            for i, (col, panel) in enumerate(zip(cols, row_panels)):
                panel_idx = row_idx * 3 + i
                
                with col:
                    # Display panel image (or placeholder)
                    if panel_idx in st.session_state.panel_images:
                        st.image(st.session_state.panel_images[panel_idx], use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/512x512?text=Panel+Image", use_column_width=True)
                    
                    # Display panel details
                    st.markdown(f"""
                    <div class="comic-panel">
                        <div class="panel-title">Panel {panel['number']}</div>
                        <div class="panel-text">{panel['description']}</div>
                    """, unsafe_allow_html=True)
                    
                    # Display dialogues
                    for dialogue in panel['dialogues']:
                        st.markdown(f"""
                        <div class="panel-text"><strong>{dialogue['character']}:</strong> {dialogue['text']}</div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Generate image button
                    if st.button(f"Generate Image {panel['number']}", key=f"gen_img_{panel_idx}"):
                        with st.spinner(f"Generating image for panel {panel['number']}..."):
                            image_url = generate_image_for_panel(panel, st.session_state.characters, art_style)
                            if image_url:
                                st.session_state.panel_images[panel_idx] = image_url
                                st.success(f"Image for panel {panel['number']} generated!")
                                st.rerun()
        
        # Generate all images button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate All Images", use_container_width=True):
                with st.spinner("Generating all panel images..."):
                    for i, panel in enumerate(st.session_state.panels):
                        if i not in st.session_state.panel_images:
                            image_url = generate_image_for_panel(panel, st.session_state.characters, art_style)
                            if image_url:
                                st.session_state.panel_images[i] = image_url
                    
                    # Save to history if all images generated
                    if len(st.session_state.panel_images) == len(st.session_state.panels):
                        history_item = {
                            "prompt": story_prompt,
                            "genre": genre,
                            "art_style": art_style,
                            "comic_data": st.session_state.comic_data,
                            "characters": st.session_state.characters,
                            "panels": st.session_state.panels,
                            "images": st.session_state.panel_images.copy()
                        }
                        st.session_state.comic_history.append(history_item)
                    
                    st.success("All images generated!")
                    st.rerun()
        
        with col2:
            if st.button("Save to History", use_container_width=True):
                # Save to history
                history_item = {
                    "prompt": story_prompt,
                    "genre": genre,
                    "art_style": art_style,
                    "comic_data": st.session_state.comic_data,
                    "characters": st.session_state.characters,
                    "panels": st.session_state.panels,
                    "images": st.session_state.panel_images.copy()
                }
                st.session_state.comic_history.append(history_item)
                st.success("Comic saved to history!")

# History tab
with tab2:
    st.markdown("### Your Comic History")
    
    if not st.session_state.comic_history:
        st.info("You haven't created any comics yet. Go to the Create tab to get started!")
    else:
        for i, comic in enumerate(reversed(st.session_state.comic_history)):
            with st.expander(f"Comic #{len(st.session_state.comic_history)-i}: {comic['prompt'][:50]}...", expanded=False):
                st.markdown(f"**Genre:** {comic['genre']} | **Style:** {comic['art_style']}")
                st.markdown(f"**Prompt:** {comic['prompt']}")
                
                # Display characters
                st.markdown("#### Characters")
                char_cols = st.columns(len(comic['characters']))
                for j, character in enumerate(comic['characters']):
                    with char_cols[j]:
                        st.markdown(f"""
                        <div class="character-sheet">
                            <div class="character-name">{character.get('full_name', character.get('name', 'Character'))}</div>
                            <div class="character-detail"><strong>Name:</strong> {character.get('name', 'Unknown')}</div>
                            <div class="character-detail"><strong>Age:</strong> {character.get('age', 'Unknown')}</div>
                            <div class="character-detail"><strong>Physical:</strong> {character.get('physical', 'No description')}</div>
                            <div class="character-detail"><strong>Personality:</strong> {character.get('personality', 'No description')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display panels
                st.markdown("#### Panels")
                panel_rows = [comic['panels'][i:i+3] for i in range(0, len(comic['panels']), 3)]
                
                for row_idx, row_panels in enumerate(panel_rows):
                    cols = st.columns(len(row_panels))
                    
                    for j, (col, panel) in enumerate(zip(cols, row_panels)):
                        panel_idx = row_idx * 3 + j
                        
                        with col:
                            # Display panel image if available
                            if 'images' in comic and panel_idx in comic['images']:
                                st.image(comic['images'][panel_idx], use_column_width=True)
                            
                            # Display panel details
                            st.markdown(f"""
                            <div class="comic-panel">
                                <div class="panel-title">Panel {panel['number']}</div>
                                <div class="panel-text">{panel['description']}</div>
                            """, unsafe_allow_html=True)
                            
                            # Display dialogues
                            for dialogue in panel['dialogues']:
                                st.markdown(f"""
                                <div class="panel-text"><strong>{dialogue['character']}:</strong> {dialogue['text']}</div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)

# New Comic button in sidebar
with st.sidebar:
    if st.button("New Comic", use_container_width=True):
        st.session_state.comic_data = None
        st.session_state.characters = []
        st.session_state.panels = []
        st.session_state.panel_images = {}
        st.rerun()