import streamlit as st
from story_generator1 import StoryGenerator
import json
import re

# Initialize the story generator
model_path = r"D:\Intel\models\Mistral-Nemo-Instruct-2407-Q3_K_L.gguf"
generator = StoryGenerator(model_path=model_path)

# Streamlit UI
st.title("ComicCrafter - Story Generator")

# Sidebar for input
with st.sidebar:
    st.header("Story Details")
    story_prompt = st.text_area("Story Concept", "A teenage inventor discovers a portal to a parallel universe...")
    genre = st.selectbox("Genre", ["Superhero", "Science Fiction", "Fantasy", "Mystery", "Adventure", "Comedy", "Drama"])
    max_tokens = st.slider("Maximum Length (tokens)", 1000, 4000, 2500, 100)

# Initialize session state
if "comic_story" not in st.session_state:
    st.session_state.comic_story = None
    st.session_state.raw_response = None

# Generate comic story
if st.button("Generate Comic Story"):
    with st.spinner("Generating your comic book story... (This might take a while on CPU)"):
        story = generator.generate_comic_story(story_prompt, genre)
        st.session_state.comic_story = story
        st.session_state.raw_response = story

# Function to manually extract panels for debugging
def manually_extract_panels(story_text):
    panels_found = []
    panel_pattern = r"\[PANEL\s+(\d+):\s+([^\]]+)\]"
    matches = re.findall(panel_pattern, story_text)
    for match in matches:
        panels_found.append(f"Panel {match[0]}: {match[1]}")
    return panels_found

# Display comic story
if st.session_state.comic_story:
    st.subheader("Generated Comic Book Story")
    
    # Get raw text
    if isinstance(st.session_state.raw_response, dict) and 'choices' in st.session_state.raw_response:
        raw_text = st.session_state.raw_response['choices'][0]['text']
    else:
        raw_text = st.session_state.raw_response
    
    # Debug: Show what panels we can find manually
    with st.expander("Debug: Panels Found in Raw Text"):
        panels = manually_extract_panels(raw_text)
        if panels:
            for panel in panels:
                st.write(panel)
        else:
            st.write("No panels found with the pattern [PANEL X: description]")
    
    # Parse the story
    parsed_story = generator.parse_story(st.session_state.comic_story)
    
    # Display each section
    if parsed_story:
        for section, panels in parsed_story.items():
            st.subheader(section.upper())
            
            # Display each panel in the section
            for panel in panels:
                with st.expander(f"Panel {panel['number']}"):
                    # Description
                    st.markdown("**Description:**")
                    st.write(panel['description'])
                    
                    # Narration
                    if panel['narration']:
                        st.markdown("**Narration:**")
                        for narr in panel['narration']:
                            st.write(narr)
                    
                    # Dialogues
                    if panel['dialogues']:
                        st.markdown("**Dialogues:**")
                        for dialogue in panel['dialogues']:
                            if dialogue['type'] == 'thought':
                                st.write(f"💭 {dialogue['character']}: {dialogue['text']}")
                            else:
                                st.write(f"💬 {dialogue['character']}: {dialogue['text']}")
    else:
        st.error("Failed to parse the story structure")
        
        # Show raw text for manual inspection
        st.subheader("Raw Generated Story")
        st.write(raw_text)
        
        # Try to extract parts manually for fallback
        parts_found = re.findall(r"Part\s+\d+:\s+([^\n]+)", raw_text)
        if parts_found:
            st.write("Parts found in raw text:")
            for part in parts_found:
                st.write(f"- {part}")