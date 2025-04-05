import streamlit as st
from llama_cpp import Llama
import re

class StoryGenerator:
    def __init__(self, model_path=r"D:\Intel\models\Mistral-Nemo-Instruct-2407-Q3_K_L.gguf", n_ctx=2048, n_threads=8):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads
        )
    
    def generate_comic_story(self, prompt, genre):
        system_prompt = f"""You are an expert comic book writer specializing in {genre} stories. 
        Create a compelling 4-part comic book story based on this prompt: "{prompt}"
        
        The story must have these four distinct parts:
        1. INTRODUCTION: Set the scene, introduce main characters and the central conflict
        2. STORYLINE: Develop the plot with rising action and obstacles
        3. CLIMAX: Present the most intense moment where the conflict peaks
        4. MORAL: Conclude with a resolution and the lesson or meaning of the story
        
        Format the story exactly as follows:

        Part 1: Introduction
        
        [PANEL 1: Visual description here]
        NARRATION: Narration text here...
        CHARACTER NAME: Dialogue text here...
        CHARACTER NAME (THOUGHT): Thought bubble text here...
        
        [PANEL 2: Another visual description]
        ...and so on for each panel in the Introduction.
        
        Part 2: Storyline
        
        [PANEL 1: Visual description here]
        ...and so on for each panel in the Storyline.
        
        Part 3: Climax
        
        [PANEL 1: Visual description here]
        ...and so on for each panel in the Climax.
        
        Part 4: Moral
        
        [PANEL 1: Visual description here]
        ...and so on for each panel in the Moral.
        """
        
        full_prompt = f"{system_prompt}\n\nAssistant:"
        
        response = self.llm(
            full_prompt,
            max_tokens=2500,
            temperature=0.7,
        )
        return response

    def parse_story(self, response):
        try:
            # Get the story text from the response
            if isinstance(response, dict) and 'choices' in response:
                story_text = response['choices'][0]['text']
            else:
                story_text = response
                
            # Print raw story text for debugging
            st.write("Parsing story from raw text:")
            with st.expander("View Raw Text"):
                st.code(story_text)
            
            # Dictionary to store all parts and their panels
            story_parts = {}
            
            # Pattern to identify story parts
            part_pattern = r"Part\s+\d+:\s+([^\n]+)"
            parts = re.split(part_pattern, story_text)
            
            # First element is empty or introduction text, skip it
            if parts and not parts[0].strip():
                parts.pop(0)
                
            # Process each part and its content
            for i in range(0, len(parts), 2):
                if i+1 >= len(parts):
                    break
                    
                part_name = parts[i].strip().lower()
                part_content = parts[i+1].strip()
                
                # Process panels in this part
                panel_pattern = r"\[PANEL\s+(\d+):\s+([^\]]+)\]"
                panel_blocks = re.split(panel_pattern, part_content)
                
                # Process panel blocks
                panels = []
                j = 1  # Start at 1 to skip any text before the first panel
                while j < len(panel_blocks)-2:
                    panel_num = panel_blocks[j].strip()
                    panel_desc = panel_blocks[j+1].strip()
                    panel_content = panel_blocks[j+2].strip()
                    
                    # Create panel object
                    panel = {
                        'number': panel_num,
                        'description': panel_desc,
                        'dialogues': [],
                        'narration': []
                    }
                    
                    # Process the content lines
                    lines = panel_content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Narration
                        if line.startswith("NARRATION:"):
                            narr_text = line.replace("NARRATION:", "").strip()
                            panel['narration'].append(narr_text)
                        # Thought bubble
                        elif "(THOUGHT)" in line or "(THOUGHT BUBBLE)" in line:
                            char_text = line.split(':', 1)
                            if len(char_text) == 2:
                                char_name = char_text[0].split('(')[0].strip()
                                text = char_text[1].strip()
                                panel['dialogues'].append({
                                    'type': 'thought',
                                    'character': char_name,
                                    'text': text
                                })
                        # Regular dialogue
                        elif ":" in line and not line.startswith("NARRATION:"):
                            char_text = line.split(':', 1)
                            if len(char_text) == 2:
                                char_name = char_text[0].strip()
                                text = char_text[1].strip()
                                panel['dialogues'].append({
                                    'type': 'speech',
                                    'character': char_name,
                                    'text': text
                                })
                    
                    panels.append(panel)
                    j += 3
                
                # If we have panels, add them to the story part
                if panels:
                    story_parts[part_name] = panels
                
            return story_parts
            
        except Exception as e:
            st.error(f"Error parsing story: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None