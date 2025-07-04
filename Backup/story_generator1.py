import streamlit as st
from llama_cpp import Llama

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
        
        Format each part with a clear heading and include vivid descriptions that would work well with comic book illustrations.
        Use comic-style dialogue with speech bubbles, thought bubbles, and narration boxes.
        Include specific image description suggestions for key moments in [PANEL: description] format.
        """
        
        full_prompt = f"{system_prompt}\n\nAssistant:"
        
        response = self.llm(
            full_prompt,
            max_tokens=1200,
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
            
            # Dictionary to store all parts and their panels
            story_parts = {}
            current_part = None
            current_panels = []
            
            # Split by lines and process
            lines = story_text.split('\n')
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and dividers
                if not line or line == '---':
                    continue
                    
                # Check for section headers
                if '**[' in line and ']**' in line:
                    # If we were processing a previous section, save it
                    if current_part:
                        story_parts[current_part] = current_panels
                    
                    # Start new section
                    current_part = line.replace('**[', '').replace(']**', '').lower()
                    current_panels = []
                    continue
                
                # Process panels and their content
                if line.startswith('*PANEL'):
                    # Start new panel
                    panel_num = line.split(':')[0].replace('*PANEL', '').strip()
                    panel_content = {
                        'number': panel_num,
                        'description': '',
                        'dialogues': [],
                        'narration': []
                    }
                    current_panels.append(panel_content)
                
                # Process different types of content
                elif current_panels:  # Make sure we have a current panel
                    current_panel = current_panels[-1]  # Get the last panel
                    
                    if line.startswith('NARRATION BOX:'):
                        current_panel['narration'].append(line.replace('NARRATION BOX:', '').strip())
                    elif 'THOUGHT BUBBLE' in line:
                        current_panel['dialogues'].append({
                            'type': 'thought',
                            'character': line.split('(THOUGHT BUBBLE)')[0].strip(),
                            'text': line.split(':')[-1].strip()
                        })
                    elif ':' in line and '(' not in line:  # Regular dialogue
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            current_panel['dialogues'].append({
                                'type': 'speech',
                                'character': parts[0].strip(),
                                'text': parts[1].strip()
                            })
                    elif not line.startswith('*'):  # Description text
                        if current_panel['description']:
                            current_panel['description'] += ' ' + line
                        else:
                            current_panel['description'] = line
            
            # Save the last section
            if current_part and current_panels:
                story_parts[current_part] = current_panels
            
            return story_parts
        
        except Exception as e:
            print(f"Error parsing story: {str(e)}")
            return None
