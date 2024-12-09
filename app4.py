import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PIL import Image
import io
import base64
import cv2
import numpy as np
from gtts import gTTS
import os

# Configure Google Generative AI and Langchain
def initialize_google_llm():
    """Initialize Google Generative AI model."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

# Configure Google Generative AI
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# def image_to_base64(image):
#     """Convert PIL Image to base64 encoded string."""
#     buffered = io.BytesIO()
#     image.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_scene_description(image):
    """Generate a detailed scene description using Google's Generative AI."""
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content([
        """You are a compassionate AI accessibility guide, transforming visual 
        content into a comprehensive mental map for a visually impaired individual.

Craft a description that:
- Provides precise spatial understanding
- Appeals to multiple senses
- Prioritizes practical, safety-related context
- Enables independent navigation and comprehension

Key description elements:
1. Precise environment and setting
2. Significant objects and their spatial relationships
3. Potential interactions or activities
4. Sensory details (texture, implied sounds, temperature)
5. Safety-critical information

Approach the description as a trusted guide:
- Walk the listener through the scene
- Offer actionable, context-rich insights
- Communicate with clarity and warmth
- Keep it concise

Transform visual information into an empowering, 
rich mental representation that supports understanding and independence.""",
        image
    ])
    return response.text


def object_detection(image):
    """Generate a list of objects from an image using Gemini API."""
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content([
        """You are an AI designed to analyze images and list all significant objects visible in the scene. 
        Provide the list of objects (in bullet points) without additional context or description.""",
        image
    ])
    # The response should now be a simple, clean list of objects
    return response.text.strip()

def text_to_speech(text, language='en-us'):
    """Convert text to speech using gTTS."""
    try:
        # Create a temporary MP3 file
        tts = gTTS(text=text, lang=language)
        audio_file = "output.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Text-to-speech conversion error: {e}")
        return None
    

def generate_task_guidance(scene_description, llm):
    """Generate personalized task guidance using Langchain."""
    task_prompt = PromptTemplate(
        input_variables=["scene_description"],
        template="""Based on the following scene description, provide comprehensive, 
        practical guidance for a visually impaired person:

        1. Identify key objects and their potential interactions
        2. Suggest safe navigation strategies
        3. Provide context-specific task recommendations
        4. Highlight important details for awareness and safety

        keep it clear and concise.

        Scene Description: {scene_description}
        
        Guidance:
        """
    )

    task_chain = LLMChain(
        llm=llm,
        prompt=task_prompt
    )

    task_guidance = task_chain.run(scene_description=scene_description)
    return task_guidance

def main():
    st.set_page_config(
        page_title="AI Assistive Vision Companion", 
        page_icon="👁️", 
        layout="wide"
    )

    st.title("👁️ AI Assistive Vision Companion")
    st.write("Upload an image to get detailed scene understanding and personalized guidance.")

    # Initialize LLM
    llm = initialize_google_llm()

    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'], 
        help="Upload an image to analyze"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Process image with different features
        if st.button("Analyze Image"):
            with st.spinner('Analyzing image...'):
                # Scene Description
                scene_description = generate_scene_description(image)
                st.subheader("Scene Description")
                st.write(scene_description)

                # Text-to-Speech for Scene Description
                audio_file = text_to_speech(scene_description)
                if audio_file:
                    st.audio(audio_file, format='audio/mp3')
                    os.remove(audio_file)  # Clean up temporary file


                # Object Detection
                detected_objects = object_detection(image)
                st.subheader("Detected Objects")
                st.write(detected_objects)

                # Task Guidance
                task_guidance = generate_task_guidance(scene_description, llm)
                st.subheader("Personalized Task Guidance")
                st.write(task_guidance)

    # Sidebar information
    st.sidebar.title("About the App")
    st.sidebar.info(""" 
    AI Assistive Vision Companion helps visually impaired individuals:
    🖼️ Understand image contents
    🔊 Convert descriptions to speech
    🤖 Detect objects and obstacles
    🤝 Provide personalized task guidance
    """)

if __name__ == "__main__":
    main()