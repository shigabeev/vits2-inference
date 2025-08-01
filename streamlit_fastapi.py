import streamlit as st
import requests

if __name__ == "__main__":
    st.title("Синтез речи на русском языке.")
    
    text = st.text_input(
        'Type text to synthesize (recommended min length 10 characters, max len 200 characters, one sentence!)',
        max_chars=1000)
    
    length_scale = st.slider("Choose speed:", 0.5, 2.0, value=0.9, step=0.05)
    
    if text.strip() and 5 <= len(text) < 1000:
        # Prepare the data to send to the FastAPI service
        data = {
            "text": text,
            "length_scale": length_scale
        }
        
        # Specify the URL of your FastAPI service
        url = "http://localhost:8000/synthesize/"
        
        # Send a POST request to the FastAPI service
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            # If request was successful, get the audio data from the response
            audio_data = response.content
            
            # Use Streamlit to display the audio
            st.audio(audio_data, format='audio/wav')
        else:
            # If there was an error, display it
            st.error(f"Error in TTS generation: {response.text}")
