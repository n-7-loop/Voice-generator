import streamlit as st
from transformers import pipeline
import torch
import soundfile as sf
import numpy as np
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Voice-Text Converter",
    page_icon="🎤",
    layout="wide"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_stt_model():
    """Load the Speech-to-Text model once at startup."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        stt_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=device
        )
        return stt_pipe, device
    except Exception as e:
        st.error(f"Error loading STT model: {str(e)}")
        return None, None

@st.cache_resource
def load_tts_model():
    """Load the Text-to-Speech model once at startup."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_pipe = pipeline(
            "text-to-speech",
            model="facebook/mms-tts-eng",
            device=device
        )
        return tts_pipe, device
    except Exception as e:
        st.error(f"Error loading TTS model: {str(e)}")
        return None, None

def speech_to_text(audio_data, stt_pipe):
    """Convert speech to text using the STT model."""
    try:
        result = stt_pipe(audio_data)
        return result["text"]
    except Exception as e:
        return f"Error in STT: {str(e)}"

def text_to_speech(text, tts_pipe):
    """Convert text to speech using the TTS model."""
    try:
        result = tts_pipe(text)
        
        # Extract audio data and sampling rate
        audio_array = result["audio"]
        sampling_rate = result["sampling_rate"]
        
        # Convert to numpy array if it's a tensor
        if torch.is_tensor(audio_array):
            audio_array = audio_array.cpu().numpy()
        
        # Ensure the audio is in the correct format
        if len(audio_array.shape) > 1:
            audio_array = audio_array.squeeze()
        
        return audio_array, sampling_rate
    except Exception as e:
        st.error(f"Error in TTS: {str(e)}")
        return None, None

def save_audio_to_bytes(audio_array, sampling_rate):
    """Save audio array to bytes for playback."""
    try:
        buffer = BytesIO()
        sf.write(buffer, audio_array, sampling_rate, format='WAV')
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return None

def process_audio_file(audio_file, stt_pipe, tts_pipe):
    """Process uploaded audio file: STT -> display text -> TTS -> play audio."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Perform STT
        with st.spinner("Transcribing audio..."):
            transcribed_text = speech_to_text(tmp_file_path, stt_pipe)
        
        # Display transcribed text
        st.success("✅ Transcription complete!")
        st.text_area("Transcribed Text", transcribed_text, height=100)
        
        # Perform TTS on transcribed text
        with st.spinner("Generating speech from transcribed text..."):
            audio_array, sampling_rate = text_to_speech(transcribed_text, tts_pipe)
        
        if audio_array is not None:
            audio_bytes = save_audio_to_bytes(audio_array, sampling_rate)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")

def process_recorded_audio(audio_bytes, stt_pipe, tts_pipe):
    """Process recorded audio: STT -> display text -> TTS -> play audio."""
    try:
        # Save recorded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Perform STT
        with st.spinner("Transcribing recorded audio..."):
            transcribed_text = speech_to_text(tmp_file_path, stt_pipe)
        
        # Display transcribed text
        st.success("✅ Transcription complete!")
        st.text_area("Transcribed Text", transcribed_text, height=100, key="recorded_text")
        
        # Perform TTS on transcribed text
        with st.spinner("Generating speech from transcribed text..."):
            audio_array, sampling_rate = text_to_speech(transcribed_text, tts_pipe)
        
        if audio_array is not None:
            audio_bytes_output = save_audio_to_bytes(audio_array, sampling_rate)
            if audio_bytes_output:
                st.audio(audio_bytes_output, format="audio/wav")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.error(f"Error processing recorded audio: {str(e)}")

def process_text_input(text, tts_pipe):
    """Process text input: TTS -> play audio."""
    try:
        if not text.strip():
            st.warning("Please enter some text.")
            return
        
        # Perform TTS
        with st.spinner("Generating speech..."):
            audio_array, sampling_rate = text_to_speech(text, tts_pipe)
        
        if audio_array is not None:
            audio_bytes = save_audio_to_bytes(audio_array, sampling_rate)
            if audio_bytes:
                st.success("✅ Speech generated!")
                st.audio(audio_bytes, format="audio/wav")
        
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")

# Main app
def main():
    st.title("🎤 Voice-Text Converter")
    st.markdown("Convert speech to text and text to speech using AI models.")
    
    # Load models
    with st.spinner("Loading AI models... This may take a moment on first run."):
        stt_pipe, stt_device = load_stt_model()
        tts_pipe, tts_device = load_tts_model()
    
    if stt_pipe is None or tts_pipe is None:
        st.error("Failed to load models. Please refresh the page.")
        return
    
    st.success(f"✅ Models loaded successfully! Running on: {stt_device.upper()}")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio", "✍️ Text to Speech"])
    
    # Tab 1: Upload Audio File
    with tab1:
        st.subheader("Upload Audio File")
        st.markdown("Upload an audio file to transcribe it to text, then hear it spoken back.")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            key="upload"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")
            
            if st.button("Process Audio File", key="process_upload"):
                process_audio_file(uploaded_file, stt_pipe, tts_pipe)
    
    # Tab 2: Record Audio
    with tab2:
        st.subheader("Record Audio")
        st.markdown("Record your voice to transcribe it to text, then hear it spoken back.")
        
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("Process Recording", key="process_recording"):
                process_recorded_audio(audio_bytes, stt_pipe, tts_pipe)
    
    # Tab 3: Text to Speech
    with tab3:
        st.subheader("Text to Speech")
        st.markdown("Enter text to convert it to speech.")
        
        text_input = st.text_area(
            "Enter text",
            placeholder="Type something here...",
            height=150,
            key="text_input"
        )
        
        if st.button("Generate Speech", key="generate_speech"):
            process_text_input(text_input, tts_pipe)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This app uses Whisper (STT) and MMS-TTS (TTS) models from Hugging Face. "
        "All processing is done locally without external API calls."
    )
    
    # Model info in sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown("""
        **Speech-to-Text (STT):**
        - Model: OpenAI Whisper Tiny
        - Fast and efficient for transcription
        
        **Text-to-Speech (TTS):**
        - Model: Facebook MMS-TTS English
        - Natural voice synthesis
        
        **Device:** {}
        """.format(stt_device.upper()))
        
        st.markdown("---")
        st.markdown("### 📝 How to Use")
        st.markdown("""
        1. **Upload Audio:** Upload a file and click 'Process'
        2. **Record Audio:** Click the mic icon to record
        3. **Text to Speech:** Type text and click 'Generate Speech'
        """)

if __name__ == "__main__":
    main()
