# app.py

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import json
import io
import time
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"
DEBUG = True  # Set to False in production
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def debug_log(message):
    """Print debug messages if DEBUG is enabled"""
    if DEBUG:
        st.write(f"🔍 Debug: {message}")

def create_custom_gradient():
    """Create custom gradient colormap for the graph"""
    colors = ['#1f77b4', '#2ecc71', '#3498db', '#9b59b6']
    return LinearSegmentedColormap.from_list('custom', colors)

@st.cache_data(ttl=3600)
def get_api_info():
    """Get API model information with caching"""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def validate_audio_file(file):
    """Validate the uploaded audio file"""
    try:
        # Check file size
        if file.size > MAX_FILE_SIZE:
            return False, "File size too large (max 10MB)"
        
        # Check file extension
        if not file.name.lower().endswith(('.wav', '.mp3')):
            return False, "Invalid file format (only WAV or MP3)"
        
        # Check if file is empty
        if file.size == 0:
            return False, "File is empty"
        
        return True, ""
    except Exception as e:
        return False, f"File validation error: {str(e)}"

def process_audio_file(uploaded_file):
    """Process audio file through API"""
    try:
        if DEBUG:
            debug_log(f"File name: {uploaded_file.name}")
            debug_log(f"File size: {uploaded_file.size} bytes")
            debug_log(f"File type: {uploaded_file.type}")
        
        # Create files dictionary with proper file name and MIME type
        files = {
            'file': (
                uploaded_file.name,
                uploaded_file.getvalue(),
                'audio/wav' if uploaded_file.name.endswith('.wav') else 'audio/mp3'
            )
        }
        
        # Make API request with timeout
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            timeout=30
        )
        
        # Check response status
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get('detail', response.text)
            except:
                error_detail = response.text
            st.error(f"API Error ({response.status_code}): {error_detail}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Please check if the API server is running.")
        return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_uroflow_graph(time, flow_rate, parameters):
    """Create uroflowmetry graph with parameters"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create gradient effect
        gradient_cmap = create_custom_gradient()
        points = np.array([time, flow_rate]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Plot with gradient
        lc = plt.collections.LineCollection(segments, cmap=gradient_cmap)
        lc.set_array(np.linspace(0, 1, len(time)))
        ax.add_collection(lc)
        
        # Set plot limits and labels
        ax.set_xlim(min(time), max(time))
        ax.set_ylim(0, max(flow_rate) * 1.2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Flow Rate (ml/s)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title('Sound-Based Uroflowmetry Graph', fontsize=14, pad=20)
        
        # Add parameters box
        param_text = '\n'.join([f"{k}: {v}" for k, v in parameters.items()])
        plt.text(1.02, 0.98, param_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                verticalalignment='top',
                fontsize=10)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating graph: {str(e)}")
        return None

def display_parameters(parameters):
    """Display parameters in a two-column layout"""
    col1, col2 = st.columns(2)
    
    for i, (key, value) in enumerate(parameters.items()):
        if i % 2 == 0:
            col1.metric(key, value)
        else:
            col2.metric(key, value)

def save_results(time, flow_rate, parameters):
    """Save results to session state"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    result = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'time': time.tolist(),
        'flow_rate': flow_rate.tolist(),
        'parameters': parameters
    }
    
    st.session_state.history.append(result)

def show_history():
    """Show historical results"""
    if 'history' in st.session_state and st.session_state.history:
        with st.expander("📊 History"):
            for i, result in enumerate(st.session_state.history):
                st.write(f"### Analysis {i+1} - {result['timestamp']}")
                display_parameters(result['parameters'])
                st.write("---")

def main():
    # Page configuration
    st.set_page_config(
        page_title="Uroflowmetry Analysis",
        page_icon="🌊",
        layout="wide"
    )
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Title and description
    st.title("Sound-Based Uroflowmetry Analysis")
    st.write("Upload an audio file to generate uroflowmetry graph")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API service is not available. Please try again later.")
        return
    
    # Display API info
    api_info = get_api_info()
    if api_info:
        with st.expander("ℹ️ Model Information"):
            for key, value in api_info.items():
                st.write(f"**{key}:** {value}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3'],
        help="Upload a WAV or MP3 file of urination sound"
    )
    
    if uploaded_file is not None:
        # Validate file
        is_valid, error_message = validate_audio_file(uploaded_file)
        if not is_valid:
            st.error(error_message)
            return
            
        try:
            # Show processing status
            with st.spinner("Processing audio file..."):
                # Process file through API
                result = process_audio_file(uploaded_file)
                
                if result:
                    # Extract data from API response
                    flow_rate = np.array(result['flow_rate'])
                    time = np.array(result['time'])
                    parameters = result['parameters']
                    
                    # Save results
                    save_results(time, flow_rate, parameters)
                    
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["📈 Graph", "📊 Parameters"])
                    
                    with tab1:
                        # Create and display graph
                        fig = create_uroflow_graph(time, flow_rate, parameters)
                        if fig:
                            st.pyplot(fig)
                            
                            # Add download button for graph
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            buf.seek(0)
                            st.download_button(
                                label="📥 Download Graph",
                                data=buf,
                                file_name=f"uroflowmetry_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                    
                    with tab2:
                        # Display parameters
                        st.subheader("Uroflowmetry Parameters")
                        display_parameters(parameters)
                    
                    # Show history
                    show_history()
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            if DEBUG:
                st.exception(e)
            st.write("Please try with a different audio file")

if __name__ == "__main__":
    main()