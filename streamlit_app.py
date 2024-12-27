# app.py

import os
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import io
from datetime import datetime

# Configuration with fallback
try:
    API_URL = st.secrets["api_url"]
except KeyError:
    API_URL = "https://uroflowmetry-api.onrender.com"  # Replace with your actual Render URL
    if not st.session_state.get("api_warning_shown"):
        st.warning("⚠️ Using default API URL. For production, please set up secrets in Streamlit Cloud.")
        st.session_state.api_warning_shown = True

def create_custom_gradient():
    """Create custom gradient colormap for the graph"""
    colors = ['#1f77b4', '#2ecc71', '#3498db', '#9b59b6']
    return LinearSegmentedColormap.from_list('custom', colors)

def create_uroflow_graph(time, flow_rate, parameters):
    """Create uroflowmetry graph with parameters"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create gradient effect
        gradient_cmap = create_custom_gradient()
        
        # Create line segments
        points = np.array([time, flow_rate]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create LineCollection
        lc = LineCollection(segments, cmap=gradient_cmap)
        lc.set_array(np.linspace(0, 1, len(time)))
        
        # Add collection to axis
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

def process_audio_file(uploaded_file):
    """Process audio file through API"""
    try:
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
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Please check if the API is available.")
        if st.button("Show API URL"):
            st.code(API_URL)
        return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
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
    if file is None:
        return False, "No file uploaded"
        
    # Check file size (10MB limit)
    if file.size > 10 * 1024 * 1024:
        return False, "File size too large (max 10MB)"
        
    # Check file extension
    if not file.name.lower().endswith(('.wav', '.mp3')):
        return False, "Invalid file format (only WAV or MP3)"
        
    return True, ""

def main():
    # Page configuration
    st.set_page_config(
        page_title="Uroflowmetry Analysis",
        page_icon="🌊",
        layout="wide"
    )
    
    # Title and description
    st.title("Sound-Based Uroflowmetry Analysis")
    st.write("Upload an audio file to generate uroflowmetry graph")
    
    # Debug information in development
    if st.secrets.get("dev_mode"):
        with st.expander("🔧 Debug Information"):
            st.write("API URL:", API_URL)
            st.write("API Health:", "✅ Online" if check_api_health() else "❌ Offline")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API service is not available. Please try again later.")
        if st.button("Retry Connection"):
            st.experimental_rerun()
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3'],
        help="Upload a WAV or MP3 file of urination sound (max 10MB)"
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
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please try with a different audio file")
            if st.secrets.get("dev_mode"):
                st.exception(e)

if __name__ == "__main__":
    main()