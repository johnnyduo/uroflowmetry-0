# combined_app.py

import streamlit as st
import uvicorn
import threading
import nest_asyncio
from fastapi import FastAPI, UploadFile, File
import asyncio
from api import app as fastapi_app  # Import your existing FastAPI app
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import io
from datetime import datetime

# Configuration
API_PORT = 8000
API_URL = f"http://localhost:{API_PORT}"

# Initialize FastAPI in background
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=API_PORT)

# Start FastAPI in a separate thread
thread = threading.Thread(target=run_fastapi)
thread.daemon = True
thread.start()

# Streamlit Interface
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
        st.error("Could not connect to the API. Please wait while the API starts...")
        return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Uroflowmetry Analysis",
        page_icon="🌊",
        layout="wide"
    )
    
    st.title("Sound-Based Uroflowmetry Analysis")
    st.write("Upload an audio file to generate uroflowmetry graph")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3'],
        help="Upload a WAV or MP3 file of urination sound"
    )
    
    if uploaded_file is not None:
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

if __name__ == "__main__":
    main()