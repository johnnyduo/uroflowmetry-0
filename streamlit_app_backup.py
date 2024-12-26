import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import io
from matplotlib.colors import LinearSegmentedColormap

def process_audio(audio_file):
    try:
        # Load audio file with librosa, ensuring we get a valid signal
        y, sr = librosa.load(audio_file, sr=None, duration=60)  # limit to 60 seconds
        
        # Ensure we have data
        if len(y) == 0:
            raise ValueError("No audio data found in file")
        
        # Calculate frame-wise RMS energy
        frame_length = int(sr * 0.1)  # 100ms frames
        hop_length = int(frame_length / 2)  # 50% overlap
        
        # Calculate RMS energy for each frame
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Create time array
        time = np.linspace(0, len(y)/sr, len(rms))
        
        # Smooth the flow rate curve
        flow_rate = np.interp(rms, (rms.min(), rms.max()), (0, 20))
        flow_rate = np.convolve(flow_rate, np.ones(5)/5, mode='same')
        
        return time, flow_rate
    
    except Exception as e:
        st.error(f"Error in audio processing: {str(e)}")
        return None, None

def calculate_parameters(time, flow_rate):
    try:
        max_flow = np.max(flow_rate)
        avg_flow = np.mean(flow_rate)
        voiding_duration = time[-1]
        voided_volume = np.trapz(flow_rate, time)
        time_to_max = time[np.argmax(flow_rate)]
        
        # Safely calculate flow at 2 seconds
        idx_2s = np.where(time >= 2.0)[0]
        flow_at_2s = flow_rate[idx_2s[0]] if len(idx_2s) > 0 else 0
        acceleration = flow_at_2s / 2.0 if flow_at_2s > 0 else 0
        
        return {
            "Maximum Flow Rate": f"{max_flow:.2f} ml/s",
            "Average Flow Rate": f"{avg_flow:.2f} ml/s",
            "Voiding Duration": f"{voiding_duration:.2f} s",
            "Voided Volume": f"{voided_volume:.2f} ml",
            "Time to Max Flow": f"{time_to_max:.2f} s",
            "Flow at 2 Seconds": f"{flow_at_2s:.2f} ml/s",
            "Acceleration": f"{acceleration:.2f} ml/s²"
        }
    except Exception as e:
        st.error(f"Error in parameter calculation: {str(e)}")
        return None

def create_uroflow_graph(time, flow_rate, parameters):
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create gradient effect
        colors = ['#1f77b4', '#2ecc71', '#3498db', '#9b59b6']
        gradient_cmap = LinearSegmentedColormap.from_list('custom', colors)
        
        # Plot with gradient
        ax.plot(time, flow_rate, color=colors[0], linewidth=2, label='Flow Rate')
        
        # Customize plot
        ax.set_xlim(0, max(time))
        ax.set_ylim(0, max(flow_rate) * 1.2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Flow Rate (ml/s)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title('Sound-Based Uroflowmetry Graph', fontsize=14, pad=20)
        
        # Add parameters box
        if parameters:
            param_text = '\n'.join([f"{k}: {v}" for k, v in parameters.items()])
            plt.text(1.02, 0.98, param_text,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                    verticalalignment='top',
                    fontsize=10)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error in graph creation: {str(e)}")
        return None

def main():
    st.title("Sound-Based Uroflowmetry Analysis")
    st.write("Upload an audio file to generate uroflowmetry graph")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
    
    if uploaded_file is not None:
        try:
            # Add a status message
            status = st.empty()
            status.text("Processing audio file...")
            
            # Process audio file
            time, flow_rate = process_audio(uploaded_file)
            
            if time is not None and flow_rate is not None and len(time) > 0 and len(flow_rate) > 0:
                # Calculate parameters
                parameters = calculate_parameters(time, flow_rate)
                
                if parameters:
                    # Create and display graph
                    fig = create_uroflow_graph(time, flow_rate, parameters)
                    if fig:
                        status.empty()
                        st.pyplot(fig)
                        
                        # Display parameters in a nice format
                        st.subheader("Uroflowmetry Parameters")
                        col1, col2 = st.columns(2)
                        
                        for i, (key, value) in enumerate(parameters.items()):
                            if i % 2 == 0:
                                col1.metric(key, value)
                            else:
                                col2.metric(key, value)
                    else:
                        status.error("Failed to create graph")
                else:
                    status.error("Failed to calculate parameters")
            else:
                status.error("No valid data extracted from audio file")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please try with a different audio file")

if __name__ == "__main__":
    main()