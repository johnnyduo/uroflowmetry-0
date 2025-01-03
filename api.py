# api.py

import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter, butter, filtfilt
import io
from pydantic import BaseModel
from typing import List, Dict
import soundfile as sf

app = FastAPI(title="Uroflowmetry API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    flow_rate: List[float]
    parameters: Dict[str, str]
    time: List[float]

def calculate_rms(signal, frame_length, hop_length):
    """Calculate RMS manually without using librosa"""
    try:
        # Handle NaN and Inf values
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad the signal
        pad_length = frame_length - 1
        padded_signal = np.pad(signal, (pad_length // 2, pad_length - pad_length // 2))
        
        # Calculate frames
        n_frames = 1 + (len(signal) - frame_length) // hop_length
        frames = np.zeros((n_frames, frame_length))
        
        for i in range(n_frames):
            start = i * hop_length
            frames[i] = padded_signal[start:start + frame_length]
        
        # Calculate RMS with small epsilon to avoid division by zero
        rms = np.sqrt(np.mean(frames ** 2 + 1e-10, axis=1))
        
        # Handle any remaining NaN or Inf values
        rms = np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0)
        
        return rms
    except Exception as e:
        raise ValueError(f"RMS calculation failed: {str(e)}")

def apply_bandpass_filter(signal, sr, lowcut=20, highcut=2000):
    """Apply bandpass filter to the audio signal"""
    try:
        # Handle NaN and Inf values
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        nyquist = sr / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        # Handle any remaining NaN or Inf values
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        
        return filtered
    except Exception as e:
        raise ValueError(f"Bandpass filter failed: {str(e)}")

def process_audio(audio_bytes):
    """Process audio file and extract features"""
    try:
        # Load audio using soundfile
        with io.BytesIO(audio_bytes) as audio_io:
            y, sr = sf.read(audio_io)
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
        
        # Ensure we have audio data
        if len(y) == 0:
            raise ValueError("No audio data found")
            
        # Handle NaN and Inf values
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply bandpass filter
        y_filtered = apply_bandpass_filter(y, sr)
        
        # Calculate frame parameters
        frame_length = int(sr * 0.1)  # 100ms frames
        hop_length = int(frame_length / 2)
        
        # Calculate RMS
        rms = calculate_rms(y_filtered, frame_length, hop_length)
        
        # Ensure we have RMS values
        if len(rms) == 0:
            raise ValueError("No RMS values calculated")
        
        # Smooth the RMS curve
        if len(rms) > 3:
            window_length = min(15, len(rms)-2 if len(rms) % 2 == 0 else len(rms)-1)
            if window_length > 2:
                rms_smoothed = savgol_filter(rms, window_length, 3)
            else:
                rms_smoothed = rms
        else:
            rms_smoothed = rms
            
        # Handle NaN and Inf values in smoothed data
        rms_smoothed = np.nan_to_num(rms_smoothed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add small epsilon to avoid division by zero
        rms_smoothed = rms_smoothed + 1e-10
        
        # Scale to realistic flow rate values (0-50 ml/s)
        scaler = MinMaxScaler(feature_range=(0, 50))
        flow_rate = scaler.fit_transform(rms_smoothed.reshape(-1, 1)).flatten()
        
        # Create time array
        duration = len(y) / sr
        time = np.linspace(0, duration, len(flow_rate))
        
        # Final check for any remaining NaN/Inf values
        flow_rate = np.nan_to_num(flow_rate, nan=0.0, posinf=50.0, neginf=0.0)
        time = np.nan_to_num(time, nan=0.0, posinf=duration, neginf=0.0)
        
        # Ensure all values are finite
        if not np.all(np.isfinite(flow_rate)) or not np.all(np.isfinite(time)):
            raise ValueError("Invalid values in processed data")
            
        return flow_rate, time
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

def calculate_parameters(time, flow_rate):
    """Calculate uroflowmetry parameters"""
    try:
        # Handle any potential NaN/Inf values
        flow_rate = np.nan_to_num(flow_rate, nan=0.0, posinf=50.0, neginf=0.0)
        time = np.nan_to_num(time, nan=0.0)
        
        max_flow = float(np.max(flow_rate))
        avg_flow = float(np.mean(flow_rate))
        voiding_duration = float(time[-1])
        voided_volume = float(np.trapz(flow_rate, time))
        time_to_max = float(time[np.argmax(flow_rate)])
        
        # Calculate flow at 2 seconds
        idx_2s = np.where(time >= 2.0)[0]
        flow_at_2s = float(flow_rate[idx_2s[0]]) if len(idx_2s) > 0 else 0.0
        acceleration = flow_at_2s / 2.0 if flow_at_2s > 0 else 0.0
        
        # Ensure all values are finite and within reasonable ranges
        max_flow = min(max_flow, 50.0)
        avg_flow = min(avg_flow, 50.0)
        voiding_duration = min(voiding_duration, 300.0)  # Max 5 minutes
        voided_volume = min(voided_volume, 1000.0)  # Max 1000 ml
        
        parameters = {
            "Maximum Flow Rate": f"{max_flow:.2f} ml/s",
            "Average Flow Rate": f"{avg_flow:.2f} ml/s",
            "Voiding Duration": f"{voiding_duration:.2f} s",
            "Voided Volume": f"{voided_volume:.2f} ml",
            "Time to Max Flow": f"{time_to_max:.2f} s",
            "Flow at 2 Seconds": f"{flow_at_2s:.2f} ml/s",
            "Acceleration": f"{acceleration:.2f} ml/s²"
        }
        
        # Validate all values are JSON serializable
        for key, value in parameters.items():
            if not isinstance(value, (str, int, float)):
                parameters[key] = str(value)
                
        return parameters
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parameter calculation failed: {str(e)}")

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Handle audio file upload and generate predictions"""
    try:
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Read file content
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process audio and get flow rate
        flow_rate, time = process_audio(contents)
        
        # Ensure arrays are finite and JSON serializable
        flow_rate = np.nan_to_num(flow_rate, nan=0.0, posinf=50.0, neginf=0.0)
        time = np.nan_to_num(time, nan=0.0)
        
        if len(flow_rate) == 0 or len(time) == 0:
            raise HTTPException(status_code=500, detail="No data generated from audio processing")
        
        # Calculate parameters
        parameters = calculate_parameters(time, flow_rate)
        
        # Convert numpy arrays to lists and ensure all values are finite
        return PredictionResponse(
            flow_rate=flow_rate.tolist(),
            parameters=parameters,
            time=time.tolist()
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(np.datetime64('now'))
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "version": "1.0",
        "supported_formats": ["wav", "mp3"],
        "max_duration": "60 seconds",
        "flow_rate_range": "0-50 ml/s",
        "processing_parameters": {
            "frame_length": "100ms",
            "bandpass_filter": {
                "lowcut": "20 Hz",
                "highcut": "2000 Hz"
            }
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)