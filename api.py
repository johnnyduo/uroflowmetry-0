# api.py

import os
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter, butter, filtfilt
import io
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="Uroflowmetry API")

# CORS middleware
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

def process_audio(audio_bytes):
    """Process audio file and extract features"""
    try:
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # Ensure we have audio data
        if len(y) == 0:
            raise ValueError("No audio data found")
        
        # Calculate RMS energy
        frame_length = int(sr * 0.1)  # 100ms frames
        hop_length = int(frame_length / 2)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Ensure we have RMS values
        if len(rms) == 0:
            raise ValueError("No RMS values calculated")
        
        # Smooth the RMS curve
        if len(rms) > 3:  # Ensure enough points for smoothing
            rms_smoothed = savgol_filter(rms, min(15, len(rms)-2 if len(rms) % 2 == 0 else len(rms)-1), 3)
        else:
            rms_smoothed = rms
            
        # Scale to realistic flow rate values (0-50 ml/s)
        scaler = MinMaxScaler(feature_range=(0, 50))
        flow_rate = scaler.fit_transform(rms_smoothed.reshape(-1, 1)).flatten()
        
        # Create time array
        duration = len(y) / sr
        time = np.linspace(0, duration, len(flow_rate))
        
        return flow_rate, time
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

def calculate_parameters(time, flow_rate):
    """Calculate uroflowmetry parameters"""
    try:
        max_flow = float(np.max(flow_rate))
        avg_flow = float(np.mean(flow_rate))
        voiding_duration = float(time[-1])
        voided_volume = float(np.trapz(flow_rate, time))
        time_to_max = float(time[np.argmax(flow_rate)])
        
        # Calculate flow at 2 seconds
        idx_2s = np.where(time >= 2.0)[0]
        flow_at_2s = float(flow_rate[idx_2s[0]]) if len(idx_2s) > 0 else 0.0
        acceleration = flow_at_2s / 2.0 if flow_at_2s > 0 else 0.0
        
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
        raise HTTPException(status_code=500, detail=f"Parameter calculation failed: {str(e)}")

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Read file content
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process audio and get flow rate
        flow_rate, time = process_audio(contents)
        
        if len(flow_rate) == 0 or len(time) == 0:
            raise HTTPException(status_code=500, detail="No data generated from audio processing")
        
        # Calculate parameters
        parameters = calculate_parameters(time, flow_rate)
        
        # Convert numpy arrays to lists and ensure all values are Python native types
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
    return {"status": "healthy"}

@app.get("/model-info")
async def model_info():
    return {
        "version": "1.0",
        "supported_formats": ["wav", "mp3"],
        "max_duration": "60 seconds",
        "flow_rate_range": "0-50 ml/s"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)