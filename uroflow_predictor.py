# uroflow_predictor.py

import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

class UroflowPredictor:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(None, 1)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

    def extract_features(self, audio_signal, sr):
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_signal, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sr)[0]
        rms_energy = librosa.feature.rms(y=audio_signal)[0]
        zero_crossings = librosa.feature.zero_crossing_rate(audio_signal)[0]
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
        
        features = np.vstack([
            spectral_centroids,
            spectral_rolloff,
            rms_energy,
            zero_crossings,
            *mfccs
        ])
        
        return features.T

    def preprocess_audio(self, audio_file):
        audio_signal, sr = librosa.load(audio_file, sr=None)
        
        nyquist = sr / 2
        low_cut = 20 / nyquist
        high_cut = 2000 / nyquist
        b, a = butter(4, [low_cut, high_cut], btype='band')
        filtered_signal = filtfilt(b, a, audio_signal)
        
        features = self.extract_features(filtered_signal, sr)
        normalized_features = self.scaler.fit_transform(features)
        
        return normalized_features

    def predict_flow_rate(self, audio_file):
        features = self.preprocess_audio(audio_file)
        X = features.reshape(1, *features.shape, 1)
        flow_rate = self.model.predict(X)
        flow_rate = self._post_process_predictions(flow_rate[0])
        return flow_rate

    def _post_process_predictions(self, predictions):
        window_size = 5
        smoothed = np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')
        smoothed = np.maximum(smoothed, 0)
        max_flow_rate = 50
        smoothed = np.minimum(smoothed, max_flow_rate)
        return smoothed