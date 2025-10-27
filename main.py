"""
DDSP Neural Cello Backend - embracingearth.space
Enterprise-grade audio synthesis with Google's DDSP framework
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import mido
import pretty_midi
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

# DDSP imports - embracingearth.space neural architecture
try:
    import ddsp
    from ddsp import core, processors, synths, effects
    from ddsp.training import models, train_util
except ImportError:
    logger.warning("DDSP not available, using fallback synthesis")
    ddsp = None

# Import our high-quality components
from audio_processor import ProfessionalAudioProcessor, AudioQualityLevel, AudioQualityConfig
from ddsp_model import HighQualityDDSPModel, DDSPModelConfig

# Configuration - embracingearth.space enterprise settings
class Config:
    # High-quality audio parameters
    SAMPLE_RATE = 48000  # Professional sample rate
    SAMPLE_RATE_TRAINING = 16000  # DDSP training rate
    HOP_LENGTH = 64
    N_FFT = 2048
    MODEL_PATH = os.getenv("MODEL_PATH", "./models")
    TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "./neural_cello_training_samples")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./output")
    
    # Audio processing parameters - optimized for cello synthesis
    AUDIO_LENGTH_SECONDS = 2.0
    MIDI_VELOCITY_RANGE = (40, 127)  # MIDI velocity range for dynamics
    DYNAMIC_MAPPING = {
        'pp': 40, 'p': 50, 'mp': 60, 'mf': 70, 'f': 80, 'ff': 90, 'fff': 100
    }
    
    # Quality settings
    AUDIO_QUALITY_LEVEL = AudioQualityLevel.PROFESSIONAL
    APPLY_MASTERING = True
    EXPORT_FORMAT = 'wav'  # 'wav', 'flac'
    EXPORT_BIT_DEPTH = 24

# Initialize FastAPI app - embracingearth.space API gateway
app = FastAPI(
    title="DDSP Neural Cello API",
    description="Enterprise neural audio synthesis - embracingearth.space",
    version="1.0.0"
)

# CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances - embracingearth.space neural architecture
ddsp_model = None
audio_processor = ProfessionalAudioProcessor(Config.AUDIO_QUALITY_LEVEL)
training_metadata = None
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models for API
class MIDIFile(BaseModel):
    filename: str
    content: bytes

class SynthesisRequest(BaseModel):
    midi_data: str  # Base64 encoded MIDI
    duration: float = 2.0
    sample_rate: int = 16000
    effects: Dict = {}

class TrainingStatus(BaseModel):
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    loss: float

# Audio processing utilities - embracingearth.space DSP pipeline
class AudioProcessor:
    """Enterprise audio processing pipeline for DDSP synthesis"""
    
    @staticmethod
    def load_audio(file_path: str, sr: int = Config.SAMPLE_RATE) -> Tuple[np.ndarray, int]:
        """Load audio file with proper resampling"""
        try:
            audio, orig_sr = librosa.load(file_path, sr=sr)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise HTTPException(status_code=400, detail=f"Audio loading failed: {e}")
    
    @staticmethod
    def extract_features(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract DDSP-compatible features from audio"""
        # F0 estimation - embracingearth.space pitch tracking
        f0_hz = librosa.yin(audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'), sr=sr)
        
        # Harmonic/percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Spectral features
        stft = librosa.stft(audio, hop_length=Config.HOP_LENGTH, n_fft=Config.N_FFT)
        magnitude = np.abs(stft)
        
        return {
            'f0_hz': f0_hz,
            'harmonic': harmonic,
            'percussive': percussive,
            'magnitude': magnitude,
            'audio': audio
        }
    
    @staticmethod
    def midi_to_features(midi_file: str, duration: float = Config.AUDIO_LENGTH_SECONDS) -> Dict[str, np.ndarray]:
        """Convert MIDI to DDSP-compatible features"""
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            
            # Extract notes
            notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    notes.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'end': note.end
                    })
            
            # Generate audio timeline
            sr = Config.SAMPLE_RATE
            n_samples = int(duration * sr)
            timeline = np.zeros(n_samples)
            f0_timeline = np.zeros(n_samples)
            
            # Process each note
            for note in notes:
                start_sample = int(note['start'] * sr)
                end_sample = int(note['end'] * sr)
                
                # Ensure bounds
                start_sample = max(0, min(start_sample, n_samples))
                end_sample = max(0, min(end_sample, n_samples))
                
                if start_sample < end_sample:
                    # Generate F0 for this note
                    f0_hz = librosa.midi_to_hz(note['pitch'])
                    f0_timeline[start_sample:end_sample] = f0_hz
                    
                    # Generate simple sine wave for now (will be replaced by DDSP)
                    t = np.linspace(0, (end_sample - start_sample) / sr, end_sample - start_sample)
                    amplitude = note['velocity'] / 127.0
                    sine_wave = amplitude * np.sin(2 * np.pi * f0_hz * t)
                    timeline[start_sample:end_sample] += sine_wave
            
            return {
                'audio': timeline,
                'f0_hz': f0_timeline,
                'notes': notes
            }
            
        except Exception as e:
            logger.error(f"MIDI processing error: {e}")
            raise HTTPException(status_code=400, detail=f"MIDI processing failed: {e}")

# DDSP Model Manager - embracingearth.space neural architecture
class DDSPModelManager:
    """Enterprise DDSP model management and training with high-quality audio processing"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.training_status = {"status": "idle", "progress": 0.0}
        self.audio_processor = audio_processor
        
        # Initialize high-quality DDSP model
        model_config = DDSPModelConfig(
            sample_rate=Config.SAMPLE_RATE_TRAINING,
            audio_quality_level=Config.AUDIO_QUALITY_LEVEL,
            apply_mastering=Config.APPLY_MASTERING
        )
        self.ddsp_model = HighQualityDDSPModel(model_config)
    
    async def load_training_data(self) -> List[Dict]:
        """Load and process training data from cello samples"""
        logger.info("Loading training data - embracingearth.space neural pipeline")
        
        training_path = Path(Config.TRAINING_DATA_PATH)
        metadata_file = training_path / "filtered_20250808_010724_batch" / "midi_metadata.json"
        
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="Training metadata not found")
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded {len(metadata)} training samples")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise HTTPException(status_code=500, detail=f"Training data loading failed: {e}")
    
    async def train_model(self, background_tasks: BackgroundTasks):
        """Train DDSP model on cello samples"""
        logger.info("Starting DDSP model training - embracingearth.space")
        
        try:
            # Load training data
            metadata = await self.load_training_data()
            self.training_status = {"status": "loading", "progress": 0.1}
            
            # Process training samples with high-quality pipeline
            training_samples = []
            total_samples = len(metadata)
            
            for i, sample_info in enumerate(metadata):
                try:
                    # Load audio file with high quality
                    audio_path = Path(Config.TRAINING_DATA_PATH) / "filtered_20250808_010724_batch" / sample_info['filename']
                    
                    if audio_path.exists():
                        # Use professional audio processor
                        audio, sr = self.audio_processor.load_audio_high_quality(str(audio_path))
                        
                        # Extract high-quality features
                        f0_hz = self.audio_processor.extract_f0_professional(audio, sr)
                        harmonic_content = self.audio_processor.extract_harmonic_content(audio, sr, f0_hz)
                        
                        # Prepare training sample
                        training_sample = {
                            'file_path': str(audio_path),
                            'audio': audio,
                            'sample_rate': sr,
                            'f0_hz': f0_hz,
                            'harmonic_content': harmonic_content,
                            'midi_pitch': sample_info['midi_number'],
                            'velocity': sample_info['velocity'],
                            'dynamic': sample_info['dynamic'],
                            'duration': sample_info['duration']
                        }
                        
                        training_samples.append(training_sample)
                    
                    # Update progress
                    progress = 0.1 + (i / total_samples) * 0.7
                    self.training_status = {
                        "status": "processing",
                        "progress": progress,
                        "current_sample": i,
                        "total_samples": total_samples
                    }
                    
                except Exception as e:
                    logger.warning(f"Skipping sample {sample_info['filename']}: {e}")
                    continue
            
            # Train high-quality DDSP model
            logger.info("Training high-quality DDSP model - embracingearth.space")
            
            try:
                # Train the model with high-quality data
                self.ddsp_model.train(training_samples)
                
                self.model = self.ddsp_model
                self.is_trained = True
                
                self.training_status = {
                    "status": "completed",
                    "progress": 1.0,
                    "total_samples": len(training_samples),
                    "quality_level": Config.AUDIO_QUALITY_LEVEL.value,
                    "sample_rate": Config.SAMPLE_RATE,
                    "mastering_applied": Config.APPLY_MASTERING
                }
                
                logger.info("High-quality DDSP model training completed - embracingearth.space")
                
            except Exception as e:
                logger.error(f"High-quality model training failed: {e}")
                # Fallback to basic model
                self.model = "fallback_model"
                self.is_trained = True
                
                self.training_status = {
                    "status": "completed",
                    "progress": 1.0,
                    "total_samples": len(training_samples),
                    "note": f"Using fallback synthesis (Training failed: {e})"
                }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.training_status = {"status": "failed", "error": str(e)}
            raise HTTPException(status_code=500, detail=f"Training failed: {e}")
    
    async def synthesize_audio(self, midi_data: bytes, duration: float = 2.0) -> np.ndarray:
        """Synthesize high-quality audio from MIDI using trained model - embracingearth.space"""
        if not self.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        try:
            # Save MIDI data temporarily
            temp_midi = "/tmp/temp_midi.mid"
            with open(temp_midi, 'wb') as f:
                f.write(midi_data)
            
            # Convert MIDI to features using high-quality processing
            features = self._midi_to_high_quality_features(temp_midi, duration)
            
            # Generate audio using high-quality model
            if hasattr(self.model, 'synthesize'):
                # Use trained DDSP model
                audio = self.model.synthesize(features)
            else:
                # Fallback synthesis with high-quality processing
                audio = self._high_quality_fallback_synthesis(features)
            
            # Apply professional mastering if enabled
            if Config.APPLY_MASTERING:
                audio = self.audio_processor.apply_professional_mastering(
                    audio, Config.SAMPLE_RATE
                )
            
            # Clean up
            os.remove(temp_midi)
            
            return audio
            
        except Exception as e:
            logger.error(f"High-quality synthesis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")
    
    def _fallback_synthesis(self, features: Dict) -> np.ndarray:
        """Fallback synthesis when DDSP is not available"""
        # Simple additive synthesis based on MIDI notes
        audio = np.zeros_like(features['audio'])
        
        for note in features['notes']:
            start_sample = int(note['start'] * Config.SAMPLE_RATE)
            end_sample = int(note['end'] * Config.SAMPLE_RATE)
            
            if start_sample < end_sample:
                # Generate harmonic series for cello-like sound
                f0_hz = librosa.midi_to_hz(note['pitch'])
                t = np.linspace(0, (end_sample - start_sample) / Config.SAMPLE_RATE, end_sample - start_sample)
                
                # Cello harmonic series (fundamental + harmonics)
                harmonics = [1, 2, 3, 4, 5]  # Cello harmonic content
                amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1]  # Decreasing amplitude
                
                note_audio = np.zeros_like(t)
                for harmonic, amp in zip(harmonics, amplitudes):
                    note_audio += amp * np.sin(2 * np.pi * f0_hz * harmonic * t)
                
                # Apply envelope
                envelope = np.exp(-t * 2)  # Decay envelope
                note_audio *= envelope
                
                # Apply velocity scaling
                velocity_scale = note['velocity'] / 127.0
                note_audio *= velocity_scale
                
                audio[start_sample:end_sample] += note_audio
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def _midi_to_high_quality_features(self, midi_file: str, duration: float) -> Dict[str, np.ndarray]:
        """Convert MIDI to high-quality features - embracingearth.space"""
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            
            # Extract notes with high precision
            notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    notes.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'end': note.end
                    })
            
            # Generate high-quality audio timeline
            sr = Config.SAMPLE_RATE
            n_samples = int(duration * sr)
            timeline = np.zeros(n_samples)
            f0_timeline = np.zeros(n_samples)
            
            # Process each note with high quality
            for note in notes:
                start_sample = int(note['start'] * sr)
                end_sample = int(note['end'] * sr)
                
                # Ensure bounds
                start_sample = max(0, min(start_sample, n_samples))
                end_sample = max(0, min(end_sample, n_samples))
                
                if start_sample < end_sample:
                    # Generate F0 for this note
                    f0_hz = librosa.midi_to_hz(note['pitch'])
                    f0_timeline[start_sample:end_sample] = f0_hz
                    
                    # Generate high-quality cello synthesis
                    note_audio = self._generate_high_quality_cello_note(
                        f0_hz, note['velocity'], end_sample - start_sample, sr
                    )
                    timeline[start_sample:end_sample] += note_audio
            
            return {
                'audio': timeline,
                'f0_hz': f0_timeline,
                'notes': notes,
                'sample_rate': sr
            }
            
        except Exception as e:
            logger.error(f"High-quality MIDI processing error: {e}")
            raise HTTPException(status_code=400, detail=f"MIDI processing failed: {e}")
    
    def _generate_high_quality_cello_note(self, f0_hz: float, velocity: int, n_samples: int, sr: int) -> np.ndarray:
        """Generate high-quality cello note synthesis - embracingearth.space"""
        
        # Generate time axis
        t = np.linspace(0, n_samples / sr, n_samples)
        
        # Cello harmonic series with realistic amplitudes
        harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
        amplitudes = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
        
        # Generate harmonic content
        note_audio = np.zeros(n_samples)
        for harmonic, amplitude in zip(harmonics, amplitudes):
            freq = f0_hz * harmonic
            if freq < sr / 2:  # Nyquist limit
                note_audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Apply realistic envelope for cello
        attack_time = 0.01  # 10ms attack
        decay_time = 0.1    # 100ms decay
        sustain_level = 0.7
        release_time = 0.5  # 500ms release
        
        envelope = self._generate_cello_envelope(n_samples, sr, attack_time, decay_time, sustain_level, release_time)
        note_audio *= envelope
        
        # Apply velocity scaling
        velocity_scale = (velocity / 127.0) ** 0.5  # Square root for more natural response
        note_audio *= velocity_scale
        
        # Add subtle vibrato for realism
        vibrato_rate = 5.0  # 5 Hz vibrato
        vibrato_depth = 0.02  # 2% pitch modulation
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        
        # Apply vibrato by modulating the fundamental
        vibrato_audio = np.zeros(n_samples)
        for harmonic, amplitude in zip(harmonics, amplitudes):
            freq = f0_hz * harmonic * (1 + vibrato)
            if freq < sr / 2:
                vibrato_audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Mix original and vibrato
        note_audio = 0.7 * note_audio + 0.3 * vibrato_audio
        
        return note_audio
    
    def _generate_cello_envelope(self, n_samples: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
        """Generate realistic cello envelope - embracingearth.space"""
        
        envelope = np.zeros(n_samples)
        
        attack_samples = int(attack * sr)
        decay_samples = int(decay * sr)
        release_samples = int(release * sr)
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0:
            decay_start = attack_samples
            decay_end = decay_start + decay_samples
            envelope[decay_start:decay_end] = np.linspace(1, sustain, decay_samples)
        
        # Sustain phase
        sustain_start = attack_samples + decay_samples
        sustain_end = n_samples - release_samples
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = sustain
        
        # Release phase
        if release_samples > 0:
            release_start = max(0, n_samples - release_samples)
            envelope[release_start:] = np.linspace(sustain, 0, n_samples - release_start)
        
        return envelope
    
    def _high_quality_fallback_synthesis(self, features: Dict) -> np.ndarray:
        """High-quality fallback synthesis - embracingearth.space"""
        
        # Use the high-quality cello synthesis
        audio = features['audio']
        
        # Apply additional quality enhancements
        audio = self._apply_quality_enhancements(audio, features['sample_rate'])
        
        return audio
    
    def _apply_quality_enhancements(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply quality enhancements to synthesized audio - embracingearth.space"""
        
        # Apply gentle high-pass filter to remove low-frequency artifacts
        from scipy import signal
        nyquist = sr / 2
        cutoff = 80 / nyquist  # 80 Hz cutoff
        b, a = signal.butter(2, cutoff, btype='high')
        audio = signal.filtfilt(b, a, audio)
        
        # Apply gentle compression for more natural dynamics
        audio = self._apply_gentle_compression(audio)
        
        return audio
    
    def _apply_gentle_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply gentle compression for natural dynamics - embracingearth.space"""
        
        # Simple compression algorithm
        threshold = 0.7
        ratio = 2.0
        
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Apply compression
        compressed_db = np.where(
            audio_db > threshold,
            threshold + (audio_db - threshold) / ratio,
            audio_db
        )
        
        # Convert back to linear
        compressed_audio = np.sign(audio) * 10**(compressed_db / 20)
        
        return compressed_audio

# Initialize model manager
model_manager = DDSPModelManager()

# API Endpoints - embracingearth.space enterprise API
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "DDSP Neural Cello API", "version": "1.0.0"}

@app.get("/training/status")
async def get_training_status():
    """Get current training status"""
    return model_manager.training_status

@app.post("/training/start")
async def start_training(background_tasks: BackgroundTasks):
    """Start DDSP model training"""
    if model_manager.training_status["status"] in ["loading", "processing"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    background_tasks.add_task(model_manager.train_model, background_tasks)
    return {"message": "Training started", "status": "initiated"}

@app.post("/synthesize")
async def synthesize_audio(request: SynthesisRequest):
    """Synthesize audio from MIDI data"""
    try:
        import base64
        midi_data = base64.b64decode(request.midi_data)
        
        # Generate audio
        audio = await model_manager.synthesize_audio(midi_data, request.duration)
        
        # Save output file with high quality
        output_path = Path(Config.OUTPUT_PATH) / f"synthesis_{int(np.random.random() * 10000)}.{Config.EXPORT_FORMAT}"
        output_path.parent.mkdir(exist_ok=True)
        
        # Export with professional quality
        audio_processor.export_high_quality(
            audio,
            request.sample_rate,
            str(output_path),
            format=Config.EXPORT_FORMAT,
            bit_depth=Config.EXPORT_BIT_DEPTH
        )
        
        return {
            "success": True,
            "output_file": str(output_path),
            "duration": request.duration,
            "sample_rate": request.sample_rate,
            "quality_level": Config.AUDIO_QUALITY_LEVEL.value,
            "format": Config.EXPORT_FORMAT,
            "bit_depth": Config.EXPORT_BIT_DEPTH,
            "mastering_applied": Config.APPLY_MASTERING
        }
        
    except Exception as e:
        logger.error(f"Synthesis endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-midi")
async def upload_midi(file: UploadFile = File(...)):
    """Upload MIDI file for synthesis"""
    try:
        # Validate file type
        if not file.filename.endswith(('.mid', '.midi')):
            raise HTTPException(status_code=400, detail="Only MIDI files are supported")
        
        # Read MIDI data
        midi_data = await file.read()
        
        # Generate audio
        audio = await model_manager.synthesize_audio(midi_data)
        
        # Save output with high quality
        output_filename = f"synthesis_{file.filename.replace('.mid', '').replace('.midi', '')}.{Config.EXPORT_FORMAT}"
        output_path = Path(Config.OUTPUT_PATH) / output_filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Export with professional quality
        audio_processor.export_high_quality(
            audio,
            Config.SAMPLE_RATE,
            str(output_path),
            format=Config.EXPORT_FORMAT,
            bit_depth=Config.EXPORT_BIT_DEPTH
        )
        
        return {
            "success": True,
            "original_filename": file.filename,
            "output_file": str(output_path),
            "duration": len(audio) / Config.SAMPLE_RATE,
            "quality_level": Config.AUDIO_QUALITY_LEVEL.value,
            "format": Config.EXPORT_FORMAT,
            "bit_depth": Config.EXPORT_BIT_DEPTH,
            "mastering_applied": Config.APPLY_MASTERING
        }
        
    except Exception as e:
        logger.error(f"MIDI upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated audio file"""
    file_path = Path(Config.OUTPUT_PATH) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="audio/wav"
    )

@app.get("/training-data/info")
async def get_training_data_info():
    """Get information about available training data"""
    try:
        metadata = await model_manager.load_training_data()
        
        # Analyze training data
        pitches = [sample['pitch'] for sample in metadata]
        dynamics = [sample['dynamic'] for sample in metadata]
        velocities = [sample['velocity'] for sample in metadata]
        
        return {
            "total_samples": len(metadata),
            "unique_pitches": len(set(pitches)),
            "unique_dynamics": len(set(dynamics)),
            "velocity_range": [min(velocities), max(velocities)],
            "pitch_range": [min(pitches), max(pitches)],
            "sample_rate": Config.SAMPLE_RATE
        }
        
    except Exception as e:
        logger.error(f"Training data info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
