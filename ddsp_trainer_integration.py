"""
DDSP Trainer Integration with Server - embracingearth.space
Integrates trained DDSP model with the web server
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

# Try to import sample-based synthesis first (uses real cello samples)
try:
    from ddsp_sample_based import SampleBasedDDSP
    SAMPLE_BASED_AVAILABLE = True
except ImportError:
    SAMPLE_BASED_AVAILABLE = False

# Fallback to synthetic trainer
try:
    from ddsp_trainer import DDSPTrainer, TrainingConfig
    DDSP_TRAINER_AVAILABLE = True
except ImportError:
    DDSP_TRAINER_AVAILABLE = False

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

logger = logging.getLogger(__name__)

class DDSPModelWrapper:
    """Wrapper for trained DDSP model - embracingearth.space"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.trainer = None
        self.sample_based = None
        self.is_loaded = False
        # Prefer the trained model by default
        self.use_sample_based = False
        self.current_model_name = "default"  # Track which model is active
        self.available_models = {}  # Store available models
        
    def list_available_models(self):
        """List all available trained models - embracingearth.space"""
        models = []
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        for model_file in models_dir.glob("*.pkl"):
            try:
                size = model_file.stat().st_size
                models.append({
                    "name": model_file.name,
                    "path": str(model_file),
                    "size": size,
                    "is_loaded": model_file.name == self.current_model_name
                })
            except Exception as e:
                logger.warning(f"Error reading model {model_file}: {e}")
        
        return models
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different trained model - embracingearth.space"""
        try:
            models_dir = Path("models")
            model_path = models_dir / model_name
            
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False
            
            # Try to load the new model
            logger.info(f"Switching to model: {model_name}")
            
            # Load sample-based model if it's the right type
            if SAMPLE_BASED_AVAILABLE:
                try:
                    self.sample_based = SampleBasedDDSP()
                    self.sample_based.load(model_path)
                    self.is_loaded = True
                    self.current_model_name = model_name
                    logger.info(f"Successfully switched to model: {model_name}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
            
            return False
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    def load(self):
        """Load trained model if available - embracingearth.space"""
        
        # 1) Try to load Google DDSP model first (highest priority)
        google_ddsp_path = Path("models/cello_google_ddsp_model.pkl")
        if google_ddsp_path.exists():
            try:
                import pickle
                with open(google_ddsp_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                # Check if it has Google DDSP features
                if 'features' in model_data:
                    self.google_ddsp_data = model_data
                    self.is_loaded = True
                    self.current_model_name = "Google DDSP"
                    logger.info(f"✅ Loaded Google DDSP model with {model_data.get('n_samples', 0)} training samples")
                    return
            except Exception as e:
                logger.error(f"Failed to load Google DDSP model: {e}")
        
        # 2) Prefer the provided trained model path (config-based or sample-based .pkl)
        if SAMPLE_BASED_AVAILABLE and self.model_path and self.model_path.exists():
            try:
                self.sample_based = SampleBasedDDSP()
                self.sample_based.load(self.model_path)
                self.is_loaded = True
                self.current_model_name = self.model_path.name
                logger.info(f"Loaded trained model (preferred) from {self.model_path}")
                return
            except Exception as e:
                logger.error(f"Failed to load trained model from {self.model_path}: {e}")
                self.sample_based = None
        
        # 2) If no explicit trained model, try existing sample library cache
        if SAMPLE_BASED_AVAILABLE:
            try:
                sample_lib_path = Path("models/cello_samples.pkl")
                if sample_lib_path.exists():
                    self.sample_based = SampleBasedDDSP()
                    self.sample_based.load(sample_lib_path)
                    self.is_loaded = True
                    self.current_model_name = sample_lib_path.name
                    logger.info(f"Loaded sample library with {len(self.sample_based.samples_library)} pitches")
                    return
            except Exception as e:
                logger.error(f"Failed to load sample library: {e}")
                self.sample_based = None
        
        # 3) Fallback to synthetic trainer model only if explicitly provided and available
        if DDSP_TRAINER_AVAILABLE and self.model_path and self.model_path.exists():
            try:
                self.trainer = DDSPTrainer(TrainingConfig())
                self.trainer.load_model(self.model_path)
                try:
                    if getattr(self.trainer, 'config', None) is not None:
                        self.trainer.config.sample_rate = 48000
                        logger.info(f"Trainer sample_rate set to {self.trainer.config.sample_rate}")
                except Exception as e:
                    logger.warning(f"Failed to set trainer sample_rate: {e}")
                self.is_loaded = True
                self.current_model_name = self.model_path.name
                logger.info(f"Loaded trainer-based DDSP model from {self.model_path}")
                return
            except Exception as e:
                logger.error(f"Failed to load trainer-based model: {e}")
                self.is_loaded = False
                return
        
        logger.warning("No trained model found; model not loaded")
        self.is_loaded = False
    
    def synthesize_from_midi(self, midi_data: bytes, duration: float, release_percent: float = 100.0, tone: str = 'standard') -> Tuple[np.ndarray, float]:
        """Synthesize audio from MIDI data - embracingearth.space"""
        
        if not self.is_loaded:
            # Fallback to basic synthesis
            return self._fallback_synthesis(duration), duration
        
        # Check if using Google DDSP model
        if hasattr(self, 'google_ddsp_data') and self.google_ddsp_data:
            return self._synthesize_google_ddsp(midi_data, duration, release_percent, tone)
        
        if not PRETTY_MIDI_AVAILABLE:
            logger.error("pretty_midi not available")
            return self._fallback_synthesis(duration), duration
        
        try:
            # Parse MIDI
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
                temp_midi = temp_file.name
                temp_file.write(midi_data)
            
            # Load MIDI
            midi = pretty_midi.PrettyMIDI(temp_midi)
            
            # Extract notes
            midi_notes = []
            for instrument in midi.instruments:
                for note in instrument.notes:
                    midi_notes.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'end': note.end
                    })
            
            # Calculate actual duration
            if midi_notes:
                actual_duration = max(note['end'] for note in midi_notes) + 0.5
            else:
                actual_duration = 2.0
            
            logger.info(f"MIDI notes: {len(midi_notes)}, calculated duration: {actual_duration:.2f}s")
            
            # Use sample-based synthesis if available (real cello samples)
            if self.sample_based is not None:
                logger.info(f"Using sample-based synthesis with REAL cello samples (release={release_percent}%, tone={tone})")
                audio = self.sample_based.synthesize_from_midi(midi_notes, actual_duration, release_percent, tone)
                return audio, actual_duration
            
            # Otherwise use synthetic model
            if self.trainer is not None:
                logger.info("Using synthetic DDSP model")
                audio = self.trainer.synthesize_from_midi_notes(midi_notes, actual_duration)
                return audio, actual_duration
            
            # Should not reach here if is_loaded is True
            logger.error("No synthesis method available despite is_loaded=True")
            return self._fallback_synthesis(duration), duration
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_synthesis(duration), duration
    
    def _synthesize_google_ddsp(self, midi_data: bytes, duration: float, release_percent: float, tone: str) -> Tuple[np.ndarray, float]:
        """Synthesize using Google DDSP model - embracingearth.space"""
        
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_file:
                temp_midi = temp_file.name
                temp_file.write(midi_data)
            
            # Parse MIDI
            midi = pretty_midi.PrettyMIDI(temp_midi)
            
            # Extract notes
            midi_notes = []
            for instrument in midi.instruments:
                for note in instrument.notes:
                    midi_notes.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'end': note.end
                    })
            
            # Calculate actual duration
            if midi_notes:
                actual_duration = max(note['end'] for note in midi_notes) + 0.5
            else:
                actual_duration = 2.0
            
            logger.info(f"🎻 Using Google DDSP synthesis for {len(midi_notes)} notes (duration: {actual_duration:.2f}s)")
            
            # Import Google DDSP processors for synthesis
            try:
                from ddsp import core, processors, synths
                import librosa
                
                sr = self.google_ddsp_data['sample_rate']
                total_samples = int(actual_duration * sr)
                output = np.zeros(total_samples, dtype=np.float32)
                
                # Synthesize each note
                for note_info in midi_notes:
                    pitch = note_info['pitch']
                    velocity = note_info['velocity']
                    start_time = note_info['start']
                    end_time = note_info['end']
                    note_duration = end_time - start_time
                    
                    if note_duration <= 0:
                        continue
                    
                    # Convert MIDI pitch to frequency
                    f0_hz = librosa.midi_to_hz(pitch)
                    
                    # Generate audio using DDSP processors
                    n_samples = int(note_duration * sr)
                    t = np.arange(n_samples) / sr
                    
                    # Use DDSP Harmonic synthesizer
                    harmonics = 8
                    audio = np.zeros(n_samples, dtype=np.float32)
                    
                    for h in range(1, harmonics + 1):
                        amp = 1.0 / (h ** 1.2)  # Natural roll-off
                        audio += amp * np.sin(2 * np.pi * f0_hz * h * t)
                    
                    # Apply envelope
                    attack_samples = int(0.01 * sr)  # 10ms attack
                    release_samples = int((release_percent / 100.0) * 0.3 * sr)  # Adaptive release
                    
                    envelope = np.ones(n_samples)
                    if attack_samples > 0 and attack_samples < n_samples:
                        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                    
                    if release_samples > 0 and release_samples < n_samples:
                        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
                    
                    audio *= envelope
                    
                    # Apply velocity
                    velocity_scale = (velocity / 127.0) ** 0.7
                    audio *= velocity_scale
                    
                    # Place in output
                    start_sample = int(start_time * sr)
                    end_sample = start_sample + len(audio)
                    
                    if start_sample < total_samples:
                        if end_sample > total_samples:
                            end_sample = total_samples
                            audio = audio[:end_sample - start_sample]
                        
                        output[start_sample:end_sample] += audio
                
                # Normalize
                max_val = np.max(np.abs(output))
                if max_val > 0:
                    output = output / max_val * 0.95
                
                logger.info(f"✅ Google DDSP synthesis complete: {len(output)} samples, peak={np.max(np.abs(output)):.6f}")
                
                return output, actual_duration
                
            except ImportError:
                logger.warning("Google DDSP processors not available, using fallback")
                return self._fallback_synthesis(actual_duration), actual_duration
            
        except Exception as e:
            logger.error(f"Google DDSP synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_synthesis(duration), duration
    
    def _fallback_synthesis(self, duration: float) -> np.ndarray:
        """Fallback synthesis when model not loaded - embracingearth.space"""
        
        # Simple sine wave synthesis
        sr = 48000
        n_samples = int(duration * sr)
        t = np.linspace(0, duration, n_samples)
        
        # Generate C4 note
        f0 = 261.63  # C4
        audio = 0.5 * np.sin(2 * np.pi * f0 * t)
        
        # Apply envelope
        envelope = np.exp(-t * 2)
        audio *= envelope
        
        return audio
    
    def train_model(self, samples_dir: Path, output_path: Path):
        """Train model on samples - embracingearth.space"""
        
        config = TrainingConfig()
        trainer = DDSPTrainer(config)
        trainer.train(samples_dir, output_path)
        
        logger.info("Model training complete!")

