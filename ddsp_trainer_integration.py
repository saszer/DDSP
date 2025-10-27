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
        self.use_sample_based = True  # Prefer real samples
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
        
        # Try to load sample-based model first (uses real cello samples)
        if SAMPLE_BASED_AVAILABLE and self.use_sample_based:
            try:
                # Check for sample library
                sample_lib_path = Path("models/cello_samples.pkl")
                if sample_lib_path.exists():
                    self.sample_based = SampleBasedDDSP()
                    self.sample_based.load(sample_lib_path)
                    self.is_loaded = True
                    logger.info(f"Loaded sample-based model with {len(self.sample_based.samples_library)} pitches")
                    return
                else:
                    logger.info("Sample library not found, creating from scratch...")
                    self.sample_based = SampleBasedDDSP()
                    samples_dir = Path("neural_cello_training_samples/filtered_20250808_010724_batch")
                    if samples_dir.exists():
                        self.sample_based.load_samples(samples_dir)
                        self.sample_based.save(sample_lib_path)
                        self.is_loaded = True
                        logger.info(f"Created sample-based model with {len(self.sample_based.samples_library)} pitches")
                        return
            except Exception as e:
                logger.error(f"Failed to load sample-based model: {e}")
                self.sample_based = None
        
        # Fallback to synthetic model
        if not DDSP_TRAINER_AVAILABLE:
            logger.warning("DDSP trainer not available")
            self.is_loaded = False
            return
        
        if self.model_path and self.model_path.exists():
            try:
                self.trainer = DDSPTrainer(TrainingConfig())
                self.trainer.load_model(self.model_path)
                # Force consistent sample rate with server (48kHz) to avoid array length mismatches
                try:
                    if getattr(self.trainer, 'config', None) is not None:
                        self.trainer.config.sample_rate = 48000
                        logger.info(f"Trainer sample_rate set to {self.trainer.config.sample_rate}")
                except Exception as e:
                    logger.warning(f"Failed to set trainer sample_rate: {e}")
                self.is_loaded = True
                logger.info(f"Loaded synthetic DDSP model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.is_loaded = False
        else:
            logger.warning("No trained model found, using fallback")
            self.is_loaded = False
    
    def synthesize_from_midi(self, midi_data: bytes, duration: float, release_percent: float = 100.0, tone: str = 'standard') -> Tuple[np.ndarray, float]:
        """Synthesize audio from MIDI data - embracingearth.space"""
        
        if not self.is_loaded:
            # Fallback to basic synthesis
            return self._fallback_synthesis(duration), duration
        
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

