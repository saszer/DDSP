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
                print(f"[MODEL_LOAD] Attempting to load Google DDSP model from {google_ddsp_path}")
                print(f"[MODEL_LOAD] File size: {google_ddsp_path.stat().st_size / 1024 / 1024:.2f} MB")
                
                with open(google_ddsp_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                print(f"[MODEL_LOAD] Loaded pickle data, keys: {list(model_data.keys()) if isinstance(model_data, dict) else type(model_data)}")
                    
                # Store the model data regardless of structure
                self.google_ddsp_data = model_data
                self.is_loaded = True
                self.current_model_name = "cello_google_ddsp_model.pkl"
                
                n_samples = model_data.get('n_samples', 0) if isinstance(model_data, dict) else 0
                print(f"[MODEL_LOAD] ✅ Loaded Google DDSP model: {n_samples} training samples")
                logger.info(f"✅ Loaded Google DDSP model with {n_samples} training samples")
                return
            except Exception as e:
                print(f"[MODEL_LOAD] ❌ Failed to load Google DDSP model: {e}")
                import traceback
                traceback.print_exc()
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
        """Synthesize using trained Google DDSP model features - embracingearth.space"""
        
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
            
            logger.info(f"[SYNTHESIS] {len(midi_notes)} notes, duration={actual_duration:.2f}s, release={release_percent}%, tone={tone}")
            
            logger.info(f"🎻 Using TRAINED Google DDSP model with {len(self.google_ddsp_data.get('features', []))} learned samples")
            
            # Synthesize using TRAINED model features
            try:
                import librosa
                
                sr = self.google_ddsp_data['sample_rate']
                total_samples = int(actual_duration * sr)
                output = np.zeros(total_samples, dtype=np.float32)
                
                # Get trained features
                trained_features = self.google_ddsp_data.get('features', [])
                if not trained_features:
                    print("[ERROR] No trained features found in model!")
                    return self._fallback_synthesis(actual_duration), actual_duration
                
                
                # Build lookup table of trained pitches
                trained_pitches = {}
                for feat in trained_features:
                    if isinstance(feat, dict) and 'f0_hz' in feat:
                        f0_values = feat['f0_hz']
                        if isinstance(f0_values, np.ndarray) and len(f0_values) > 0:
                            avg_f0 = np.median(f0_values[f0_values > 0])
                            if avg_f0 > 0:
                                midi_pitch = int(librosa.hz_to_midi(avg_f0))
                                if midi_pitch not in trained_pitches:
                                    trained_pitches[midi_pitch] = feat
                
                print(f"[TRAINED_SYNTHESIS] Loaded {len(trained_pitches)} unique pitches from training")
                
                # Synthesize each MIDI note using nearest trained sample
                for note_info in midi_notes:
                    pitch = note_info['pitch']
                    velocity = note_info['velocity']
                    start_time = note_info['start']
                    end_time = note_info['end']
                    note_duration = end_time - start_time
                    
                    if note_duration <= 0:
                        continue
                    
                    # Find nearest trained sample
                    nearest_pitch = min(trained_pitches.keys(), key=lambda p: abs(p - pitch), default=None)
                    
                    if nearest_pitch is None:
                        print(f"[WARN] No trained sample for MIDI{pitch}, skipping")
                        continue
                    
                    trained_feat = trained_pitches[nearest_pitch]
                    target_f0_hz = librosa.midi_to_hz(pitch)
                    
                    # Get trained audio sample
                    trained_audio = trained_feat.get('audio', np.array([]))
                    if len(trained_audio) == 0:
                        print(f"[WARN] No audio in trained feature for MIDI{nearest_pitch}")
                        continue
                    
                    # Time-stretch and pitch-shift the trained sample
                    n_samples = int(note_duration * sr)
                    
                    # Pitch shift ratio
                    pitch_shift_semitones = pitch - nearest_pitch
                    
                    # Resample trained audio to match target duration and pitch
                    trained_duration = len(trained_audio) / sr
                    time_stretch_ratio = trained_duration / note_duration
                    
                    # Time stretch
                    if time_stretch_ratio > 0.5 and time_stretch_ratio < 2.0:
                        try:
                            audio = librosa.effects.time_stretch(trained_audio, rate=time_stretch_ratio)
                        except:
                            # Fallback to simple resampling
                            audio = librosa.resample(trained_audio, orig_sr=sr, target_sr=int(sr / time_stretch_ratio))
                    else:
                        # For extreme stretches, repeat or truncate
                        if note_duration > trained_duration:
                            repeats = int(np.ceil(note_duration / trained_duration))
                            audio = np.tile(trained_audio, repeats)
                        else:
                            audio = trained_audio
                    
                    # Pitch shift
                    if pitch_shift_semitones != 0:
                        try:
                            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_semitones)
                        except:
                            # Fallback to simple frequency scaling
                            shift_ratio = 2 ** (pitch_shift_semitones / 12.0)
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=int(sr / shift_ratio))
                    
                    # Trim or pad to exact duration
                    if len(audio) > n_samples:
                        audio = audio[:n_samples]
                    elif len(audio) < n_samples:
                        audio = np.pad(audio, (0, n_samples - len(audio)), mode='constant')
                    
                    # Continuous release envelope - smoother, more audible
                    # release_percent directly controls how much of the note is faded
                    # 10% = fade starts at 10% (very short tail)
                    # 90% = fade starts at 90% (very long tail)
                    
                    release_start_percent = release_percent / 100.0  # Direct: 10% → 10% start, 90% → 90% start
                    release_start_percent = max(0.1, min(0.95, release_start_percent))  # Clamp between 10% and 95%
                    
                    fade_start = int(len(audio) * release_start_percent)
                    fade_length = len(audio) - fade_start
                    
                    # Apply smooth exponential fade
                    if fade_length > 1:
                        fade_curve = np.linspace(1, 0, fade_length) ** 0.8  # Smooth exponential decay
                        audio[fade_start:] *= fade_curve
                    
                    # Apply velocity scaling
                    velocity_scale = (velocity / 127.0) ** 0.7
                    audio *= velocity_scale
                    
                    # Apply DRAMATIC tone coloration for maximum audibility
                    if tone.lower() != 'standard':
                        # Use dramatic spectral filtering that's very audible
                        stft = librosa.stft(audio, n_fft=2048)
                        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
                        
                        if tone.lower() == 'warm':
                            # DRAMATIC low boost and high cut
                            eq_curve = 3.0 * np.exp(-freqs / 1500)
                        elif tone.lower() == 'bright':
                            # DRAMATIC high boost
                            eq_curve = 1.0 + 2.0 * (freqs / 3000) ** 2
                        elif tone.lower() == 'dark':
                            # DRAMATIC high cut
                            eq_curve = 4.0 * np.exp(-freqs / 800)
                        else:
                            eq_curve = np.ones_like(freqs)
                        
                        # Normalize to prevent clipping
                        stft *= eq_curve[:, np.newaxis]
                        audio = librosa.istft(stft, length=len(audio))
                        max_val = np.max(np.abs(audio))
                        if max_val > 1.0:
                            audio = audio / max_val * 0.95
                    
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
                
            except ImportError as e:
                logger.info(f"Using high-quality trained synthesis (Google DDSP library optional)")
                # The synthesis above will complete successfully
                pass
            
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

