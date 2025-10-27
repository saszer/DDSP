"""
Google DDSP Trainer - Proper Implementation - embracingearth.space
Trains a DDSP model on real cello samples with proper feature extraction
"""

import numpy as np
import librosa
import soundfile as sf
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from scipy import signal
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration - embracingearth.space"""
    sample_rate: int = 48000
    n_fft: int = 2048
    hop_length: int = 256
    n_harmonics: int = 60
    n_noise_frames: int = 65
    
    # Training parameters
    batch_size: int = 32
    n_epochs: int = 50
    learning_rate: float = 1e-4

class DDSPTrainer:
    """Proper DDSP trainer using real cello samples - embracingearth.space"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.training_samples = []
        self.model_params = None
        
        # Learned parameters from training
        self.note_freq_map = {}  # Note name -> frequency
        self.dynamic_profile = {}  # Dynamic level -> spectral profile
        self.spectral_envelope = None  # Learned spectral shape
        
    def load_training_samples(self, samples_dir: Path) -> List[Dict]:
        """Load all cello training samples - embracingearth.space"""
        
        logger.info(f"Loading training samples from {samples_dir}")
        
        samples = []
        wav_files = list(samples_dir.glob("*.wav"))
        
        logger.info(f"Found {len(wav_files)} training samples")
        
        for i, wav_file in enumerate(wav_files):
            try:
                # Parse filename: e.g., "C4_mf_note09.wav"
                name_parts = wav_file.stem.split('_')
                if len(name_parts) >= 3:
                    note = name_parts[0]  # e.g., "C4"
                    dynamic = name_parts[1]  # e.g., "mf"
                    note_variant = name_parts[2]  # e.g., "note09"
                    
                    samples.append({
                        'file_path': str(wav_file),
                        'note': note,
                        'dynamic': dynamic,
                        'note_variant': note_variant
                    })
                    
                if i % 100 == 0 and i > 0:
                    logger.info(f"Loaded {i}/{len(wav_files)} samples...")
                    
            except Exception as e:
                logger.warning(f"Error loading {wav_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(samples)} training samples")
        return samples
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract proper DDSP features - embracingearth.space"""
        
        # Resample to training rate if needed
        if sr != self.config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
            sr = self.config.sample_rate
        
        # Extract F0 using librosa's pyin (robust pitch tracking)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C1'),  # Cello range starts at C1
            fmax=librosa.note_to_hz('C6'),
            frame_length=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        # Fill in unvoiced regions with previous value or NaN
        f0_smooth = np.copy(f0)
        for i in range(1, len(f0_smooth)):
            if np.isnan(f0_smooth[i]):
                f0_smooth[i] = f0_smooth[i-1]
        
        # Extract loudness (RMS energy in dB)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.config.n_fft,
            hop_length=self.config.hop_length,
            center=True
        )[0]
        
        loudness_db = 20 * np.log10(rms + 1e-10)
        
        # Extract harmonic content
        stft = librosa.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft
        )
        
        magnitude = np.abs(stft)
        
        # Estimate harmonic amplitudes
        harmonic_amplitudes = self._extract_harmonic_amplitudes(magnitude, f0_smooth, self.config.n_harmonics)
        
        # Estimate noise content
        noise_envelope = self._extract_noise_envelope(magnitude, f0_smooth)
        
        return {
            'f0_hz': f0_smooth,
            'loudness_db': loudness_db,
            'harmonic_amplitudes': harmonic_amplitudes,
            'noise_envelope': noise_envelope,
            'audio': audio
        }
    
    def _extract_harmonic_amplitudes(self, magnitude: np.ndarray, f0: np.ndarray, n_harmonics: int) -> np.ndarray:
        """Extract harmonic amplitudes from magnitude spectrum - embracingearth.space"""
        
        freqs = librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=self.config.n_fft)
        harmonic_amplitudes = []
        
        for i, f0_val in enumerate(f0):
            if np.isnan(f0_val) or f0_val <= 0:
                harmonic_amplitudes.append(np.zeros(n_harmonics))
                continue
            
            amps = []
            for h in range(1, n_harmonics + 1):
                harmonic_freq = f0_val * h
                
                # Find closest frequency bin
                if harmonic_freq < self.config.sample_rate / 2:
                    freq_idx = np.argmin(np.abs(freqs - harmonic_freq))
                    amplitude = magnitude[freq_idx, i]
                else:
                    amplitude = 0
                
                amps.append(amplitude)
            
            harmonic_amplitudes.append(np.array(amps))
        
        return np.array(harmonic_amplitudes)
    
    def _extract_noise_envelope(self, magnitude: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """Extract noise envelope from spectral content - embracingearth.space"""
        
        # For each frame, compute the noise floor
        noise_envelope = []
        
        for i, f0_val in enumerate(f0):
            if np.isnan(f0_val) or f0_val <= 0:
                noise_envelope.append(0.1)
                continue
            
            # Find fundamental and harmonics
            fundamental_idx = int(f0_val / self.config.sample_rate * self.config.n_fft)
            
            # Get noise floor as average of regions not dominated by harmonics
            magnitude_frame = magnitude[:, i]
            
            # Remove harmonic energy
            harmonic_indices = []
            for h in range(1, 21):  # First 20 harmonics
                harm_freq = f0_val * h
                if harm_freq < self.config.sample_rate / 2:
                    freq_idx = int(harm_freq / self.config.sample_rate * self.config.n_fft)
                    harmonic_indices.append(freq_idx)
            
            # Noise is the average of non-harmonic frequencies
            non_harmonic_mask = np.ones(len(magnitude_frame), dtype=bool)
            for idx in harmonic_indices:
                # Don't count harmonic regions
                if 0 <= idx < len(non_harmonic_mask):
                    non_harmonic_mask[max(0, idx - 2):min(len(non_harmonic_mask), idx + 3)] = False
            
            noise_floor = np.mean(magnitude_frame[non_harmonic_mask])
            noise_envelope.append(noise_floor)
        
        return np.array(noise_envelope)
    
    def train(self, samples_dir: Path, output_model_path: Optional[Path] = None):
        """Train DDSP model on cello samples - embracingearth.space"""
        
        logger.info("Starting DDSP training on cello samples...")
        
        # Load samples
        samples = self.load_training_samples(samples_dir)
        if not samples:
            raise ValueError("No training samples found!")
        
        # Extract features from all samples
        all_features = []
        
        for i, sample in enumerate(samples):
            try:
                # Load audio
                audio, sr = sf.read(sample['file_path'])
                
                # Extract features
                features = self.extract_features(audio, sr)
                features['note'] = sample['note']
                features['dynamic'] = sample['dynamic']
                features['note_variant'] = sample['note_variant']
                
                all_features.append(features)
                
                if i % 100 == 0 and i > 0:
                    logger.info(f"Processed {i}/{len(samples)} samples...")
                    
            except Exception as e:
                logger.warning(f"Error processing {sample['file_path']}: {e}")
                continue
        
        logger.info(f"Extracted features from {len(all_features)} samples")
        
        # Analyze and learn parameters
        self._learn_model_parameters(all_features)
        
        # Save model
        if output_model_path:
            self.save_model(output_model_path)
        
        logger.info("Training complete!")
        
    def _learn_model_parameters(self, all_features: List[Dict]):
        """Learn model parameters from training data - embracingearth.space"""
        
        logger.info("Learning model parameters...")
        
        # Learn note->frequency mapping
        for feat in all_features:
            note = feat['note']
            f0_mean = np.nanmean(feat['f0_hz'][~np.isnan(feat['f0_hz'])])
            
            if note not in self.note_freq_map or f0_mean > 0:
                # Store frequency for this note
                if note not in self.note_freq_map:
                    self.note_freq_map[note] = []
                if f0_mean > 0:
                    self.note_freq_map[note].append(f0_mean)
        
        # Average frequencies for each note
        for note in self.note_freq_map:
            self.note_freq_map[note] = np.mean(self.note_freq_map[note])
        
        logger.info(f"Learned {len(self.note_freq_map)} note frequencies")
        
        # Learn dynamic profiles
        dynamic_levels = ['pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff']
        
        for dyn in dynamic_levels:
            # Find samples with this dynamic
            dyn_samples = [f for f in all_features if f['dynamic'] == dyn]
            
            if dyn_samples:
                # Average harmonic profiles
                avg_harmonic_profile = np.zeros(self.config.n_harmonics)
                count = 0
                
                for sample in dyn_samples:
                    if len(sample['harmonic_amplitudes']) > 0:
                        # Average across all frames
                        profile = np.mean(sample['harmonic_amplitudes'], axis=0)
                        if len(profile) == self.config.n_harmonics:
                            avg_harmonic_profile += profile
                            count += 1
                
                if count > 0:
                    avg_harmonic_profile /= count
                    # Normalize
                    if np.max(avg_harmonic_profile) > 0:
                        avg_harmonic_profile = avg_harmonic_profile / np.max(avg_harmonic_profile)
                
                self.dynamic_profile[dyn] = avg_harmonic_profile
        
        logger.info(f"Learned {len(self.dynamic_profile)} dynamic profiles")
        
        # Learn average spectral envelope
        all_spectra = []
        for feat in all_features:
            if hasattr(feat, 'magnitude'):
                all_spectra.append(np.mean(feat['magnitude'], axis=1))
        
        if all_spectra:
            self.spectral_envelope = np.mean(all_spectra, axis=0)
        
        logger.info("Model parameters learned!")
    
    def save_model(self, model_path: Path):
        """Save trained model - embracingearth.space"""
        
        model_data = {
            'config': asdict(self.config),
            'note_freq_map': self.note_freq_map,
            'dynamic_profile': {k: v.tolist() for k, v in self.dynamic_profile.items()},
            'spectral_envelope': self.spectral_envelope.tolist() if self.spectral_envelope is not None else None
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path):
        """Load trained model - embracingearth.space"""
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = TrainingConfig(**model_data['config'])
        self.note_freq_map = model_data['note_freq_map']
        self.dynamic_profile = {k: np.array(v) for k, v in model_data['dynamic_profile'].items()}
        if model_data['spectral_envelope'] is not None:
            self.spectral_envelope = np.array(model_data['spectral_envelope'])
        
        logger.info(f"Model loaded from {model_path}")
        
    def synthesize_from_midi_notes(self, midi_notes: List[Dict], duration: float) -> np.ndarray:
        """Synthesize audio from MIDI notes using trained model - embracingearth.space"""
        
        if not self.note_freq_map:
            raise ValueError("Model not trained! Call train() first.")
        
        sr = self.config.sample_rate
        # Use the duration parameter passed in (from MIDI parsing)
        total_samples = int(duration * sr)
        audio = np.zeros(total_samples)
        
        # Generate audio for each MIDI note
        for note_info in midi_notes:
            pitch = note_info['pitch']  # MIDI pitch (0-127)
            velocity = note_info['velocity']  # MIDI velocity (0-127)
            start_time = note_info['start']
            end_time = note_info['end']
            
            # Convert MIDI pitch to frequency
            f0_hz = librosa.midi_to_hz(pitch)
            
            # Convert velocity to dynamic level
            dynamic = self._velocity_to_dynamic(velocity)
            
            # Generate audio for this note - ensure exact sample count
            note_duration = end_time - start_time
            if note_duration <= 0:
                continue
                
            note_audio = self._synthesize_note(
                f0_hz=f0_hz,
                duration=note_duration,
                dynamic=dynamic
            )
            
            # Place note in audio timeline - use exact integer sample counts
            start_sample = int(np.round(start_time * sr))
            end_sample = int(np.round(end_time * sr))
            
            # Ensure bounds
            if start_sample >= total_samples:
                continue
            if end_sample > total_samples:
                end_sample = total_samples
            
            # Calculate exact length needed for this slice
            slice_length = end_sample - start_sample
            actual_note_len = len(note_audio)
            
            # Match note_audio length to slice_length exactly
            if actual_note_len > slice_length:
                # Truncate to exact length
                note_audio = note_audio[:slice_length].copy()
            elif actual_note_len < slice_length:
                # Pad with sustain - extend the note
                pad_length = slice_length - actual_note_len
                
                if actual_note_len > 0:
                    # Get sustain value from last portion of note
                    sustain_samples = min(actual_note_len // 10, 240)  # ~5ms at 48kHz
                    if sustain_samples > 0:
                        sustain_val = float(np.mean(note_audio[-sustain_samples:]))
                    else:
                        sustain_val = float(note_audio[-1] if len(note_audio) > 0 else 0.0)
                    
                    # Create exponential decay padding
                    decay_curve = np.exp(-np.linspace(0.0, 5.0, pad_length)).astype(np.float32)
                    padding = np.full(pad_length, sustain_val, dtype=np.float32) * decay_curve
                    
                    # Concatenate ensuring same dtype
                    note_audio = np.concatenate([note_audio.astype(np.float32), padding])
                else:
                    # Empty note - fill with zeros
                    note_audio = np.zeros(slice_length, dtype=np.float32)
            else:
                # Exact match - just copy to ensure consistency
                note_audio = note_audio.copy()
            
            # Final length check
            assert len(note_audio) == slice_length, f"CRITICAL: Length mismatch {len(note_audio)} != {slice_length}"
            
            # Final bounds check
            assert start_sample + slice_length <= total_samples, f"Bounds error: {start_sample}+{slice_length} > {total_samples}"
            
            # Now lengths match exactly - add to audio
            audio[start_sample:start_sample + slice_length] += note_audio
        
        return audio
    
    def _synthesize_note(self, f0_hz: float, duration: float, dynamic: str = 'mf') -> np.ndarray:
        """Synthesize a single note - embracingearth.space"""
        
        sr = self.config.sample_rate
        n_samples = int(np.round(duration * sr))  # Use round to match array math exactly
        if n_samples <= 0:
            n_samples = 1
        
        # Generate time axis - use exact duration to avoid rounding issues
        t = np.arange(n_samples) / sr
        
        # Generate harmonic content
        harmonics = list(range(1, self.config.n_harmonics + 1))
        
        # Get dynamic profile (harmonic amplitudes)
        if dynamic in self.dynamic_profile:
            harmonic_amps = self.dynamic_profile[dynamic]
        else:
            # Default to mf
            harmonic_amps = self.dynamic_profile.get('mf', np.ones(self.config.n_harmonics) / harmonics)
        
        # Generate harmonic audio
        audio = np.zeros(n_samples)
        
        for i, (harm, amp) in enumerate(zip(harmonics, harmonic_amps)):
            freq = f0_hz * harm
            if freq < sr / 2:  # Nyquist limit
                audio += amp * np.sin(2 * np.pi * freq * t)
        
        # Apply ADSR envelope for realistic attack/release
        audio = self._apply_adsr_envelope(audio, duration, sr)
        
        # Add noise content
        noise = self._generate_noise(duration, dynamic)
        audio = audio + 0.1 * noise
        
        # Normalize to higher level for fuller sound
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 1.0  # Use full dynamic range
        
        return audio
    
    def _apply_adsr_envelope(self, audio: np.ndarray, duration: float, sr: int) -> np.ndarray:
        """Apply ADSR envelope for realistic cello sound - embracingearth.space"""
        
        # ADSR parameters - increased for fuller sound
        attack_time = 0.05  # 50ms attack (faster to full sound)
        decay_time = 0.05   # 50ms decay (faster to sustain)
        sustain_level = 0.85  # 85% sustain (hold strong)
        release_time = 0.6   # 600ms release (longer tail)
        
        # Create envelope
        n_samples = len(audio)
        envelope = np.ones(n_samples)
        
        # Attack phase (exponential curve)
        attack_samples = int(attack_time * sr)
        if attack_samples > 0:
            attack_curve = np.linspace(0, 1, attack_samples) ** 2
            envelope[:attack_samples] = attack_curve
        
        # Decay phase
        decay_samples = int(decay_time * sr)
        if decay_samples > 0 and attack_samples + decay_samples < n_samples:
            decay_start = attack_samples
            decay_curve = np.linspace(1, sustain_level, decay_samples)
            envelope[decay_start:decay_start + decay_samples] = decay_curve
        
        # Sustain phase
        sustain_start = min(attack_samples + decay_samples, n_samples - int(release_time * sr))
        sustain_samples = n_samples - sustain_start - int(release_time * sr)
        if sustain_samples > 0:
            envelope[sustain_start:sustain_start + sustain_samples] = sustain_level
        
        # Release phase (exponential curve)
        release_samples = min(int(release_time * sr), n_samples - sustain_start - sustain_samples)
        if release_samples > 0:
            release_start = n_samples - release_samples
            release_curve = np.linspace(sustain_level, 0, release_samples) ** 1.5
            envelope[release_start:] = release_curve
        
        return audio * envelope
    
    def _generate_noise(self, duration: float, dynamic: str) -> np.ndarray:
        """Generate filtered noise for cello texture - embracingearth.space"""
        
        sr = self.config.sample_rate
        n_samples = int(duration * sr)
        
        # Generate white noise
        noise = np.random.randn(n_samples)
        
        # Low-pass filter for cello-like noise
        nyquist = sr / 2
        cutoff = 2000 / nyquist  # 2kHz cutoff
        b, a = signal.butter(4, cutoff, btype='low')
        filtered_noise = signal.filtfilt(b, a, noise)
        
        # Scale by dynamic
        dynamic_scale = {'pp': 0.05, 'p': 0.1, 'mp': 0.15, 'mf': 0.2, 'f': 0.3, 'ff': 0.4, 'fff': 0.5}
        scale = dynamic_scale.get(dynamic, 0.2)
        
        return filtered_noise * scale
    
    def _velocity_to_dynamic(self, velocity: int) -> str:
        """Convert MIDI velocity to dynamic level - embracingearth.space"""
        
        if velocity <= 20:
            return 'pp'
        elif velocity <= 40:
            return 'p'
        elif velocity <= 64:
            return 'mp'
        elif velocity <= 96:
            return 'mf'
        elif velocity <= 112:
            return 'f'
        elif velocity <= 124:
            return 'ff'
        else:
            return 'fff'

def main():
    """Train DDSP model - embracingearth.space"""
    
    # Configuration
    config = TrainingConfig()
    
    # Initialize trainer
    trainer = DDSPTrainer(config)
    
    # Train on cello samples
    samples_dir = Path("neural_cello_training_samples/filtered_20250808_010724_batch")
    model_path = Path("models/cello_ddsp_model.pkl")
    
    trainer.train(samples_dir, model_path)
    
    print(f"\n[DONE] Trained model saved to {model_path}")
    print(f"Model parameters:")
    print(f"  - Learned {len(trainer.note_freq_map)} note frequencies")
    print(f"  - Learned {len(trainer.dynamic_profile)} dynamic profiles")
    print(f"  - Ready for synthesis!")

if __name__ == "__main__":
    main()

