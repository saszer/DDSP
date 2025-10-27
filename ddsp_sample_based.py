"""
Sample-based DDSP synthesis using real cello samples - embracingearth.space
Uses actual recorded cello samples instead of synthetic generation
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleBasedDDSP:
    """Uses real cello samples for synthesis - embracingearth.space"""
    
    def __init__(self):
        self.sample_rate = 48000
        self.samples_library = {}  # pitch -> list of audio samples
        self.is_loaded = False
        
    def load_samples(self, samples_dir: Path):
        """Load all cello samples into memory"""
        
        logger.info(f"Loading cello samples from {samples_dir}")
        
        wav_files = list(samples_dir.glob("*.wav"))
        logger.info(f"Found {len(wav_files)} samples")
        
        loaded_count = 0
        for wav_file in wav_files:
            try:
                # Parse filename to get pitch
                name_parts = wav_file.stem.split('_')
                if len(name_parts) >= 1:
                    note_name = name_parts[0]  # e.g., "C4"
                    
                    # Convert note name to MIDI pitch
                    try:
                        midi_pitch = librosa.note_to_midi(note_name)
                    except:
                        continue
                    
                    # Load audio
                    audio, sr = sf.read(wav_file)
                    
                    # Resample if needed
                    if sr != self.sample_rate:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    
                    # Store sample
                    if midi_pitch not in self.samples_library:
                        self.samples_library[midi_pitch] = []
                    
                    self.samples_library[midi_pitch].append(audio)
                    loaded_count += 1
                    
            except Exception as e:
                logger.warning(f"Error loading {wav_file}: {e}")
                continue
        
        logger.info(f"Loaded {loaded_count} samples covering {len(self.samples_library)} pitches")
        self.is_loaded = len(self.samples_library) > 0
    
    def _synthesize_from_config(self, pitch: int, duration: float, velocity: int, 
                                release_percent: float, tone: str) -> np.ndarray:
        """Synthesize using config-based model (new format) - embracingearth.space"""
        
        logger.info(f"[SYNTH DEBUG] Starting synthesis: pitch={pitch}, duration={duration}, velocity={velocity}, release={release_percent}%, tone={tone}")
        
        # Frequency from MIDI (fallback if no explicit map for pitch)
        freq_hz = librosa.midi_to_hz(pitch)
        logger.info(f"[SYNTH DEBUG] Frequency: {freq_hz:.2f} Hz")
        
        # Duration and time vector
        n_samples = int(max(1, round(duration * self.sample_rate)))
        t = np.arange(n_samples) / float(self.sample_rate)
        logger.info(f"[SYNTH DEBUG] Generating {n_samples} samples ({duration:.2f}s at {self.sample_rate}Hz)")
        
        # Determine harmonic count from trained config
        trained_config = getattr(self, 'config', {}) or {}
        n_harmonics = int(trained_config.get('n_harmonics', 60))
        logger.info(f"[SYNTH DEBUG] Using {n_harmonics} harmonics")
        
        # Map velocity to dynamic and fetch trained harmonic profile
        dynamic = self._velocity_to_dynamic(velocity)
        logger.info(f"[SYNTH DEBUG] Velocity {velocity} -> dynamic '{dynamic}'")
        
        harmonic_profile = None
        if hasattr(self, 'dynamic_profile') and isinstance(self.dynamic_profile, dict):
            logger.info(f"[SYNTH DEBUG] dynamic_profile type: {type(self.dynamic_profile)}, keys: {list(self.dynamic_profile.keys())}")
            harmonic_profile = self.dynamic_profile.get(dynamic)
            logger.info(f"[SYNTH DEBUG] Harmonic profile for '{dynamic}': {harmonic_profile is not None}")
            if harmonic_profile is not None:
                logger.info(f"[SYNTH DEBUG] Harmonic profile shape: {np.asarray(harmonic_profile).shape}, first few values: {np.asarray(harmonic_profile)[:5]}")
        
        if harmonic_profile is None:
            logger.warning(f"[SYNTH DEBUG] No harmonic profile found, using fallback")
            # Fallback: gentle roll-off
            harmonic_profile = np.linspace(1.0, 0.1, n_harmonics)
        else:
            # Ensure correct length
            harmonic_profile = np.asarray(harmonic_profile)
            if len(harmonic_profile) < n_harmonics:
                pad_len = n_harmonics - len(harmonic_profile)
                harmonic_profile = np.pad(harmonic_profile, (0, pad_len), mode='edge')
            elif len(harmonic_profile) > n_harmonics:
                harmonic_profile = harmonic_profile[:n_harmonics]
        
        # Optional spectral envelope weighting per harmonic
        spectral_weight = np.ones(n_harmonics)
        spectral_envelope = getattr(self, 'spectral_envelope', None)
        logger.info(f"[SYNTH DEBUG] spectral_envelope available: {spectral_envelope is not None}")
        if spectral_envelope is not None:
            try:
                spectral_envelope = np.asarray(spectral_envelope)
                logger.info(f"[SYNTH DEBUG] spectral_envelope shape: {spectral_envelope.shape}")
                n_fft = int(trained_config.get('n_fft', 2048))
                # Estimate bin per harmonic and sample envelope
                freqs = np.linspace(0, self.sample_rate / 2.0, n_fft // 2 + 1)
                weights = []
                for h in range(1, n_harmonics + 1):
                    fhz = freq_hz * h
                    if fhz >= freqs[-1]:
                        weights.append(0.0)
                    else:
                        idx = int(np.argmin(np.abs(freqs - fhz)))
                        weights.append(float(spectral_envelope[min(idx, len(spectral_envelope) - 1)]))
                weights = np.asarray(weights)
                # Normalize weights to max 1 to avoid huge boosts
                if np.max(weights) > 0:
                    spectral_weight = weights / np.max(weights)
                logger.info(f"[SYNTH DEBUG] Applied spectral envelope, weight range: [{np.min(spectral_weight):.3f}, {np.max(spectral_weight):.3f}]")
            except Exception as e:
                logger.error(f"[SYNTH DEBUG] Error applying spectral envelope: {e}")
                # If anything goes wrong, keep flat weighting
                spectral_weight = np.ones(n_harmonics)
        
        # Sum harmonics using trained amplitudes with pitch-dependent natural roll-off
        audio = np.zeros(n_samples, dtype=np.float32)
        active_harmonics = 0
        for h in range(1, n_harmonics + 1):
            amp = float(harmonic_profile[h - 1]) * float(spectral_weight[h - 1])
            fhz = freq_hz * h
            
            # Apply natural pitch-dependent roll-off (higher pitches have fewer strong harmonics)
            pitch_rolloff = 1.0 / (1.0 + (fhz / 1000.0) ** 1.5)  # Gentle roll-off above 1kHz
            amp *= pitch_rolloff
            
            if fhz < self.sample_rate / 2.0 and amp > 1e-6:
                audio += amp * np.sin(2 * np.pi * fhz * t).astype(np.float32)
                active_harmonics += 1
        
        logger.info(f"[SYNTH DEBUG] Generated audio with {active_harmonics} active harmonics, RMS: {np.sqrt(np.mean(audio**2)):.6f}")
        
        # Apply tone EQ (pre-envelope) if requested
        if tone != 'standard':
            audio = self._apply_tone_eq(audio, tone)
        
        # ADSR envelope with release controlled by slider
        attack_time = 0.02
        decay_time = 0.08
        sustain_level = 0.8
        release_time = max(0.05, (release_percent / 100.0) * 0.6)
        logger.info(f"[SYNTH DEBUG] Envelope: attack={attack_time}s, decay={decay_time}s, sustain={sustain_level}, release={release_time}s")
        env = self._generate_envelope(n_samples, self.sample_rate, attack_time, decay_time, sustain_level, release_time)
        audio *= env.astype(audio.dtype)
        
        # Velocity scaling (match trainer's perceptual curve)
        velocity_scale = (velocity / 127.0) ** 0.5
        logger.info(f"[SYNTH DEBUG] Velocity scale: {velocity_scale:.3f}")
        audio *= velocity_scale
        
        # Preserve trained model's natural amplitude ratios - don't over-normalize
        # The model was trained to produce realistic amplitudes, so trust it
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        logger.info(f"[SYNTH DEBUG] Peak before scaling: {peak:.6f}")
        if peak > 0:
            # Only normalize if significantly over unity (clipping risk)
            if peak > 1.2:
                audio = (audio / peak) * 0.95
                logger.info(f"[SYNTH DEBUG] Normalized to prevent clipping")
            elif peak < 0.1:
                # Boost very quiet signals
                audio = audio * (0.3 / peak)
                logger.info(f"[SYNTH DEBUG] Boosted quiet signal")
        
        logger.info(f"[SYNTH DEBUG] Final audio: {len(audio)} samples, RMS: {np.sqrt(np.mean(audio**2)):.6f}, peak: {np.max(np.abs(audio)):.6f}")
        
        return audio
    
    def _generate_envelope(self, n_samples: int, sr: int, attack: float, 
                          decay: float, sustain: float, release: float) -> np.ndarray:
        """Generate ADSR envelope - embracingearth.space"""
        envelope = np.zeros(n_samples)
        
        attack_samples = int(attack * sr)
        decay_samples = int(decay * sr)
        release_samples = int(release * sr)
        
        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, n_samples)
        for i in range(decay_start, decay_end):
            envelope[i] = 1.0 - (i - decay_start) / decay_samples * (1.0 - sustain)
        
        # Sustain
        sustain_start = decay_end
        sustain_end = max(0, n_samples - release_samples)
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = sustain
        
        # Release
        release_start = max(0, n_samples - release_samples)
        for i in range(release_start, n_samples):
            envelope[i] = sustain * (1.0 - (i - release_start) / release_samples)
        
        return envelope
        
    def synthesize_note(self, pitch: int, duration: float, velocity: int, 
                       release_percent: float = 100.0, tone: str = 'standard') -> np.ndarray:
        """Synthesize a note using real samples or config-based model - embracingearth.space"""
        
        # Check if we're using config-based model (new format)
        if hasattr(self, 'note_freq_map') and hasattr(self, 'spectral_envelope'):
            return self._synthesize_from_config(pitch, duration, velocity, release_percent, tone)
        
        # Otherwise use sample-based approach
        # Find closest pitch in library
        if pitch in self.samples_library:
            samples = self.samples_library[pitch]
        else:
            # Find nearest pitch
            available_pitches = list(self.samples_library.keys())
            if not available_pitches:
                return np.zeros(int(duration * self.sample_rate))
            
            nearest_pitch = min(available_pitches, key=lambda x: abs(x - pitch))
            samples = self.samples_library[nearest_pitch]
            
            # Calculate pitch shift needed
            semitone_shift = pitch - nearest_pitch
        
        # Select random sample for variation
        sample = samples[np.random.randint(len(samples))].copy()
        
        # Apply pitch shift if needed
        if 'semitone_shift' in locals() and semitone_shift != 0:
            sample = librosa.effects.pitch_shift(
                sample, sr=self.sample_rate, n_steps=semitone_shift
            )
        
        # Apply tone EQ before processing
        if tone != 'standard':
            sample = self._apply_tone_eq(sample, tone)
        
        # Adjust duration with release control
        target_samples = int(duration * self.sample_rate)
        release_samples = int((release_percent / 100.0) * duration * self.sample_rate)
        
        if len(sample) > release_samples:
            # Truncate based on release setting
            fade_samples = int(0.05 * self.sample_rate)  # 50ms fade
            sample = sample[:release_samples]
            if len(sample) > fade_samples:
                fade = np.linspace(1, 0, fade_samples)
                sample[-fade_samples:] *= fade
        elif len(sample) < target_samples:
            # Loop and fade (for sustain)
            loops_needed = target_samples // len(sample) + 1
            sample = np.tile(sample, loops_needed)[:target_samples]
            
            # Smooth loop points
            fade_samples = int(0.01 * self.sample_rate)  # 10ms crossfade
            for i in range(1, loops_needed):
                loop_point = i * len(sample)
                if loop_point - fade_samples >= 0 and loop_point + fade_samples < len(sample):
                    # Crossfade at loop point
                    fade_out = np.linspace(1, 0, fade_samples)
                    fade_in = np.linspace(0, 1, fade_samples)
                    sample[loop_point-fade_samples:loop_point] *= fade_out
                    sample[loop_point:loop_point+fade_samples] *= fade_in
        
        # Apply velocity scaling
        velocity_scale = (velocity / 127.0) ** 0.7
        sample *= velocity_scale
        
        return sample
    
    def _velocity_to_dynamic(self, velocity: int) -> str:
        """Convert MIDI velocity to dynamic level consistent with training."""
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
    
    def _apply_tone_eq(self, audio: np.ndarray, tone: str) -> np.ndarray:
        """Apply tone-based EQ for different cello timbres"""
        from scipy import signal
        
        if tone == 'warm':
            # Slight bass boost
            b, a = signal.butter(2, 300/(self.sample_rate/2), 'low')
            return signal.lfilter(b, a, audio) * 1.1
        elif tone == 'bright':
            # High frequency boost
            b, a = signal.butter(2, 2000/(self.sample_rate/2), 'high')
            return audio + signal.lfilter(b, a, audio) * 0.3
        elif tone == 'dark':
            # High frequency cut
            b, a = signal.butter(4, 1000/(self.sample_rate/2), 'low')
            return signal.lfilter(b, a, audio) * 1.05
        elif tone == 'vintage':
            # Slight mid cut for vintage sound
            b, a = signal.butter(4, [400/(self.sample_rate/2), 4000/(self.sample_rate/2)], 'bandstop')
            return signal.lfilter(b, a, audio) * 0.95
        else:
            return audio
    
    def synthesize_from_midi(self, midi_notes: List[Dict], duration: float, 
                             release_percent: float = 100.0, tone: str = 'standard') -> np.ndarray:
        """Synthesize audio from MIDI using real samples"""
        
        if not self.is_loaded:
            raise ValueError("No samples loaded!")
        
        logger.info(f"[MIDI SYNTH] Processing {len(midi_notes)} notes, duration={duration:.2f}s, release={release_percent}%, tone={tone}")
        
        total_samples = int(duration * self.sample_rate)
        output = np.zeros(total_samples)
        
        for idx, note_info in enumerate(midi_notes):
            pitch = note_info['pitch']
            velocity = note_info['velocity']
            start_time = note_info['start']
            end_time = note_info['end']
            note_duration = end_time - start_time
            
            if note_duration <= 0:
                continue
            
            logger.info(f"[MIDI SYNTH] Note {idx+1}/{len(midi_notes)}: pitch={pitch}, velocity={velocity}, start={start_time:.3f}s, duration={note_duration:.3f}s")
            
            # Generate note using real sample with tone and release
            note_audio = self.synthesize_note(pitch, note_duration, velocity, release_percent, tone)
            logger.info(f"[MIDI SYNTH] Generated audio for note {idx+1}: {len(note_audio)} samples, peak={np.max(np.abs(note_audio)):.6f}")
            
            # Place in output
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + len(note_audio)
            
            if start_sample < total_samples:
                if end_sample > total_samples:
                    end_sample = total_samples
                    note_audio = note_audio[:end_sample - start_sample]
                
                # Mix with overlap
                output[start_sample:end_sample] += note_audio
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0.9:
            output = output / max_val * 0.9
        
        return output
    
    def save(self, path: Path):
        """Save the sample library"""
        with open(path, 'wb') as f:
            pickle.dump({
                'samples_library': self.samples_library,
                'sample_rate': self.sample_rate
            }, f)
    
    def load(self, path: Path):
        """Load saved sample library - embracingearth.space"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
            # Handle both old format (with samples_library) and new format (config-based)
            if 'samples_library' in data:
                # Old format: sample-based model
                self.samples_library = data['samples_library']
                self.sample_rate = data.get('sample_rate', self.sample_rate)
                self.is_loaded = len(self.samples_library) > 0
            elif 'config' in data:
                # New format: config-based model (frequency mapping)
                logger.info("Loading config-based model (frequency mapping)")
                # Store the model data for synthesis
                self.config = data.get('config', {})
                self.note_freq_map = data.get('note_freq_map', {})
                self.dynamic_profile = data.get('dynamic_profile', {})
                self.spectral_envelope = data.get('spectral_envelope', {})
                self.is_loaded = True
                logger.info("Config-based model loaded successfully")
            else:
                logger.error(f"Unknown model format in {path}")
                self.is_loaded = False


# Quick test
if __name__ == "__main__":
    # Create sample-based synthesizer
    synth = SampleBasedDDSP()
    
    # Load cello samples
    samples_dir = Path("neural_cello_training_samples/filtered_20250808_010724_batch")
    synth.load_samples(samples_dir)
    
    # Save for quick loading
    synth.save(Path("models/cello_samples.pkl"))
    
    print(f"Sample-based model ready!")
    print(f"Loaded samples for {len(synth.samples_library)} different pitches")
