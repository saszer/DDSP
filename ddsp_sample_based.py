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
        
        # Get frequency from note map
        freq_hz = librosa.midi_to_hz(pitch)  # Default to librosa conversion
        
        # Generate samples for duration
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Generate cello-like sound with harmonics
        audio = np.zeros(n_samples)
        
        # Fundamental frequency
        audio += 0.5 * np.sin(2 * np.pi * freq_hz * t)
        
        # Add harmonics for cello character
        harmonics = [2, 3, 4, 5]
        amplitudes = [0.3, 0.2, 0.1, 0.05]
        for harmonic, amp in zip(harmonics, amplitudes):
            audio += amp * np.sin(2 * np.pi * freq_hz * harmonic * t)
        
        # Apply envelope based on release_percent
        attack_time = 0.01
        decay_time = 0.1
        sustain_level = 0.7
        release_time = (release_percent / 100.0) * 0.5  # Adjust release based on slider
        
        envelope = self._generate_envelope(n_samples, self.sample_rate, 
                                          attack_time, decay_time, sustain_level, release_time)
        audio *= envelope
        
        # Apply velocity scaling
        velocity_scale = (velocity / 127.0) ** 0.5
        audio *= velocity_scale
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
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
        
        total_samples = int(duration * self.sample_rate)
        output = np.zeros(total_samples)
        
        for note_info in midi_notes:
            pitch = note_info['pitch']
            velocity = note_info['velocity']
            start_time = note_info['start']
            end_time = note_info['end']
            note_duration = end_time - start_time
            
            if note_duration <= 0:
                continue
            
            # Generate note using real sample with tone and release
            note_audio = self.synthesize_note(pitch, note_duration, velocity, release_percent, tone)
            
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
                # New format: config-based model (frequency maps and profiles)
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
