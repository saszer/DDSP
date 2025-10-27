"""
Professional Audio Processing Pipeline - embracingearth.space
Enterprise-grade audio quality for DDSP neural cello synthesis
"""

import numpy as np
import librosa
import soundfile as sf
import resampy
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Professional audio libraries - embracingearth.space quality standards
try:
    import pyworld as pw
    PYWORLD_AVAILABLE = True
except ImportError:
    PYWORLD_AVAILABLE = False
    logging.warning("PyWorld not available - using librosa F0 estimation")

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    logging.warning("CREPE not available - using librosa pitch tracking")

try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logging.warning("Essentia not available - using librosa features")

@dataclass
class AudioQualityConfig:
    """Professional audio quality configuration - embracingearth.space standards"""
    # Sample rates - professional audio standards
    SAMPLE_RATE: int = 48000  # Professional sample rate
    SAMPLE_RATE_TRAINING: int = 16000  # DDSP training rate
    BIT_DEPTH: int = 24  # Professional bit depth
    
    # Audio processing parameters
    HOP_LENGTH: int = 256  # Higher resolution for quality
    N_FFT: int = 4096  # Larger FFT for better frequency resolution
    WIN_LENGTH: int = 2048  # Window length for analysis
    
    # Quality thresholds
    SNR_THRESHOLD: float = 20.0  # Signal-to-noise ratio threshold
    DYNAMIC_RANGE_THRESHOLD: float = 40.0  # Dynamic range threshold
    
    # Professional mastering parameters
    LUFS_TARGET: float = -14.0  # Broadcast loudness standard
    TRUE_PEAK_LIMIT: float = -1.0  # True peak limiting
    
    # Cello-specific parameters
    CELLO_F0_MIN: float = 65.41  # C2 (lowest cello note)
    CELLO_F0_MAX: float = 1046.5  # C6 (highest cello note)
    CELLO_HARMONICS: int = 8  # Number of harmonics to model

class AudioQualityLevel(Enum):
    """Audio quality levels - embracingearth.space standards"""
    DRAFT = "draft"  # Fast processing, lower quality
    STANDARD = "standard"  # Balanced quality/speed
    PROFESSIONAL = "professional"  # High quality, slower
    MASTERING = "mastering"  # Maximum quality, slowest

class ProfessionalAudioProcessor:
    """Enterprise-grade audio processing pipeline - embracingearth.space"""
    
    def __init__(self, quality_level: AudioQualityLevel = AudioQualityLevel.PROFESSIONAL):
        self.quality_level = quality_level
        self.config = AudioQualityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set processing parameters based on quality level
        self._configure_quality_level()
    
    def _configure_quality_level(self):
        """Configure processing parameters based on quality level"""
        if self.quality_level == AudioQualityLevel.DRAFT:
            self.config.HOP_LENGTH = 512
            self.config.N_FFT = 2048
            self.config.WIN_LENGTH = 1024
        elif self.quality_level == AudioQualityLevel.STANDARD:
            self.config.HOP_LENGTH = 256
            self.config.N_FFT = 4096
            self.config.WIN_LENGTH = 2048
        elif self.quality_level == AudioQualityLevel.PROFESSIONAL:
            self.config.HOP_LENGTH = 128
            self.config.N_FFT = 8192
            self.config.WIN_LENGTH = 4096
        elif self.quality_level == AudioQualityLevel.MASTERING:
            self.config.HOP_LENGTH = 64
            self.config.N_FFT = 16384
            self.config.WIN_LENGTH = 8192
    
    def load_audio_high_quality(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio with professional quality resampling - embracingearth.space"""
        try:
            # Load at original sample rate first
            audio, orig_sr = librosa.load(file_path, sr=None)
            
            # Use professional resampling if target rate is different
            if target_sr and orig_sr != target_sr:
                # Use resampy for high-quality resampling
                audio = resampy.resample(audio, orig_sr, target_sr, filter='kaiser_best')
                sr = target_sr
            else:
                sr = orig_sr
            
            # Quality validation
            self._validate_audio_quality(audio, sr)
            
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"High-quality audio loading failed: {e}")
            raise
    
    def _validate_audio_quality(self, audio: np.ndarray, sr: int):
        """Validate audio quality metrics - embracingearth.space standards"""
        # Check for clipping
        if np.max(np.abs(audio)) >= 1.0:
            self.logger.warning("Audio clipping detected - quality may be compromised")
        
        # Check dynamic range
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))
        if dynamic_range < self.config.DYNAMIC_RANGE_THRESHOLD:
            self.logger.warning(f"Low dynamic range: {dynamic_range:.1f} dB")
        
        # Check for silence
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-6:
            self.logger.warning("Audio appears to be silent or very quiet")
    
    def extract_f0_professional(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract F0 using professional methods - embracingearth.space pitch tracking"""
        
        if CREPE_AVAILABLE and self.quality_level in [AudioQualityLevel.PROFESSIONAL, AudioQualityLevel.MASTERING]:
            # Use CREPE for highest quality F0 estimation
            self.logger.info("Using CREPE for professional F0 estimation")
            _, f0_hz, confidence, _ = crepe.predict(audio, sr, model_capacity='full', viterbi=True)
            
            # Apply confidence threshold
            f0_hz[confidence < 0.7] = 0
            
        elif PYWORLD_AVAILABLE:
            # Use PyWorld for professional F0 estimation
            self.logger.info("Using PyWorld for professional F0 estimation")
            
            # PyWorld parameters optimized for cello
            f0_hz, t = pw.harvest(
                audio.astype(np.float64),
                sr,
                f0_floor=self.config.CELLO_F0_MIN,
                f0_ceil=self.config.CELLO_F0_MAX,
                frame_period=self.config.HOP_LENGTH / sr * 1000
            )
            
            # Convert to Hz
            f0_hz = f0_hz.astype(np.float32)
            
        else:
            # Fallback to librosa YIN
            self.logger.info("Using librosa YIN for F0 estimation")
            f0_hz = librosa.yin(
                audio,
                fmin=self.config.CELLO_F0_MIN,
                fmax=self.config.CELLO_F0_MAX,
                sr=sr,
                hop_length=self.config.HOP_LENGTH
            )
        
        return f0_hz
    
    def extract_harmonic_content(self, audio: np.ndarray, sr: int, f0_hz: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract harmonic content for cello synthesis - embracingearth.space"""
        
        # Harmonic/percussive separation
        harmonic, percussive = librosa.effects.hpss(
            audio,
            kernel_size=31,
            power=2.0,
            margin=1.0
        )
        
        # Spectral analysis with high resolution
        stft = librosa.stft(
            audio,
            hop_length=self.config.HOP_LENGTH,
            n_fft=self.config.N_FFT,
            win_length=self.config.WIN_LENGTH,
            window='hann'
        )
        
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Extract harmonic series
        harmonic_series = self._extract_harmonic_series(magnitude, f0_hz, sr)
        
        return {
            'harmonic': harmonic,
            'percussive': percussive,
            'magnitude': magnitude,
            'phase': phase,
            'harmonic_series': harmonic_series,
            'f0_hz': f0_hz
        }
    
    def _extract_harmonic_series(self, magnitude: np.ndarray, f0_hz: np.ndarray, sr: int) -> np.ndarray:
        """Extract harmonic series for cello modeling - embracingearth.space"""
        
        n_frames = magnitude.shape[1]
        n_harmonics = self.config.CELLO_HARMONICS
        harmonic_series = np.zeros((n_harmonics, n_frames))
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.config.N_FFT)
        
        for frame_idx in range(n_frames):
            if f0_hz[frame_idx] > 0:
                # Find harmonic frequencies
                for harmonic_idx in range(1, n_harmonics + 1):
                    harmonic_freq = f0_hz[frame_idx] * harmonic_idx
                    
                    # Find closest frequency bin
                    freq_bin = np.argmin(np.abs(freqs - harmonic_freq))
                    
                    # Extract harmonic amplitude
                    if freq_bin < magnitude.shape[0]:
                        harmonic_series[harmonic_idx - 1, frame_idx] = magnitude[freq_bin, frame_idx]
        
        return harmonic_series
    
    def apply_professional_mastering(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply professional mastering chain - embracingearth.space"""
        
        # Normalize to prevent clipping
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Apply gentle compression for cello dynamics
        audio = self._apply_gentle_compression(audio, sr)
        
        # Apply EQ for cello frequency response
        audio = self._apply_cello_eq(audio, sr)
        
        # Apply subtle reverb for realism
        audio = self._apply_subtle_reverb(audio, sr)
        
        # Final limiting
        audio = self._apply_true_peak_limiting(audio, sr)
        
        return audio
    
    def _apply_gentle_compression(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply gentle compression for cello dynamics - embracingearth.space"""
        # Simple compression algorithm
        threshold = 0.7
        ratio = 3.0
        attack = 0.01
        release = 0.1
        
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
    
    def _apply_cello_eq(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply EQ optimized for cello frequency response - embracingearth.space"""
        # Cello frequency response enhancement
        # Boost fundamental range (65-500 Hz)
        # Gentle boost in presence range (2-5 kHz)
        
        # Simple EQ using FFT
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        
        # Create EQ curve
        eq_gain = np.ones_like(freqs)
        
        # Boost fundamental range
        fundamental_mask = (freqs >= 65) & (freqs <= 500)
        eq_gain[fundamental_mask] *= 1.2
        
        # Gentle presence boost
        presence_mask = (freqs >= 2000) & (freqs <= 5000)
        eq_gain[presence_mask] *= 1.1
        
        # Apply EQ
        eq_fft = fft * eq_gain
        eq_audio = np.real(np.fft.ifft(eq_fft))
        
        return eq_audio
    
    def _apply_subtle_reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply subtle reverb for cello realism - embracingearth.space"""
        # Simple convolution reverb
        reverb_length = int(0.5 * sr)  # 0.5 second reverb
        reverb_ir = np.random.randn(reverb_length) * np.exp(-np.linspace(0, 5, reverb_length))
        
        # Convolve with reverb
        reverb_audio = np.convolve(audio, reverb_ir, mode='same')
        
        # Mix dry and wet signals
        dry_wet_ratio = 0.8  # 80% dry, 20% wet
        mixed_audio = dry_wet_ratio * audio + (1 - dry_wet_ratio) * reverb_audio
        
        return mixed_audio
    
    def _apply_true_peak_limiting(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply true peak limiting - embracingearth.space mastering"""
        # Simple peak limiter
        threshold = 0.95
        
        # Apply soft limiting
        limited_audio = np.tanh(audio / threshold) * threshold
        
        return limited_audio
    
    def export_high_quality(self, audio: np.ndarray, sr: int, file_path: str, 
                          format: str = 'wav', bit_depth: int = 24) -> str:
        """Export audio with professional quality - embracingearth.space"""
        
        # Apply mastering
        mastered_audio = self.apply_professional_mastering(audio, sr)
        
        # Convert bit depth
        if bit_depth == 16:
            mastered_audio = (mastered_audio * 32767).astype(np.int16)
        elif bit_depth == 24:
            mastered_audio = (mastered_audio * 8388607).astype(np.int32)
        else:
            mastered_audio = mastered_audio.astype(np.float32)
        
        # Export with high quality settings
        if format.lower() == 'wav':
            sf.write(
                file_path,
                mastered_audio,
                sr,
                subtype='PCM_24' if bit_depth == 24 else 'PCM_16',
                format='WAV'
            )
        elif format.lower() == 'flac':
            sf.write(
                file_path,
                mastered_audio,
                sr,
                subtype='FLAC',
                format='FLAC'
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"High-quality audio exported: {file_path}")
        return file_path
    
    def analyze_audio_quality(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze audio quality metrics - embracingearth.space"""
        
        # Signal-to-noise ratio
        signal_power = np.mean(audio**2)
        noise_floor = np.percentile(np.abs(audio), 1)
        snr = 20 * np.log10(signal_power / (noise_floor**2 + 1e-10))
        
        # Dynamic range
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
        
        # Spectral centroid (brightness)
        stft = librosa.stft(audio, hop_length=self.config.HOP_LENGTH)
        magnitude = np.abs(stft)
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        avg_spectral_centroid = np.mean(spectral_centroid)
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
        
        return {
            'snr_db': snr,
            'dynamic_range_db': dynamic_range,
            'spectral_centroid_hz': avg_spectral_centroid,
            'zero_crossing_rate': zcr,
            'peak_level_db': 20 * np.log10(peak + 1e-10),
            'rms_level_db': 20 * np.log10(rms + 1e-10)
        }






