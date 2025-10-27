"""Deep audit of synthesis pipeline - embracingearth.space"""

import numpy as np
import soundfile as sf
import pretty_midi
import sys

def audit_generated_audio():
    """Deep audit of generated audio"""
    
    print("="*60)
    print("DEEP AUDIT - DDSP Neural Cello")
    print("="*60)
    print("\n[STEP 1] Analyzing MIDI Input...")
    
    # Load MIDI
    midi_file = "MIDI Files/MIDI Files/Cello Arpegio.mid"
    midi = pretty_midi.PrettyMIDI(midi_file)
    
    all_notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            all_notes.append({
                'pitch': note.pitch,
                'velocity': note.velocity,
                'start': note.start,
                'end': note.end,
                'duration': note.end - note.start
            })
    
    print(f"  Total notes: {len(all_notes)}")
    print(f"  Duration range: {min(n['start'] for n in all_notes):.2f}s - {max(n['end'] for n in all_notes):.2f}s")
    print(f"  Note durations: {min(n['duration'] for n in all_notes):.3f}s - {max(n['duration'] for n in all_notes):.3f}s")
    print(f"  Pitch range: {min(n['pitch'] for n in all_notes)} - {max(n['pitch'] for n in all_notes)}")
    
    # Show notes over time
    print("\n  Timeline (first 2 seconds):")
    for note in sorted(all_notes, key=lambda x: x['start'])[:10]:
        print(f"    {note['start']:.2f}s-{note['end']:.2f}s: Pitch {note['pitch']}, Vel {note['velocity']}")
    
    print("\n[STEP 2] Analyzing Generated Audio...")
    
    # Load audio
    audio_file = "test_output_synthesis_Cello Arpegio.wav"
    audio, sr = sf.read(audio_file)
    
    print(f"  Duration: {len(audio) / sr:.2f}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Samples: {len(audio)}")
    print(f"  Max amplitude: {np.max(np.abs(audio)):.4f}")
    
    # Analyze audio in 100ms chunks
    chunk_size = sr // 10
    num_chunks = len(audio) // chunk_size
    
    chunk_energies = []
    chunk_rms = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio[start:end]
        energy = np.sum(chunk ** 2)
        rms = np.sqrt(np.mean(chunk ** 2))
        chunk_energies.append(energy)
        chunk_rms.append(rms)
    
    print(f"\n  Energy analysis ({num_chunks} chunks):")
    print(f"    Mean energy: {np.mean(chunk_energies):.2f}")
    print(f"    Max energy: {np.max(chunk_energies):.2f}")
    print(f"    Energy std: {np.std(chunk_energies):.2f}")
    print(f"    Active chunks (>10% mean): {sum(1 for e in chunk_energies if e > np.mean(chunk_energies) * 0.1)}")
    
    # Check for silence
    silent_chunks = sum(1 for rms in chunk_rms if rms < 0.001)
    print(f"    Silent chunks: {silent_chunks}/{num_chunks}")
    
    # Compare MIDI notes to audio energy
    print("\n[STEP 3] MIDI-to-Audio Alignment Check...")
    
    # For each MIDI note, check if corresponding audio has energy
    aligned_count = 0
    for note in all_notes[:20]:  # Check first 20 notes
        note_start_sample = int(note['start'] * sr)
        note_end_sample = int(note['end'] * sr)
        
        if note_start_sample < len(audio) and note_end_sample <= len(audio):
            note_audio = audio[note_start_sample:note_end_sample]
            note_energy = np.sum(note_audio ** 2)
            
            if note_energy > 0.001:  # Has significant energy
                aligned_count += 1
    
    print(f"  Notes with audio energy: {aligned_count}/20 (checked first 20)")
    
    # Frequency analysis
    print("\n[STEP 4] Frequency Content Analysis...")
    
    # Use librosa for spectral analysis
    try:
        import librosa
        import librosa.display
        
        # Get dominant frequencies over time
        hop_length = 512
        frame_length = 2048
        
        D = librosa.stft(audio, hop_length=hop_length, n_fft=frame_length)
        S_db = librosa.amplitude_to_db(np.abs(D))
        
        # Find dominant frequency in each frame
        dominant_freqs = []
        for frame in S_db.T:
            # Find peak in frequency domain
            peak_idx = np.argmax(frame)
            freq_bin = peak_idx * sr / frame_length
            dominant_freqs.append(freq_bin)
        
        print(f"  Dominant frequencies: {min(dominant_freqs):.1f} Hz - {max(dominant_freqs):.1f} Hz")
        print(f"  Frequency variation: {np.std(dominant_freqs):.1f} Hz")
        
        # Check if frequencies match MIDI pitches
        midi_freqs = [librosa.midi_to_hz(n['pitch']) for n in all_notes]
        print(f"  MIDI frequency range: {min(midi_freqs):.1f} Hz - {max(midi_freqs):.1f} Hz")
        
    except ImportError:
        print("  librosa not available, skipping frequency analysis")
    
    print("\n[STEP 5] Audio Quality Assessment...")
    
    # Check for clipping
    clipped = np.sum(np.abs(audio) > 0.95)
    print(f"  Clipped samples: {clipped}/{len(audio)} ({100*clipped/len(audio):.2f}%)")
    
    # Check dynamic range
    dynamic_range = np.max(audio) - np.min(audio)
    print(f"  Dynamic range: {dynamic_range:.4f}")
    
    # Check for DC offset
    dc_offset = np.mean(audio)
    print(f"  DC offset: {dc_offset:.6f}")
    
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    
    issues = []
    if len(all_notes) > 20 and aligned_count < 10:
        issues.append("Many MIDI notes don't have corresponding audio energy")
    if np.max(np.abs(audio)) < 0.01:
        issues.append("Audio has very low amplitude")
    if silent_chunks > num_chunks * 0.5:
        issues.append("More than 50% of audio is silent")
    
    if issues:
        print("\n[WARNING] Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nStatus: NEEDS ATTENTION")
    else:
        print("\n[OK] Audio generation appears correct")
        print("Status: PASSED")
    
    print("="*60)

if __name__ == "__main__":
    audit_generated_audio()

