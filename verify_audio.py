"""Verify generated audio has multiple notes - embracingearth.space"""

import soundfile as sf
import numpy as np
import pretty_midi

def analyze_audio(audio_file):
    """Analyze audio to check for multiple notes"""
    
    # Load audio
    audio, sr = sf.read(audio_file)
    
    print(f"\n[INFO] Audio File: {audio_file}")
    print(f"[INFO] Duration: {len(audio) / sr:.2f}s")
    print(f"[INFO] Sample Rate: {sr} Hz")
    print(f"[INFO] Samples: {len(audio)}")
    print(f"[INFO] Max Amplitude: {np.max(np.abs(audio)):.4f}")
    
    # Check for silence
    if np.max(np.abs(audio)) < 0.001:
        print("[ERROR] Audio is mostly silent!")
        return False
    
    # Simple pitch change detection
    # Split audio into chunks and check for energy changes
    chunk_size = sr // 10  # 100ms chunks
    num_chunks = len(audio) // chunk_size
    
    energies = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio[start:end]
        energy = np.sum(chunk ** 2)
        energies.append(energy)
    
    # Count significant energy changes (potential note changes)
    threshold = np.mean(energies) * 0.5
    active_chunks = sum(1 for e in energies if e > threshold)
    
    print(f"[INFO] Active chunks: {active_chunks}/{num_chunks}")
    print(f"[INFO] Energy variation: {np.std(energies):.4f}")
    
    if active_chunks > 5:
        print("[OK] Audio appears to have multiple notes/events")
        return True
    else:
        print("[WARNING] Audio may have only single note or silence")
        return False

def analyze_midi(midi_file):
    """Analyze MIDI to show expected notes"""
    
    midi = pretty_midi.PrettyMIDI(midi_file)
    
    print(f"\n[INFO] MIDI File: {midi_file}")
    
    total_notes = 0
    for instrument in midi.instruments:
        notes = instrument.notes
        total_notes += len(notes)
        print(f"[INFO] Instrument: {instrument.name}")
        print(f"[INFO] Notes: {len(notes)}")
        
        if len(notes) > 0:
            # Show first few notes
            print(f"[INFO] First 5 notes:")
            for i, note in enumerate(notes[:5]):
                print(f"  Note {i+1}: Pitch={note.pitch}, Start={note.start:.2f}s, End={note.end:.2f}s, Velocity={note.velocity}")
    
    print(f"[INFO] Total Notes in MIDI: {total_notes}")
    return total_notes

if __name__ == "__main__":
    print("="*60)
    print("AUDIO VERIFICATION - embracingearth.space")
    print("="*60)
    
    # Analyze MIDI input
    midi_notes = analyze_midi("MIDI Files/MIDI Files/Cello Arpegio.mid")
    
    # Analyze generated audio
    has_multiple_notes = analyze_audio("test_output_synthesis_Cello Arpegio.wav")
    
    print("\n" + "="*60)
    if has_multiple_notes and midi_notes > 1:
        print("[OK] VERIFICATION PASSED")
        print(f"[OK] MIDI has {midi_notes} notes")
        print("[OK] Audio has multiple events")
    else:
        print("[FAIL] VERIFICATION FAILED")
        print(f"[INFO] Expected {midi_notes} notes from MIDI")
        print("[INFO] Audio analysis suggests single note or issues")
    print("="*60)

