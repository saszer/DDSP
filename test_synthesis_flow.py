import sys
sys.path.insert(0, 'h:\\DDSP')

from ddsp_server_hybrid import model_manager, Config
import pickle
import numpy as np

# Test 1: Check if model loaded
print("=" * 60)
print("TEST 1: Model Loading")
print("=" * 60)
print(f"is_trained: {model_manager.is_trained}")
print(f"model: {model_manager.model}")
print(f"model_path: {model_manager.model_path}")
print(f"use_google_ddsp: {model_manager.use_google_ddsp}")
print(f"google_ddsp.google_ddsp_data type: {type(model_manager.google_ddsp.google_ddsp_data)}")
if isinstance(model_manager.google_ddsp.google_ddsp_data, dict):
    print(f"Model keys: {list(model_manager.google_ddsp.google_ddsp_data.keys())}")
    print(f"Num features: {len(model_manager.google_ddsp.google_ddsp_data.get('features', []))}")

# Test 2: Parse a simple MIDI
print("\n" + "=" * 60)
print("TEST 2: MIDI Parsing")
print("=" * 60)

# Create a simple MIDI with mido
try:
    import mido
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Add notes: C4 (1s), D4 (1s), E4 (0.5s)
    track.append(mido.Message('program_change', program=0, time=0))
    track.append(mido.Message('note_on', note=60, velocity=80, time=0))  # C4
    track.append(mido.Message('note_off', note=60, velocity=0, time=mid.ticks_per_beat))  # 1 beat = 0.5s
    track.append(mido.Message('note_on', note=62, velocity=75, time=0))  # D4
    track.append(mido.Message('note_off', note=62, velocity=0, time=mid.ticks_per_beat))  # 1 beat = 0.5s
    track.append(mido.Message('note_on', note=64, velocity=85, time=0))  # E4
    track.append(mido.Message('note_off', note=64, velocity=0, time=int(mid.ticks_per_beat/2)))  # 0.5 beat = 0.25s
    
    # Save to bytes
    import io
    midi_bytes = io.BytesIO()
    mid.save(file=midi_bytes)
    midi_data = midi_bytes.getvalue()
    
    print(f"Created test MIDI: {len(midi_data)} bytes")
    
    # Parse it
    notes = model_manager._parse_midi_simple(midi_data)
    print(f"Parsed {len(notes)} notes:")
    for i, note in enumerate(notes):
        print(f"  Note {i}: freq={note['frequency']:.1f}Hz, vel={note['velocity']}, start={note.get('start', 'N/A')}, dur={note['duration']:.2f}s")
    
    # Calculate expected total duration
    if notes:
        max_end = max(note.get('start', 0) + note['duration'] for note in notes)
        print(f"Expected total duration: {max_end:.2f}s")
    
    # Test 3: Synthesize with trained model
    print("\n" + "=" * 60)
    print("TEST 3: Trained Audio Clip Synthesis")
    print("=" * 60)
    
    audio_list = model_manager._synthesize_with_trained_audio_clips(notes)
    if audio_list:
        print(f"[OK] Synthesized {len(audio_list)} samples = {len(audio_list)/Config.SAMPLE_RATE:.2f}s at {Config.SAMPLE_RATE}Hz")
    else:
        print("[FAIL] Synthesis returned None or empty")
    
    # Test 4: Full synthesis_audio call
    print("\n" + "=" * 60)
    print("TEST 4: Full synthesize_audio (MIDI bytes → WAV bytes)")
    print("=" * 60)
    
    wav_data = model_manager.synthesize_audio(midi_data)
    if wav_data:
        print(f"✅ Generated WAV: {len(wav_data)} bytes")
        # Extract duration from WAV header
        import struct
        if len(wav_data) >= 40:
            sample_rate = struct.unpack('<I', wav_data[24:28])[0]
            subchunk2_size = struct.unpack('<I', wav_data[40:44])[0]
            duration_samples = subchunk2_size // 3  # 24-bit = 3 bytes
            duration_seconds = duration_samples / sample_rate
            print(f"   WAV header says: {sample_rate}Hz, {duration_samples} samples = {duration_seconds:.2f}s")
    else:
        print("❌ WAV generation failed")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
