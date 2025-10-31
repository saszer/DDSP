#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test full synthesis workflow to verify trained model usage and correct duration"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ddsp_server_hybrid import model_manager, Config
import mido
import io
import struct

print("=" * 70)
print("FULL SYNTHESIS TEST - Verifying Trained Model Usage")
print("=" * 70)

# Test 1: Verify model is loaded
print("\n[TEST 1] Model Loading Status")
print("-" * 70)
print(f"is_trained: {model_manager.is_trained}")
print(f"model: {model_manager.model}")
print(f"model_path: {model_manager.model_path}")
print(f"use_google_ddsp: {model_manager.use_google_ddsp}")
print(f"google_ddsp_data exists: {model_manager.google_ddsp.google_ddsp_data is not None}")

if model_manager.google_ddsp.google_ddsp_data:
    model_data = model_manager.google_ddsp.google_ddsp_data
    print(f"Model data type: {type(model_data)}")
    if isinstance(model_data, dict):
        print(f"Model keys: {list(model_data.keys())}")
        features = model_data.get('features', [])
        print(f"Number of trained features: {len(features)}")
        if features:
            print(f"Sample rate: {model_data.get('sample_rate', 'N/A')}")

# Test 2: Create a test MIDI with multiple notes at different times
print("\n[TEST 2] Creating Test MIDI")
print("-" * 70)
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

# Add notes: C4 (starts at 0s, lasts 1s), D4 (starts at 1s, lasts 1.5s), E4 (starts at 2s, lasts 0.5s)
# Total duration should be: 2.0 + 0.5 = 2.5 seconds
track.append(mido.Message('program_change', program=0, time=0))
track.append(mido.Message('note_on', note=60, velocity=80, time=0))  # C4 starts at 0
track.append(mido.Message('note_off', note=60, velocity=0, time=mid.ticks_per_beat * 2))  # 1 second (2 beats at 120 BPM)
track.append(mido.Message('note_on', note=62, velocity=75, time=0))  # D4 starts at 1s
track.append(mido.Message('note_off', note=62, velocity=0, time=mid.ticks_per_beat * 3))  # 1.5 seconds
track.append(mido.Message('note_on', note=64, velocity=85, time=0))  # E4 starts at 2.5s
track.append(mido.Message('note_off', note=64, velocity=0, time=mid.ticks_per_beat * 1))  # 0.5 seconds

midi_bytes = io.BytesIO()
mid.save(file=midi_bytes)
midi_data = midi_bytes.getvalue()

print(f"Created test MIDI: {len(midi_data)} bytes")
print(f"Expected total duration: ~2.5 seconds (last note ends at 2.5s)")

# Test 3: Parse MIDI
print("\n[TEST 3] MIDI Parsing")
print("-" * 70)
notes = model_manager._parse_midi_simple(midi_data)
print(f"Parsed {len(notes)} notes:")
total_duration = 0
for i, note in enumerate(notes):
    start = note.get('start', 'MISSING')
    duration = note.get('duration', 'MISSING')
    freq = note.get('frequency', 'MISSING')
    print(f"  Note {i}: start={start}, duration={duration}, freq={freq:.1f}Hz")
    if isinstance(start, (int, float)) and isinstance(duration, (int, float)):
        note_end = start + duration
        if note_end > total_duration:
            total_duration = note_end

print(f"Calculated total duration from notes: {total_duration:.2f}s")

# Test 4: Check which synthesis method will be used
print("\n[TEST 4] Synthesis Path Detection")
print("-" * 70)
if model_manager.is_trained and model_manager.model and model_manager.model_path:
    if 'google' in str(model_manager.model).lower():
        print("[OK] Will use Google DDSP model path")
        if model_manager.google_ddsp.google_ddsp_data:
            print("[OK] Model data is loaded - should use trained audio clips")
        else:
            print("[WARN] Model data NOT loaded - will fall back")
    else:
        print("[INFO] Will use custom trained model")
else:
    print("[WARN] No trained model - will use custom synthesis")

# Test 5: Synthesize audio
print("\n[TEST 5] Audio Synthesis")
print("-" * 70)
try:
    wav_data = model_manager.synthesize_audio(midi_data)
    
    if wav_data:
        print(f"[OK] Generated WAV: {len(wav_data)} bytes")
        
        # Parse WAV header to get actual duration
        if len(wav_data) >= 44:
            sample_rate = struct.unpack('<I', wav_data[24:28])[0]
            subchunk2_size = struct.unpack('<I', wav_data[40:44])[0]
            bits_per_sample = struct.unpack('<H', wav_data[34:36])[0]
            bytes_per_sample = bits_per_sample // 8
            num_channels = struct.unpack('<H', wav_data[22:24])[0]
            
            duration_samples = subchunk2_size // (bytes_per_sample * num_channels)
            duration_seconds = duration_samples / sample_rate
            
            print(f"WAV Header Info:")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Channels: {num_channels}")
            print(f"  Bits per sample: {bits_per_sample}")
            print(f"  Data size: {subchunk2_size} bytes")
            print(f"  Duration samples: {duration_samples}")
            print(f"  Duration seconds: {duration_seconds:.2f}s")
            
            # Verify duration is correct
            expected_duration = total_duration
            if abs(duration_seconds - expected_duration) < 0.5:
                print(f"\n[SUCCESS] Duration matches expected (~{expected_duration:.2f}s)")
            else:
                print(f"\n[ERROR] Duration mismatch!")
                print(f"  Expected: ~{expected_duration:.2f}s")
                print(f"  Actual: {duration_seconds:.2f}s")
                print(f"  Difference: {abs(duration_seconds - expected_duration):.2f}s")
        else:
            print("[ERROR] WAV file too small or invalid")
    else:
        print("[ERROR] Synthesis returned empty data")
        
except Exception as e:
    print(f"[ERROR] Synthesis failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
