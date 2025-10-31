#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test HTTP upload workflow to simulate actual frontend request"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ddsp_server_hybrid import model_manager, Config
import mido
import io
import struct

print("=" * 70)
print("HTTP WORKFLOW TEST - Simulating Frontend Upload")
print("=" * 70)

# Create a realistic MIDI file with multiple notes
print("\n[STEP 1] Creating MIDI file")
print("-" * 70)
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

# Create a melody: 4 notes over 4 seconds
track.append(mido.Message('program_change', program=0, time=0))
track.append(mido.Message('note_on', note=60, velocity=80, time=0))  # C4
track.append(mido.Message('note_off', note=60, velocity=0, time=mid.ticks_per_beat * 2))  # 1s
track.append(mido.Message('note_on', note=62, velocity=75, time=0))  # D4
track.append(mido.Message('note_off', note=62, velocity=0, time=mid.ticks_per_beat * 2))  # 1s  
track.append(mido.Message('note_on', note=64, velocity=85, time=0))  # E4
track.append(mido.Message('note_off', note=64, velocity=0, time=mid.ticks_per_beat * 2))  # 1s
track.append(mido.Message('note_on', note=65, velocity=90, time=0))  # F4
track.append(mido.Message('note_off', note=65, velocity=0, time=mid.ticks_per_beat * 2))  # 1s

midi_bytes = io.BytesIO()
mid.save(file=midi_bytes)
midi_data = midi_bytes.getvalue()

print(f"Created MIDI: {len(midi_data)} bytes")
print(f"Expected duration: ~4 seconds (4 notes × 1s each)")

# Simulate the exact workflow from synthesize_audio
print("\n[STEP 2] Parsing MIDI (as synthesize_audio does)")
print("-" * 70)
notes = model_manager._parse_midi_simple(midi_data)
print(f"Parsed {len(notes)} notes:")
for i, note in enumerate(notes):
    print(f"  Note {i}: start={note.get('start', 'MISSING')}, duration={note.get('duration', 'MISSING')}, freq={note.get('frequency', 'MISSING'):.1f}Hz")

max_end = max((note.get('start', 0) + note.get('duration', 0) for note in notes), default=0)
print(f"Max end time: {max_end:.2f}s")

# Test synthesis
print("\n[STEP 3] Synthesizing (as HTTP handler does)")
print("-" * 70)
wav_data = model_manager.synthesize_audio(midi_data)

if wav_data:
    # Parse WAV
    sample_rate = struct.unpack('<I', wav_data[24:28])[0]
    subchunk2_size = struct.unpack('<I', wav_data[40:44])[0]
    bits_per_sample = struct.unpack('<H', wav_data[34:36])[0]
    bytes_per_sample = bits_per_sample // 8
    num_channels = struct.unpack('<H', wav_data[22:24])[0]
    
    duration_samples = subchunk2_size // (bytes_per_sample * num_channels)
    duration_seconds = duration_samples / sample_rate
    
    print(f"[RESULT] WAV generated:")
    print(f"  File size: {len(wav_data)} bytes")
    print(f"  Duration: {duration_seconds:.2f}s ({duration_samples} samples)")
    print(f"  Sample rate: {sample_rate} Hz")
    
    if duration_seconds >= 3.5:
        print(f"\n[SUCCESS] Duration is correct (>= 3.5s for 4-second MIDI)")
    else:
        print(f"\n[ERROR] Duration too short! Expected ~4s, got {duration_seconds:.2f}s")
        print(f"  This indicates a problem with note timing or synthesis")
else:
    print("[ERROR] No WAV data generated")

print("\n" + "=" * 70)

