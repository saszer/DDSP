#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose MIDI file to see why audio might be 1 second"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ddsp_server_hybrid import model_manager
import sys

if len(sys.argv) < 2:
    print("Usage: python diagnose_midi.py <path_to_midi_file>")
    print("\nExample: python diagnose_midi.py myfile.mid")
    sys.exit(1)

midi_path = sys.argv[1]

if not os.path.exists(midi_path):
    print(f"Error: File not found: {midi_path}")
    sys.exit(1)

print("=" * 70)
print(f"MIDI FILE DIAGNOSIS: {midi_path}")
print("=" * 70)

# Read MIDI file
with open(midi_path, 'rb') as f:
    midi_data = f.read()

print(f"\nFile size: {len(midi_data)} bytes")

# Parse it
print("\n[PARSING MIDI]")
print("-" * 70)
notes = model_manager._parse_midi_simple(midi_data)

if not notes:
    print("[ERROR] No notes found in MIDI file!")
    print("This will result in silence or fallback audio.")
    sys.exit(1)

print(f"Found {len(notes)} notes:")
print()

total_duration = 0
all_have_start = True
all_have_duration = True

for i, note in enumerate(notes):
    start = note.get('start', None)
    duration = note.get('duration', None)
    freq = note.get('frequency', 'N/A')
    
    if start is None:
        all_have_start = False
        start = 'MISSING'
    if duration is None:
        all_have_duration = False
        duration = 'MISSING'
    
    print(f"  Note {i}:")
    print(f"    Start time: {start}")
    print(f"    Duration: {duration}")
    print(f"    Frequency: {freq}")
    
    if isinstance(start, (int, float)) and isinstance(duration, (int, float)):
        note_end = start + duration
        if note_end > total_duration:
            total_duration = note_end
        print(f"    End time: {note_end:.2f}s")
    print()

print("=" * 70)
print("DIAGNOSIS:")
print("=" * 70)

if not all_have_start:
    print("[ISSUE] Some notes are missing 'start' times!")
    print("  This will cause notes to overlap or be placed incorrectly.")
    print("  Fix: MIDI parser needs to extract start times from note_on events.")
    
if not all_have_duration:
    print("[ISSUE] Some notes are missing 'duration'!")
    print("  This will cause short or zero-length notes.")
    
if total_duration > 0:
    print(f"[OK] Calculated total duration: {total_duration:.2f} seconds")
    if total_duration < 1.5:
        print(f"[WARNING] Duration is very short ({total_duration:.2f}s)")
        print("  This might indicate:")
        print("    - MIDI file only has very short notes")
        print("    - Note durations aren't being parsed correctly")
        print("    - All notes are overlapping at time 0")
else:
    print("[ERROR] Could not calculate duration from notes!")

print()
print("Expected synthesis result:")
print(f"  If using trained audio clips: ~{total_duration:.2f}s")
print(f"  If falling back to custom: sum of durations = {sum(note.get('duration', 0) for note in notes):.2f}s")

print("\n" + "=" * 70)

