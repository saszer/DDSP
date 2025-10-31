#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test HTTP endpoint end-to-end"""

import requests
import mido
import io
import struct
import sys

print("=" * 70)
print("HTTP ENDPOINT TEST - Full Workflow")
print("=" * 70)

# Create test MIDI
print("\n[1] Creating test MIDI file")
print("-" * 70)
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

# 3 notes: C4 (1s), D4 (1s), E4 (1s) = 3 seconds total
track.append(mido.Message('program_change', program=0, time=0))
track.append(mido.Message('note_on', note=60, velocity=80, time=0))
track.append(mido.Message('note_off', note=60, velocity=0, time=mid.ticks_per_beat * 2))
track.append(mido.Message('note_on', note=62, velocity=75, time=0))
track.append(mido.Message('note_off', note=62, velocity=0, time=mid.ticks_per_beat * 2))
track.append(mido.Message('note_on', note=64, velocity=85, time=0))
track.append(mido.Message('note_off', note=64, velocity=0, time=mid.ticks_per_beat * 2))

midi_bytes = io.BytesIO()
mid.save(file=midi_bytes)
midi_data = midi_bytes.getvalue()

print(f"Created MIDI: {len(midi_data)} bytes")
print(f"Expected duration: ~3 seconds")

# Test HTTP endpoint
print("\n[2] Testing HTTP /api/upload-midi endpoint")
print("-" * 70)

try:
    files = {'midi_file': ('test.mid', midi_data, 'audio/midi')}
    data = {'selected_model': 'cello_google_ddsp_model.pkl'}
    
    response = requests.post('http://localhost:8000/api/upload-midi', files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("[OK] Upload successful!")
        print(f"  Duration: {result.get('duration', 'N/A')}s")
        print(f"  File size: {result.get('file_size', 'N/A')} bytes")
        print(f"  Synthesis mode: {result.get('synthesis_mode', 'N/A')}")
        print(f"  Output file: {result.get('output_file', 'N/A')}")
        
        duration = result.get('duration', 0)
        if duration >= 2.5:
            print(f"\n[SUCCESS] Duration is correct: {duration:.2f}s")
        else:
            print(f"\n[ERROR] Duration too short: {duration:.2f}s (expected ~3s)")
            
        # Download and verify WAV
        if 'download_url' in result:
            wav_url = f"http://localhost:8000{result['download_url']}"
            wav_response = requests.get(wav_url)
            if wav_response.status_code == 200:
                wav_data = wav_response.content
                print(f"\n[3] Verifying WAV file")
                print("-" * 70)
                print(f"Downloaded WAV: {len(wav_data)} bytes")
                
                # Parse WAV header
                if len(wav_data) >= 44:
                    sample_rate = struct.unpack('<I', wav_data[24:28])[0]
                    subchunk2_size = struct.unpack('<I', wav_data[40:44])[0]
                    bits_per_sample = struct.unpack('<H', wav_data[34:36])[0]
                    bytes_per_sample = bits_per_sample // 8
                    num_channels = struct.unpack('<H', wav_data[22:24])[0]
                    
                    duration_samples = subchunk2_size // (bytes_per_sample * num_channels)
                    duration_seconds = duration_samples / sample_rate
                    
                    print(f"  Sample rate: {sample_rate} Hz")
                    print(f"  Duration: {duration_seconds:.2f}s ({duration_samples} samples)")
                    
                    if duration_seconds >= 2.5:
                        print(f"\n[SUCCESS] WAV file duration is correct!")
                    else:
                        print(f"\n[ERROR] WAV file duration too short: {duration_seconds:.2f}s")
            else:
                print(f"[ERROR] Could not download WAV: {wav_response.status_code}")
    else:
        print(f"[ERROR] Upload failed: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("[ERROR] Could not connect to server. Is it running on http://localhost:8000?")
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)

