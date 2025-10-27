#!/usr/bin/env python3
"""
Test MIDI Upload with Real File
"""

import requests
import os

def test_real_midi():
    midi_file_path = "MIDI Files/MIDI Files/Cello Arpegio.mid"
    
    if not os.path.exists(midi_file_path):
        print(f"MIDI file not found: {midi_file_path}")
        return
    
    print(f"Uploading real MIDI file: {midi_file_path}")
    
    try:
        with open(midi_file_path, 'rb') as f:
            files = {'midi_file': ('Cello Arpegio.mid', f, 'audio/midi')}
            response = requests.post('http://localhost:8000/api/upload-midi', files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"SUCCESS!")
            print(f"Duration: {result.get('duration', 0):.2f} seconds")
            print(f"Output file: {result.get('output_file', 'N/A')}")
            print(f"Quality: {result.get('quality_level', 'N/A')}")
        else:
            print(f"Upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    test_real_midi()
